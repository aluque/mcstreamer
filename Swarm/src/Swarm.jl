module Swarm

using StaticArrays
using Statistics
using Constants: co
using LinearAlgebra
using DataFrames
using Formatting
using StructArrays
using ProgressMeter
using Distributions
using HypothesisTests
using CSV
using Dierckx
using Dates
import PyPlot as plt
using Glob
using Printf

using JuMC: load_lxcat, CollisionTable, Population, ElectronState, MultiPopulation, nparticles,
    VoidCollisionTracker, advance!, eachactive, presample, rate, instantiate, nactives,
    add_particle!, remove_particle!, energy
import JuMC

const DATA_DIR = joinpath(@__DIR__, "..", "data")
const NAIR_STP = co.bar / (co.k * 273.15)

struct FixedField{T}
    ez::T
end

(f::FixedField)(x) = @SVector [0, 0, f.ez]

function main(;nsamples=10, en=10 .^ range(-1, 3, 100), kw...)
    @info  """Note: Sometimes the HypothesisTests package complains about ties in the
    KS test.  I believe this is because at early times several particles may not have
    experienced any collision and thus have identical energies.
    """

    dfs = map(_ -> swarm(en; kw...), 1:nsamples)
    st = stats(dfs)
    sm = smoothfit(st)
    return (;st, sm)
    
end


function swarm(en;
               cross_sections=[joinpath(DATA_DIR, "LXCat-June2013.json")],
               trelax=2e-9,
               maxp=10^7,

               # Number of iterations between two KS tests.
               ntest=1000,

               # Highest p-value in the KS test to stop
               pval=0.75,
               
               resampling=true,
               n0=10_000)
    densities = Dict("N2" => NAIR_STP * 0.79,
                     "O2" => NAIR_STP * 0.21)
    
    # Create the energy grid in eV; all cross-sections are interpolated into
    # this grid
    energy = LinRange(0, 1000 * co.eV, 100_000)
    
    (;proc, rate, maxrate, origperm) = load_lxcat(cross_sections, densities, energy)
    ecolls = CollisionTable(proc, energy, rate, maxrate)

    # Particle gain for each process
    g = gains(proc)
    gplus = max.(0, g)
    gmin = -min.(0, g)
    
    dt = 1 / maxrate / 4
    nt = ceil(Int, trelax / dt)
    t = (0:nt - 1) .* dt

    @info "max collision rate" maxrate
    
    # initial state - same in all simulations
    T = Float64
    v0 = @SVector zeros(T, 3)
    x0 = @SVector zeros(T, 3)

    tracker = VoidCollisionTracker()

    # Initialize columns for output
    mobility = Float64[]
    diffusion = Float64[]
    diffusion_l = Float64[]
    diffusion_t = Float64[]    
    nu = Float64[]
    alpha = Float64[]
    eta = Float64[]
    
    k = [Float64[] for _ in eachindex(ecolls.proc)]

    
    progmeter = Progress(length(en))    
    for ien in en
        # Initialize particle population at each step.  This ensures statistical independence.
        init_particles = map(1:n0) do _
            ElectronState(x0, v0, 1.0)
        end
        
        population_index = Pair{Symbol, Any}[:electron => Population(maxp, init_particles, ecolls)]
        mpopl = MultiPopulation(population_index...)
        
        electron = mpopl.index.electron
        colls = mpopl.index.electron.collisions
        
        eb = ien * NAIR_STP * co.Td
        efield = FixedField(eb)

        # Number of time steps to compute diffusion coefficient
        l = 100
        
        sigma2 = zeros(nt, 3)
        kst = KSTester(Float64)
        
        local i
        for outer i in 1:nt
            advance!(mpopl, efield, dt, tracker)
            X = mean(p -> p.x, eachactive(electron))
            X2 = mean(p -> p.x .^ 2, eachactive(electron))
            
            S = X2 .- X .^ 2
            sigma2[i, :] .= S

            if (i % ntest) == 0
                kstest(kst, electron, pval) && break
            end
            
            if resampling
                resample!(electron, n0)
            end
            if nparticles(electron) > 0.9 * maxp
                break
            end
        end

        a = [ones(l) t[i - l + 1:i]] \ sigma2[i - l + 1:i, :]
        # The three diagonal elements of the diffusion matrix
        D = a[2, :] ./ 2

        push!(diffusion_t, (D[1] + D[2]) / 2)
        push!(diffusion_l, D[3])
        push!(diffusion, (D[1] + D[2] + D[3]) / 3)
        
        W = mean(p -> p.v[3], eachactive(electron))
        push!(mobility, -W / eb)
        
        # Reaction rates:
        k1 = mean(p -> collrate(colls, p), eachactive(electron))

        # net gain rate
        nu1 = g' * k1
        alpha1 = gplus' * k1 / abs(W)
        eta1 = gmin' * k1 / abs(W)
        
        push!(nu, nu1)
        push!(alpha, alpha1)
        push!(eta, eta1)
        
        for i in eachindex(colls.proc)
            push!(k[i], k1[origperm[i]])
        end

        ProgressMeter.next!(progmeter,
                            showvalues=[("E/n (Td)", ien)])
    end
        
    # Because of the different ordering wrt lxcat output we use this trick to ensure same
    # indexing of the rates
    knames = [sprintf1("k%03d", (((i - 1) + 25) % 42) + 1) for i in 1:(length(ecolls.proc) - 1)]
    efield = en .* co.Td .* NAIR_STP
    df = DataFrame("en" => en, "mobility" => mobility,
                   "diffusion_l" => diffusion_l, "diffusion_t" => diffusion_t,
                   "diffusion" => diffusion, "nu" => nu, "alpha" => alpha,
                   "eta" => eta,
                   map((nam, ki) -> nam => ki, knames, k[begin:end-1])...)

    return df
end

"""
    Resample the population `popl` using  Russian roulette and splitting to 
    approximate the target number of particles `n`.  Only does anything if the initial number
    is farther than `tol` from the target.
"""
function resample!(popl, n, tol=0.1)
    ncur = nactives(popl)
    #@show ncur
    if abs((ncur - n) / n) < tol
        return nothing
    end
    
    if ncur > n
        for i in 1:popl.n[]
            k = LazyRow(popl.particles, i)
            k.active || continue

            # Probability of dropping = n / ncur
            if rand() < (ncur - n) / ncur
                remove_particle!(popl, i)
            end
        end
    else
        for i in 1:popl.n[]
            k = LazyRow(popl.particles, i)
            k.active || continue

            # Probability of adding a new particle ncur / n
            nnew = rand(Poisson((n - ncur) / ncur))
            for j in 1:nnew
                add_particle!(popl, popl.particles[i])
            end
        end
    end   
    nothing
end


"""
   Return a vector of particle gains for each process in the process table proc.
"""
function gains(proc)
    collect(map(proc) do p
        if (p isa JuMC.Attachment)
            return -1
        elseif (p isa JuMC.Ionization)
            return 1
        else
            return 0
        end
    end)
end

""" Collision rates of particle p """
function collrate(c::CollisionTable, p)
    p1 = instantiate(p)
    eng = energy(p1)
    pre = presample(c, p1, eng)

    SVector(ntuple(i -> rate(c, i, pre)::Float64, Val(length(c))))
end

"""
    Compute statistics (mean, std) of a set of DataFrames.
"""
function stats(dfs)
    @assert allequal(Set(names(df)) for df in dfs)

    t = map(names(dfs[1])) do nam
        m = mean(df -> df[!, nam], dfs)
        # No first-arg function for std
        s = std(map(df -> df[!, nam], dfs)) ./ sqrt(length(dfs))
                  
        if !(nam in ["en"])
            [nam => m, "$(nam)_err" => s]
        else
            [nam => m]
        end
    end

    DataFrame(vcat(t...))
end

##
## Kolmogorov-Smirnov tests
##
mutable struct KSTester{T}
    eng::Vector{T}
    eng1::Vector{T}
end

function KSTester(T)
    KSTester(T[], T[])
end

"""
    Tests whether the current energy distribution of `popl` is statistically different from a previous
    distribution.  If they cannot be differentiated (p > pval), returns true.

"""
function kstest(kst::KSTester, popl, pval)
    (;eng, eng1) = kst
    res = false
    empty!(eng)
    for p in eachactive(popl)
        push!(eng, JuMC.energy(instantiate(p)))
    end
    
    if length(eng1) > 0
        ks = ApproximateTwoSampleKSTest(eng, eng1)
        if pvalue(ks) > pval
            @info "Stop iterations because KS test"
            res = true
        end
    end
    kst.eng1, kst.eng = eng, eng1

    return res
end


"""
    Builds smooth splines fitting the data of the DataFrame `df`.
"""
function smoothfit(df)
    fields = filter(s -> s != "en" && !endswith(s, "_err"), names(df))
    en1 = 10 .^ (-1:0.02:3)
    nonlog = ["nu"]
    
    cols = map(fields) do f
        if !(f in nonlog)
            flt = df[!, f] .> 0
        
            spl = Spline1D(log.(df.en[flt]), log.(df[flt, f]); w=(df[flt, f] ./ df[flt, f * "_err"]),
                           s=length(df.en), bc="nearest")
            y = exp.(spl.(log.(en1)))
        else
            spl = Spline1D(df.en, df[!, f]; w=(1 ./ df[!, f * "_err"]),
                           s=length(df.en), bc="nearest")
            y = spl.(en1)
        end
        return f => y
    end

    return DataFrame("en" => en1, cols...)    
end

"""
    Compare data with a bolsig+ output"
"""
function cmpswarm(df, bspath; only=nothing)
    vars = map(f -> splitext(f)[1], filter(endswith(".dat"), readdir(bspath)))
    nrm = Dict("mobility" => NAIR_STP, "diffusion" => NAIR_STP)
    
    for v in vars
        if v in names(df)
            if (!isnothing(only) && !(v in only))
                continue
            end
            
            b = DataFrame(CSV.File(joinpath(bspath, v * ".dat"), header=["en", v], comment="#"))

            plt.figure(v)
            plt.plot(df.en, df[!, v] * get(nrm, v, 1 / NAIR_STP), label="MC")
            plt.plot(b.en, b[!, v], label="Bolsig+")
            plt.loglog()
        end
    end
end

"""
Generate output in the afivo format.
"""
function afivo(fout::IOStream, df)
    println(fout, "# Generated by Swarm.jl on $(now())")
    println(fout, "# nair = $(NAIR_STP)")
    println(fout)
    
    efield = df.en .* co.Td .* NAIR_STP
    items = [("efield[V/m]_vs_mu[m2/Vs]", df.mobility),
             ("efield[V/m]_vs_dif[m2/s]", df.diffusion),
             ("efield[V/m]_vs_alpha[1/m]", df.alpha),
             ("efield[V/m]_vs_eta[1/m]", df.eta)]
    for (label, col) in items
        println(fout, label)
        println(fout, "-" ^ 23)
        for i in eachindex(efield)
            @printf(fout, " %.3e  %.3e\n", efield[i], col[i])
        end
        println(fout, "-" ^ 23)
        println(fout)
    end    
end

afivo(fout::String, df) = open(io -> afivo(io, df), fout, "w")


end # module Swarm
