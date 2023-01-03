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
using CSV

using JuMC: load_lxcat, CollisionTable, Population, ElectronState, MultiPopulation, nparticles,
    VoidCollisionTracker, advance!, eachactive, energy, presample, rate, instantiate, nactives,
    add_particle!, remove_particle!
import JuMC

const DATA_DIR = joinpath(@__DIR__, "..", "data")

struct FixedField{T}
    ez::T
end

(f::FixedField)(x) = @SVector [0, 0, f.ez]

function main(;nsamples=10, en=10 .^ range(-1, 3, 100), kw...)
    dfs = map(_ -> swarm(en; kw...), 1:nsamples)
    st = stats(dfs)
    return st
end


function swarm(en;
               cross_sections=[joinpath(DATA_DIR, "LXCat-June2013.json")],
               dt=1e-13,
               trelax=2e-9,
               maxp=10^7,
               resampling=true,
               n0=10_000)
    densities = Dict("N2" => co.nair * 0.79,
                     "O2" => co.nair * 0.21)
    
    # Create the energy grid in eV; all cross-sections are interpolated into
    # this grid
    energy = LinRange(0, 1000 * co.eV, 100_000)
    
    (;proc, rate, maxrate, origperm) = load_lxcat(cross_sections, densities, energy)
    ecolls = CollisionTable(proc, energy, rate, maxrate)
        
    @info "max collision rate" maxrate
    T = Float64
    v0 = @SVector zeros(T, 3)
    x0 = @SVector zeros(T, 3)
    nt = ceil(Int, trelax / dt)

    tracker = VoidCollisionTracker()

    mobility = Float64[]
    diffusion_l = Float64[]
    diffusion_t = Float64[]
    
    k = [Float64[] for _ in eachindex(ecolls.proc)]
    progmeter = Progress(length(en))
    t = (0:nt - 1) .* dt
    
    for ien in en        
        @info "Electric field" ien
        # Initialize particle population at each step.  This ensures statistical independence.
        init_particles = map(1:n0) do _
            ElectronState(x0, v0, 1.0)
        end
        
        population_index = Pair{Symbol, Any}[:electron => Population(maxp, init_particles, ecolls)]
        mpopl = MultiPopulation(population_index...)
        
        electron = mpopl.index.electron
        colls = mpopl.index.electron.collisions
        
        eb = ien * co.nair * co.Td
        efield = FixedField(eb)

        sigma2 = zeros(3, nt)

        # Number of time steps to compute diffusion coefficient
        l = 50
        
        local i
        for outer i in 1:nt
            advance!(mpopl, efield, dt, tracker)

            X = mean(p -> p.x, eachactive(electron))
            X2 = mean(p -> p.x .^ 2, eachactive(electron))

            S = X2 .- X .^ 2
            sigma2[:, i] .= S
            if resampling
                resample!(electron, n0)
            end
            if nparticles(electron) > 0.9 * maxp
                break
            end
        end
        rl = [ones(l) t[i - l + 1:i]] \ sigma2[3, (i - l + 1:i)]
        push!(diffusion_l, rl[2] / 2)
        
        rt = [ones(l) t[i - l + 1:i]] \ (sigma2[1, (i - l + 1:i)] .+ sigma2[2, (i - l + 1:i)])
        push!(diffusion_t, rt[2] / 4)

        W = mean(p -> p.v[3], eachactive(electron))
        push!(mobility, -W / eb)
        
        # Reaction rates:
        k1 = mean(p -> collrate(colls, p), eachactive(electron))
        
        for i in eachindex(colls.proc)
            push!(k[i], k1[origperm[i]])
        end

        ProgressMeter.next!(progmeter,
                            showvalues=[("E/n (Td)", ien)])
    end
        
    # Because of the different ordering wrt lxcat output we use this trick to ensure same
    # indexing of the rates
    knames = [sprintf1("k%03d", (((i - 1) + 25) % 42) + 1) for i in 1:(length(ecolls.proc) - 1)]
    df = DataFrame("en" => en, "mobility" => mobility,
                   "diffusion_l" => diffusion_l, "diffusion_t" => diffusion_t,
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


end # module Swarm
