## streamer.jl : Thu Jul 22 12:29:35 2021
## Streamer with autoencoder de-noising.

module Streamer
using StaticArrays
using StatsBase: sample, pweights
using Constants: co
using UnPack
using Multigrid
using OffsetArrays
import PyPlot as plt
using LaTeXStrings
using JLD2
using CodecZlib
using Formatting
using Logging
using Dates
using TOML

using JuMC: Population, CollisionTable, load_lxcat, init!, advance!, collisions!,
    Electron, add_particle!
using JuMC: collrates, repack!, meanenergy, weight, actives, CollisionPopulation,
    IonizationOutcome, AttachmentOutcome, AbstractOutcome, Particle
import JuMC: track, AbstractCollisionTracker
using BenchmarkTools

include("timesteps.jl")
include("autoencoder.jl")
using .Autoencoder: Denoiser, denoise

function logfmt(level, _module, group, id, file, line)
    return (:blue, format("{:<23}:", string(Dates.now())), "")
end


function start()
    logger = ConsoleLogger(meta_formatter=logfmt)
    with_logger(logger) do
        main()
    end
end

function main(finput=ARGS[1]; debug=false)
    @info "Starting simulation"
    poisson_save_n[] = 0
    
    input = TOML.parsefile(finput)
    
    L::Float64 = input["domain"]["L"]
    R::Float64 = input["domain"]["R"]
    
    n::Int = input["n"]
    
    densities = Dict("N2" => co.nair * 0.79,
                     "O2" => co.nair * 0.21)
    
    # Create the energy grid in eV; all cross-sections are interpolated into
    # this grid
    #energy = (0:0.01:1000) .* co.eV
    energy = LinRange(0, 1000 * co.eV, 100_000)
    
    csfile = joinpath(@__DIR__, "LXCat-June2013.json")
    proc, rate, maxrate = load_lxcat(csfile, densities, energy)
    ecolls = CollisionTable(proc, energy, rate, maxrate)
    
    Δt = 10^floor(log10(1 / (2 * ecolls.maxrate)))
    @info "Time step derived from collision rate" Δt

    tmax = input["tmax"]

    eb::Float64 = -input["eb"] * co.Td * co.nair
    
    T = Float64
    maxp::Int = input["maxp"]
    
    # Initialize the electron population
    epop = Population(Electron(),
                      n,
                      zeros(Bool, maxp),
                      zeros(SVector{3, T}, maxp),
                      zeros(SVector{3, T}, maxp),
                      ones(T, maxp),
                      zeros(Int, maxp))
    
    w::Float64 = input["seed"]["w"]
    z0::Float64 = input["seed"]["z0"]
    for i in eachindex(epop)
        epop.x[i] = @SVector [w * randn(), w * randn(), z0 + w * randn()]
    end
    
    pind = (electron = CollisionPopulation(ecolls, epop),)
    init!(pind, Electron(), Δt)

    M::Int = input["domain"]["M"]
    N::Int = input["domain"]["N"]
    
    grid = Grid(R, L, M, N)
    fields = GridFields(grid)
    density!(grid, fields.qfixed, pind, Electron(), 1.0)
    
    
    # Poisson
    bc = ((LeftBnd(), Dirichlet()),
          (RightBnd(), Dirichlet()),
          (TopBnd(), Neumann()),
          (BottomBnd(), Neumann()))
    
    mg = MGConfig(bc=bc, s=co.elementary_charge * dz(grid)^2 / co.ϵ0,
                  conn=Multigrid.CylindricalConnector{1}(),
                  levels=9,
                  tolerance=1.0e8,
                  smooth1=2,
                  smooth2=2,
                  verbosity=0,
                  g=1)
    
    ws = Multigrid.allocate(mg, parent(fields.q));
    
    efield = FieldInterp(fields)
    Δt_poisson, Δt_output, Δt_resample = map(v -> input["interval"][v],
                                             ("poisson", "output", "resample"))
    
    outpath = splitext(abspath(finput))[1]
    isdir(outpath) || mkdir(outpath)

    maxc::Int = input["maxc"]

    denoiser = Denoiser(input["denoise"]["model"], (0.0, 1.0),
                        Tuple(input["denoise"]["log10range"]))
    
    if debug
        return (;pind, efield, eb, Δt, fields, mg, ws)
    end

    nsteps(pind, Int(fld(tmax, Δt)), maxc, efield, eb, Δt,
           Δt_poisson, Δt_output, Δt_resample,
           fields, mg, ws, denoiser; outpath)
    
    # k = collrates(pind, Electron())
    density!(grid, fields.qpart, pind, Electron(), -1.0)

    (;grid, pind, efield, Δt, fields)
end


"""
    Advance the simulation by `n` steps.

    Parameters:
    * `pind` a particle population.
    * `n`: number of steps.
    * `maxc`: Max. particles per cell.
    * `efield`: Electric field interpolator.    
    * `eb`: Background electric field.
    * `Δt`: Time-step at which the particles are updated.
    * `Δt_poisson`, `Δt_output`, `Δt_resample`: Time steps for updatting
      the electrostatic field, outputting data and resampling particles.
    * `fields`: A GridFields with space for all fields.
    * `mg`: A Multigrid (MG) solver.
    * `ws`: Working space for the MG solver.
    * `denoiser`: a Denoiser
"""
function nsteps(pind, n, maxc, efield, eb, Δt, Δt_poisson, Δt_output, Δt_resample,
                fields, mg, ws, denoiser; outpath="")

    ecolls = pind.electron.colls
    epop = pind.electron.pop
    tracker = CollisionTracker(fields)

    poisson = TimeStepper(Δt_poisson)
    output = TimeStepper(Δt_output)
    resample = TimeStepper(Δt, Δt_resample)

    # Measure time used in different work
    elapsed_poisson = 0.0
    elapsed_advance = 0.0
    elapsed_collisions = 0.0
    elapsed_resample = 0.0
    
    
    for i in 1:n
        t = (i - 1) * Δt
        
        atstep(poisson, t) do _
            elapsed_poisson += @elapsed poisson!(fields, pind, eb, mg, ws,
                                                 denoiser, outpath)
        end
        
        atstep(output, t) do j
            # Use true here to enable compression.
            jldsave(joinpath(outpath, fmt("04d", j) * ".jld"), false; fields)
            
            active_superparticles = actives(epop)
            physical_particles = weight(epop)
            @info "$(j * Δt_output * 1e9) ns"  active_superparticles physical_particles
            @info "Elapsed times" elapsed_poisson elapsed_advance elapsed_collisions elapsed_resample
        end
        
        
        elapsed_advance += @elapsed advance!(pind, Electron(), efield, Δt)
        elapsed_collisions += @elapsed collisions!(pind, Electron(),  Δt, tracker)

        atstep(resample, t) do _
            elapsed_resample += @elapsed begin
                resample!(pind, fields, Electron(), maxc)
                repack!(epop)
                shuffle!(epop)
            end
        end
        
        for part in keys(pind)            
            @unpack pop = pind[part]
            # Repack always to allow LoopVectorization in advance!
            repack!(pop)
        end
    end
end

struct Grid{T}
    R::T
    L::T
    
    M::Int
    N::Int

    rc::LinRange{T, Int}
    rf::LinRange{T, Int}
    zc::LinRange{T, Int}
    zf::LinRange{T, Int}

    function Grid(R::T, L::T, M, N) where T
        rf = LinRange(0, R, M + 1)
        zf = LinRange(0, L, N + 1)
        
        rc = 0.5 * (rf[(begin + 1):end] + rf[begin:(end - 1)])
        zc = 0.5 * (zf[(begin + 1):end] + zf[begin:(end - 1)])

        new{T}(R, L, M, N, rc, rf, zc, zf)
    end
end

@inline function inside(grid::Grid, x)
    @unpack rf, zf = grid
    r = rcyl(x)
    (rf[1] < r < rf[end]) && (zf[1] < x[3] < zf[end])
end

"""
    Allocate an array for a field evaluated at cell centers of a `grid`, 
    including space for `g` ghost cells.
"""
function calloc_centers(T, grid::Grid, g::Int=1)
    arr = zeros(T, (1 - g):(grid.M + g), (1 - g):(grid.N + g))
end

calloc_centers(grid::Grid{T}, g::Int=1) where T = calloc_centers(T, grid, g)

function calloc_centers_threads(T, grid::Grid, g::Int=1)
    arr = zeros(T, (1 - g):(grid.M + g), (1 - g):(grid.N + g),
                Threads.nthreads())
end

calloc_centers_threads(grid::Grid{T}, g::Int=1) where T =
    calloc_centers_threads(T, grid, g)

# Face-centered fields always have size (M + 1) x (N + 1), even if sometimes
# one row/column is unused.  They also do not have ghost cells.
calloc_faces(T, grid::Grid, g::Int=1) = zeros(T, (1 - g):(grid.M + g + 1),
                                              (1 - g):(grid.N + g + 1))

dr(grid::Grid) = step(grid.rc)
dz(grid::Grid) = step(grid.zc)
dV(grid::Grid, i) = 2π * grid.rc[i] * dr(grid) * dz(grid)
rcyl(x) = sqrt(x[1]^2 + x[2]^2)


"""
    Index (CartesianIndex) of location `r` inside `grid`.
"""
function cellindex(grid, r)
    ρ = sqrt(r[1]^2 + r[2]^2)

    i = Int(fld(ρ - first(grid.rf), step(grid.rf))) + 1
    j = Int(fld(r[3] - first(grid.zf), step(grid.zf))) + 1
    CartesianIndex(i, j)
end


"""
    Same as `cellindext(grid, r)` but adds an index with the current thread id
"""
function cellindext(grid, r)
    I = cellindex(grid, r)
    
    CartesianIndex(I[1], I[2], Threads.threadid())
end


"""
    A container for all fields located on a grid.

    Some fields (of type `A1`) contain an extra dimension for the thread id, 
    which allows to update in parallel.
"""
struct GridFields{T,A1<:AbstractArray{T},A<:AbstractArray{T},AI<:AbstractArray{Int}}
    grid::Grid{T}
    
    # Fixed charges
    qfixed::A1

    # Charges associated with mobile particles.
    qpart::A1

    # Charge density
    q::A

    # Electrostatic potential
    u::A
    
    # r-component of the electric field
    er::A

    # z-component of the electric field
    ez::A

    # For Russian roulette; counter of particles inside each cell.
    p::AI

    # For Russian roulette; total weight of the discarded particles
    w::A
    
    """ Allocate fields for a grid `grid`. """
    function GridFields(grid::Grid{T}) where T
        qfixed = calloc_centers_threads(T, grid)
        qpart = calloc_centers_threads(T, grid)
        q = calloc_centers(T, grid)
        u = calloc_centers(T, grid)
        c = calloc_centers(T, grid)
        er = calloc_faces(T, grid)
        ez = calloc_faces(T, grid)
        p = calloc_faces(Int, grid)
        
        new{T,typeof(qfixed),typeof(q),typeof(p)}(grid, qfixed, qpart, q, u, er, ez, c, p)
    end
end

"""
    A callable to use for field interpolations.
"""
struct FieldInterp{GF <: GridFields}
    fields::GF
end


"""
    Interpolate the electric field at a given position `x`.
    Uses bi-linear interpolation of each of the r/z-components.
"""
function (fieldinterp::FieldInterp)(x)
    inside(fieldinterp.fields.grid, x) || return zero(SVector{3, eltype(x)})
    r = sqrt(x[1]^2 + x[2]^2)
    
    grid = fieldinterp.fields.grid
    fields = fieldinterp.fields
    @unpack er, ez = fields
    
    # Not staggered
    ifl, δr = divrem(r - first(grid.rf), dr(grid))
    i = Int(ifl) + 1
    jfl, δz = divrem(x[3] - first(grid.zf), dz(grid))
    j = Int(jfl) + 1
    
    # Staggered; Note that here the lowest index is 0 so we don't add 1
    i1fl, δr1 = divrem(r - first(grid.rf) + 0.5 * dr(grid), dr(grid))    
    i1 = Int(i1fl)
    
    j1fl, δz1 = divrem(x[3] - first(grid.zf) + 0.5 * dz(grid), dz(grid))
    j1 = Int(j1fl)
    
    # Normalize to 0..1
    δr /= dr(grid)
    δz /= dz(grid)
    δr1 /= dr(grid)
    δz1 /= dz(grid)
    
    # er
    er1 = (er[i, j1]     * (1 - δr) * (1 - δz1) + er[i + 1, j1]     * δr * (1 - δz1) +
           er[i, j1 + 1] * (1 - δr) * δz1       + er[i + 1, j1 + 1] * δr * δz1)
    
    ez1 = (ez[i1, j]     * (1 - δr1) * (1 - δz) + ez[i1 + 1, j]     * δr1 * (1 - δz) +
           ez[i1, j + 1] * (1 - δr1) * δz       + ez[i1 + 1, j + 1] * δr1 * δz)
    
    # Return early and avoid 0/0
    r == 0 && return @SVector [zero(r), zero(r), ez1]
    
    
    efield = @SVector [er1 * x[1] / r, er1 * x[2] / r, ez1]
    return efield
end


"""
    A tracker is called whenerever a particle experiences a collision.
    This is useful for example to keep track of fixed charges left at the
    collision location.
"""
struct CollisionTracker{S <: GridFields}
    fields::S
end

track(::CollisionTracker, ::AbstractOutcome, x, v, w) = nothing

function track1(fields, x, val)
    I = cellindext(fields.grid, x)
    checkbounds(Bool, fields.qfixed, I) || return nothing
    fields.qfixed[I] += val / dV(fields.grid, I[1])

    nothing
end

track(tracker::CollisionTracker, ::IonizationOutcome, x, v, w) =
    track1(tracker.fields, x, w)
track(tracker::CollisionTracker, ::AttachmentOutcome, x, v, w) =
    track1(tracker.fields, x, -w)



const poisson_save_n = Ref(0)

"""
    Solve Poisson equation and compute the electrostatic fields.
"""
function poisson!(fields, pind, eb, mg, ws, denoiser, outpath)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! remove outpath when finished debugging
    g = mg.g
    @unpack grid, u, er, ez = fields
    @unpack M, N = grid
    
    fields.qpart .= 0
    
    density!(grid, fields.qpart, pind, Electron(), -1.0)

    # Summing along threads: TODO: pre-allocate
    qfixed = zeros(Float32, M, N)
    qpart = zeros(Float32, M, N)
    
    ci = CartesianIndices(axes(fields.q))
    eps = 1e13
    
    for j in 1:N
        for i in 1:M
            # Adding a fixed density of 1e13 as a hack
            qfixed[i, j] = reduce(+, @view(fields.qfixed[i, j, :])) + eps
            qpart[i, j] = -reduce(+, @view(fields.qpart[i, j, :])) + eps

            # Until we implement using the charges as denoising input we use
            # yet another horrible hack to prevent negative densities in the fixed
            # charges, which are created by attachment.
            if qfixed[i, j] < 0
                qpart[i, j] -= qfixed[i, j] - eps
                qfixed[i, j] = eps
            end
        end
    end

    
    # De-noising
    qfixed1 = denoise(denoiser, qfixed)
    qpart1 = denoise(denoiser, qpart)

    if (poisson_save_n[] % 10) == 0
        ofile = joinpath(outpath, "denoise_" * fmt("04d", poisson_save_n[]) * ".jld")
    
        jldsave(ofile, true; grid, qfixed, qfixed1, qpart, qpart1)
        @info "Saved file" ofile
    end
    
    poisson_save_n[] += 1


    fields.q[1:M, 1:N] .= qfixed1 .- qpart1

    
    Multigrid.solve(mg, parent(fields.u), parent(fields.q), ws)

    Threads.@threads for i in 1:(M + 1)
        for j in 1:(N + 1)
            er[i, j] = (u[i - 1, j] - u[i, j]) / dr(grid)
            ez[i, j] = eb + (u[i, j - 1] - u[i, j]) / dz(grid)
        end
    end

    ez[0,             :] .= ez[1,     :]
    ez[M + 1,         :] .= ez[M,     :]
    er[:,             0] .= er[:,     1]
    er[:,         N + 1] .= er[:,     N]
end


"""
    Update array `arr` with the densities of particles of type 
    `particle` contained in `pind`.
"""
function density!(grid, arr, pind, particle::Particle{sym},
                  val=1.0) where sym
    @unpack pop, colls = pind[sym]

    @inbounds Threads.@threads for i in eachindex(pop)
        pop.active[i] || continue
        I = cellindext(grid, pop.x[i])

        checkbounds(Bool, arr, I) || continue
        
        arr[I] += pop.w[i] * val / dV(grid, I[1])
    end        
end


"""
    Allocate and fill an array with the count of `particle` contained
    in `pind`.
"""
function count(grid, pind, particle::Particle{sym}) where sym
    @unpack pop, colls = pind[sym]
    ctr = calloc_centers(Int, grid)
    
    for i in eachindex(pop)
        pop.active[i] || continue

        I = cellindex(grid, pop.x[i])
        ctr[I] += 1
    end

    ctr
end

"""
    Resample the population of `particle` inside `pind` using 
    Russian roulette and splitting to ensure that in each cell no more than
    nmax super-particles remain.  The weight oof the removed particles are
    distributed among all remaining particles cell-wise.
"""
function resample!(pind, fields, particle::Particle{sym}, nmax) where sym
    @unpack pop, colls = pind[sym]    
    @unpack grid, p, w = fields

    w .= 0.0
    p .= 0
    
    for i in eachindex(pop)
        pop.active[i] || continue

        I = cellindex(grid, pop.x[i])
        checkbounds(Bool, p, I) || continue

        p[I] += 1
        
        if p[I] > nmax
            pop.active[i] = false
            w[I] += pop.w[i]
        end
    end

    for i in eachindex(pop)
        pop.active[i] || continue

        I = cellindex(grid, pop.x[i])
        checkbounds(Bool, p, I) || continue

        if p[I] > nmax
            pop.w[i] += w[I] / nmax
        end
    end
end


""" 
    Average the cylindrical coordinates of two position given by cartesian
    coordinates `x1` and `x2` using normalized weights `w1` and `w2`.
"""
@inline function cylavg(x1, x2, nw1, nw2)
    r1, r2 = rcyl(x1), rcyl(x2)
    θ1, θ2 = atan(x1[2], x1[1]), atan(x2[2], x2[1])

    rm = nw1 * r1 + nw2 * r2
    θm = nw1 * θ1 + nw2 * θ2
    zm = nw1 * x1[3] + nw2 * x2[3]
    sinθm, cosθm = sincos(θm)

    @SVector [rm * cosθm, rm * sinθm, zm]
end

"""
    Randomly choose a particle depending on normalized weights nw1, nw2
    (normalized such that nw1 + nw2 = 1)
"""
@inline function choose(x1, x2, p1, p2, nw1, nw2)
    if rand() < nw1
        return (x1, p1)
    else
        return (x2, p2)
    end
end

""" 
    Shuffle a population using a permutation sampled from a uniform distribution
    of permutations.
"""
function shuffle!(pop::Population)
    n = length(pop)

    for i in n:-1:2
        j = rand(1:i)
        pop.x[i], pop.x[j] = pop.x[j], pop.x[i]
        pop.v[i], pop.v[j] = pop.v[j], pop.v[i]
        pop.w[i], pop.w[j] = pop.w[j], pop.w[i]
    end        
end


function plot(fields)
    plt.matplotlib.pyplot.style.use("granada")
    M = fields.grid.M
    N = fields.grid.N
    # qfixed_t = dropdims(sum(@view(fields.qfixed[1:M, 1:N, :]), dims=3), dims=3)
    # qpart_t = dropdims(sum(@view(fields.qpart[1:M, 1:N, :]), dims=3), dims=3)

    # q = qfixed_t .- qpart_t
    @unpack rf, zf, rc, zc = fields.grid
    
    plt.figure("Charge density")
    plt.pcolormesh(zf, rf, @view(fields.q[1:M, 1:N]))
    cbar = plt.colorbar(label=L"Charge density (C/m$^3$)")
    
    plt.figure("Electric field")

    eabs = @. @views sqrt(0.25 * (fields.er[1:M, 1:N] + fields.er[2:(M + 1), 1:N])^2 +
                          0.25 * (fields.ez[1:M, 1:N] + fields.ez[1:M, 2:(N + 1)])^2)

    plt.pcolormesh(zf, rf, eabs)
    cbar = plt.colorbar(label="Electric field (V/m)")
    
    plt.show()
end

end


if !@isdefined(nprocs) && !isinteractive()
    Streamer.start()
else
    try
        @eval using Revise
        atreplinit() do _
            Revise.track(@__FILE__)
        end
    catch e
        @warn "Failed to track file $(@__FILE__)"
    end
end

