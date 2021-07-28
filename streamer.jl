## streamer.jl : Thu Jul 22 12:29:35 2021
module Streamer
using StaticArrays
using StatsBase: sample, pweights
using Constants: co
using UnPack
using Multigrid
using OffsetArrays
import PyPlot as plt
using LaTeXStrings

using JuMC: Population, CollisionTable, load_lxcat, init!, advance!, collisions!,
    Electron, add_particle!
using JuMC: collrates, repack!, meanenergy, CollisionPopulation,
    IonizationOutcome, AttachmentOutcome, AbstractOutcome, Particle
import JuMC: track, AbstractCollisionTracker
using BenchmarkTools

struct Grid{T}
    R::T
    L::T
    
    M::Int
    N::Int

    rc::StepRangeLen{T, T, T}
    rf::StepRangeLen{T, T, T}
    zc::StepRangeLen{T, T, T}
    zf::StepRangeLen{T, T, T}

    function Grid(L::T, R::T, M, N) where T
        SRL = StepRangeLen{T, T, T}
        rf = range(0, stop=R, length=M + 1)
        zf = range(0, stop=L, length=N + 1)

        rc = 0.5 * (rf[(begin + 1):end] + rf[begin:(end - 1)])
        zc = 0.5 * (zf[(begin + 1):end] + zf[begin:(end - 1)])

        new{T}(R, L, M, N, rc, rf, zc, zf)
    end
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



struct GridFields{T,A1<:AbstractArray{T},A<:AbstractArray{T}}
    grid::Grid{T}
    
    qfixed::A1
    qpart::A1

    q::A
    u::A
    
    er::A
    ez::A

    c::A
    
    function GridFields(grid::Grid{T}) where T
        qfixed = calloc_centers_threads(T, grid)
        qpart = calloc_centers_threads(T, grid)
        q = calloc_centers(T, grid)
        u = calloc_centers(T, grid)
        c = calloc_centers(T, grid)
        er = calloc_faces(T, grid)
        ez = calloc_faces(T, grid)
                
        new{T,typeof(qfixed),typeof(q)}(grid, qfixed, qpart, q, u, er, ez, c)
    end
end


struct FieldInterp{GF <: GridFields}
    fields::GF
end


"""
    Interpolate the electric field at a given position `x`.
    Uses bi-linear interpolation of each of the r/z-components.
"""
function (fieldinterp::FieldInterp)(x)
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
    efield
end


struct CollisionTracker{S <: GridFields}
    fields::S
end

track(::CollisionTracker, ::AbstractOutcome, x, p, w) = nothing

function track1(fields, x, val)
    I = cellindext(fields.grid, x)
    checkbounds(Bool, fields.qfixed, I) || return nothing
    fields.qfixed[I] += val / dV(fields.grid, I[1])

    nothing
end

track(tracker::CollisionTracker, ::IonizationOutcome, x, p, w) =
    track1(tracker.fields, x, w)
track(tracker::CollisionTracker, ::AttachmentOutcome, x, p, w) =
    track1(tracker.fields, x, -w)


function main()
    L = 5e-3
    n = 10000
    
    densities = Dict("N2" => co.nair * 0.79,
                     "O2" => co.nair * 0.21)

    # Create the energy grid in eV; all cross-sections are interpolated into
    # this grid
    energy = (0:0.01:1000) .* co.eV
        
    proc, rate, maxrate = load_lxcat("LXCat-June2013.json", densities, energy)
    ecolls = CollisionTable(proc, energy, rate, maxrate)

    Δt = 10^floor(log10(1 / (2 * ecolls.maxrate)))
    @show Δt
    tmax = 5e-9
    eb = -150 * co.Td * co.nair
    
    T = Float64
    maxp = Int(1e7)
    
    # Initialize the electron population
    epop = Population(Electron(),
                      n,
                      zeros(Bool, maxp),
                      zeros(SVector{3, T}, maxp),
                      zeros(SVector{3, T}, maxp),
                      ones(T, maxp),
                      zeros(Int, maxp))

    epop.x .= Ref(@SVector [0.0, 0.0, L / 4])
    
    pind = (electron = CollisionPopulation(ecolls, epop),)
    init!(pind, Electron(), Δt)

    grid = Grid(L, L, 512, 512)
    fields = GridFields(grid)
    density!(grid, fields.qfixed, pind, Electron(), 1.0)


    # Poisson
    bc = ((LeftBnd(), Dirichlet()),
          (RightBnd(), Dirichlet()),
          (TopBnd(), Neumann()),
          (BottomBnd(), Neumann()))
    
    mg = MGConfig(bc=bc, s=co.elementary_charge * dz(grid)^2 / co.ϵ0,
                  conn=Multigrid.CylindricalConnector{1}(),
                  levels=8,
                  tolerance=1.0e4,
                  smooth1=3,
                  smooth2=3,
                  verbosity=0,
                  g=1)    
    @show mg.s
    
    ws = Multigrid.allocate(mg, parent(fields.q));

    efield = FieldInterp(fields)

    nsteps(pind, Int(fld(tmax, Δt)), efield, eb, Δt, fields, mg, ws)
    
    #k = collrates(pind, Electron())
    density!(grid, fields.qpart, pind, Electron(), -1.0)
    
    (;pind, efield, Δt, fields)
end


function nsteps(pind, n, efield, eb, Δt, fields, mg, ws)
    ecolls = pind.electron.colls
    epop = pind.electron.pop
    tracker = CollisionTracker(fields)
    γ = 0.99
    
    for i in 1:n
        if ((i - 1) % 100 == 0)
            poisson!(fields, pind, eb, mg, ws)
        end
        
        if ((i - 1) % 10000 == 0)
            active_superparticles = count(epop.active)
            physical_particles = sum(epop.w[epop.active])
            @info "$((i - 1) * Δt * 1e9) ns"  active_superparticles physical_particles

        end
        advance!(pind, Electron(), efield, Δt)
        collisions!(pind, Electron(),  Δt, tracker)

        if (i > 0 && (i - 1) % 10000 == 0)
            resample!(pind, fields, Electron(), γ)
            repack!(epop)
        end
        
        for part in keys(pind)            
            @unpack pop = pind[part]
            if (length(pop) / length(pop.active)) > 0.6
                repack!(pop)
            end
        end
    end
end


function poisson!(fields, pind, eb, mg, ws)
    g = mg.g
    @unpack grid, u, er, ez = fields
    @unpack M, N = grid
    
    fields.qpart .= 0
    
    density!(grid, fields.qpart, pind, Electron(), -1.0)

    ci = CartesianIndices(axes(fields.q))
    
    Threads.@threads for I in eachindex(fields.q)
        fields.q[I] = (reduce(+, @view(fields.qfixed[ci[I], :])) +
                       reduce(+, @view(fields.qpart[ci[I], :])))
    end

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


function density!(grid, arr, pind, particle::Particle{sym},
                  val=1.0) where sym
    @unpack pop, colls = pind[sym]

    @inbounds Threads.@threads for i in eachindex(pop)
        pop.active[i] || continue

        I = cellindext(grid, pop.x[i])
        arr[I] += val / dV(grid, I[1])
    end        
end


function resample!(pind, fields, particle::Particle{sym}, γ) where sym
    @unpack pop, colls = pind[sym]    
    @unpack grid, c, qfixed = fields

    c .= 1.0
    wn = ((0.0, 0),
          (0.5, 2),
          (1.0, 1),
          (2.0, 1))

    for i in eachindex(pop)
        pop.active[i] || continue

        I = cellindex(grid, pop.x[i])

        f = (pop.w[i] > 1) ? 2 : 1        
        n0 = 0.5 * (1 + f * c[I])
        c[I] *= γ

        if n0 < 1
            p = @SVector [1 - n0,
                          0.0,
                          2 * (n0 - 0.5),
                          1 - n0]
        else
            p = @SVector [(n0 - 1) / 3,
                          4 * (n0 - 1) / 3,
                          2 * (1.5 - n0),
                          (n0 - 1) / 3]
        end
        
        s = sample(1:4, pweights(p))

        w1, n1 = wn[s]
        
        wnew = pop.w[i] * w1
        
        if s == 1
            pop.active[i] = false

            # Ensure exact charge conservation
            # qfixed[I, 1] -= pop.w[i] / dV(grid, I[1])
        elseif s == 2
            newi = add_particle!(pop, pop.p[i], pop.x[i], wnew)
            pop.s[newi] = pop.s[i]
        elseif s == 4
            # Ensure exact charge conservation
            # qfixed[I, 1] += pop.w[i] / dV(grid, I[1])            
        end

        pop.w[i] = wnew

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


if !isinteractive()
    Streamer.main()
else
    try
        using Revise
        Revise.track(@__FILE__)
    catch e
        @warn "Failed to track file $(@__FILE__)"
    end
end

