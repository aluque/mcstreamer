"""
    A container for all fields located on a grid.

    Some fields (of type `A1`) contain an extra dimension for the thread id, 
    which allows to update in parallel.
"""
struct GridFields{T,A1<:AbstractArray{T},A<:AbstractArray{T},AI<:AbstractArray{Int},
                  AL}
    grid::Grid{T}
    
    # Fixed charges
    qfixed::A1

    # Charges associated with mobile particles.
    qpart::A1

    # Pre-computed charge density.  Differs from q only without denoising
    q0::A
    
    # Charge density
    q::A

    # Fluid electron density
    ne::A

    # derivative of fluid electron density
    dne::A
    
    # Electrostatic potential
    u::A
    
    # r-component of the electric field
    er::A

    # z-component of the electric field
    ez::A

    # For Russian roulette; counter of particles inside each cell.
    p::AI

    # For Russian roulette: Number of particles selected so far
    pk::AI

    # For Russian roulette: total weight inside a cell
    wtotal::A
    
    # For Russian roulette: cumulative weight visited so far
    wcum::A
    
    # For Russian roulette: next stop
    x::A

    # Array of locks for concurrence
    locks::AL
    
    # A lock for each thread. This is necesssary because we do not have guarantee that two tasks
    # won't share the same threadid so in updating operations it is important to ensure that
    # not two tasks are accessing the same cell.
    thread_locks::Vector{Threads.SpinLock}
    
    """ Allocate fields for a grid `grid`. """
    function GridFields(grid::Grid{T}) where T
        qfixed = calloc_centers_threads(T, grid)
        qpart = calloc_centers_threads(T, grid)
        q0 = calloc_centers(T, grid)
        q = calloc_centers(T, grid)
        ne = calloc_centers(T, grid, 2)
        dne = calloc_centers(T, grid, 2)
        u = calloc_centers(T, grid)
        er = calloc_faces(T, grid)
        ez = calloc_faces(T, grid)
        p = calloc_centers(Int, grid)
        pk = calloc_centers(Int, grid)
        wtotal = calloc_centers(T, grid)
        wcum = calloc_centers(T, grid)
        x = calloc_centers(T, grid)
        locks = map(_ -> Threads.SpinLock(), p)
        thread_locks = map(_ -> Threads.SpinLock(), 1:Threads.nthreads())
        
        new{T,typeof(qfixed),typeof(q),typeof(p),typeof(locks)}(grid, qfixed, qpart, q0,
                                                  q, ne, dne, u, er, ez, p, pk, wtotal, wcum, x,
                                                  locks, thread_locks)
    end
end

"""
    A callable to use for field interpolations.
"""
struct FieldInterp{GF <: GridFields}
    fields::GF
end


@inline @fastmath function normdivrem(x, y)
    r = x / y
    m = trunc(Int, r)
    return (m, r - m)
end

@inline @fastmath function staggered(i, δ)
    if δ < 1/2
        i1 = i
        δ1 = δ + 1/2
    else
        i1 = i + 1
        δ1 = δ - 1/2
    end
    return (i1, δ1)
end

"""
    Interpolate the electric field at a given position `x`.
    Uses bi-linear interpolation of each of the r/z-components.
"""
function (fieldinterp::FieldInterp)(x)
    inside(fieldinterp.fields.grid, x) || return zero(SVector{3, eltype(x)})
    r = sqrt(x[1]^2 + x[2]^2)

    # avoid 0/0
    r == 0 && (r = dr(grid) / 10)
        
    grid = fieldinterp.fields.grid
    fields = fieldinterp.fields
    (;er, ez) = fields
    
    # Turns out, divrem is consuming most of the time in this function, which
    # is performance-critical.  So here is an implementation using fewer
    # divrems
    
    i, δr = normdivrem(r - first(grid.rf), dr(grid))
    i1, δr1 = staggered(i, δr)

    j, δz = normdivrem(x[3] - first(grid.zf), dz(grid))
    j1, δz1 = staggered(j, δz)

    # For 1-based indexing; not in i1, which starts at 0
    i += 1
    j += 1

    # let _i = i, _i1 = i1, _j = j, _j1 = j1, _δr = δr, _δz = δz, _δr1 = δr1, _δz1 = δz1
    #     # Not staggered
    #     ifl, δr = divrem(r - first(grid.rf), dr(grid))
    #     i = Int(ifl) + 1
    #     jfl, δz = divrem(x[3] - first(grid.zf), dz(grid))
    #     j = Int(jfl) + 1
        
    #     # Staggered; Note that here the lowest index is 0 so we don't add 1
    #     i1fl, δr1 = divrem(r - first(grid.rf) + 0.5 * dr(grid), dr(grid))    
    #     i1 = Int(i1fl)
        
    #     j1fl, δz1 = divrem(x[3] - first(grid.zf) + 0.5 * dz(grid), dz(grid))
    #     j1 = Int(j1fl)
        
    #     # Normalize to 0..1
    #     δr  /= dr(grid)
    #     δz  /= dz(grid)
    #     δr1 /= dr(grid)
    #     δz1 /= dz(grid)

    #     @assert _i == i
    #     @assert _j == j
    #     @assert _i1 == i1
    #     @assert _j1 == j1
    #     @assert isapprox(_δr, δr)
    #     @assert isapprox(_δz, δz)
    #     @assert isapprox(_δr1, δr1)
    #     @assert isapprox(_δz1, δz1)
    # end
    
    er1 = (er[i, j1]     * (1 - δr) * (1 - δz1) + er[i + 1, j1]     * δr * (1 - δz1) +
           er[i, j1 + 1] * (1 - δr) * δz1       + er[i + 1, j1 + 1] * δr * δz1)
    
    ez1 = (ez[i1, j]     * (1 - δr1) * (1 - δz) + ez[i1 + 1, j]     * δr1 * (1 - δz) +
           ez[i1, j + 1] * (1 - δr1) * δz       + ez[i1 + 1, j + 1] * δr1 * δz)
        
    efield = @SVector [er1 * x[1] / r, er1 * x[2] / r, ez1]

    # Large electric fields are surely a bug
    if norm(efield) > 1e8
        lock(fields.thread_locks[begin]) do 
        
            jldsave("MCStreamer_error_fields.jld", false; fields);
            @info "Too high electric field" x efield er1 ez1
        
            error("Too high electric field")
        end
    end

    return efield
end


"""
    A tracker is called whenerever a particle experiences a collision.
    This is useful for example to keep track of fixed charges left at the
    collision location.
"""
struct CollisionTracker{S <: GridFields} <: AbstractCollisionTracker
    fields::S
end


function track1(fields, x, val)
    I = cellindext(fields.grid, x)
    checkbounds(Bool, fields.qfixed, I) || return nothing
    checkbounds(Bool, fields.grid.rc, I[1]) || return nothing

    lock(fields.thread_locks[I[length(I)]])
    fields.qfixed[I] += val / dV(fields.grid, I[1])
    unlock(fields.thread_locks[I[length(I)]])

    nothing
end

JuMC.track(tracker::CollisionTracker, out::NewParticleOutcome) =
    track1(tracker.fields, out.state1.x, out.state1.w)
JuMC.track(tracker::CollisionTracker, out::RemoveParticleOutcome) =
    track1(tracker.fields, out.state.x, -out.state.w)
function JuMC.track(tracker::CollisionTracker,
                    out::ReplaceParticleOutcome{PhotonState{T}, ElectronState{T}}) where {T}
    track1(tracker.fields, out.state1.x, out.state1.w)
end
JuMC.track(::CollisionTracker, ::StateChangeOutcome) = nothing
JuMC.track(::CollisionTracker, ::NullOutcome) = nothing
