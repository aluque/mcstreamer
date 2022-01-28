abstract type CollisionProcess end

"""
    Struct with a set of collisions evaluated at the same energy grid.
"""
@with_kw struct CollisionTable{T, C <: Tuple, V <: AbstractRange{T},
                               A <: AbstractMatrix{T}}    
    "Each type of collision"
    proc::C

    "Energy grid"
    energy::V

    "Array with collision rates for each process."
    rate::A

    "Max. collision rate"
    maxrate::T
end

Base.length(c::CollisionTable) = length(c.proc)


"""
Return the index and weight of the item before x in a range.
"""
function indweight(r::AbstractRange, x)
    #i = searchsortedlast(r, x)
    i = Int(fld(x - first(r), step(r))) + 1
    #@assert ip == i "$ip != $i"
    w = (r[i + 1] - x) / step(r)
    @assert 0 <= w <= 1
    return (i, w)
end

indweight(colls::CollisionTable, E) = indweight(colls.energy, E)


#
# Collision outcomes
#
# Each collision type must return always the same type of collision to ensure
# type stability.

abstract type AbstractOutcome end

struct AttachmentOutcome{P <: Particle} <: AbstractOutcome
end

struct IonizationOutcome{P <: Particle, T} <: AbstractOutcome
    v1::SVector{3, T}
    v2::SVector{3, T}
end

struct VelocityChangeOutcome{P <: Particle, T} <: AbstractOutcome
    v::SVector{3, T}
end

"""
   appply!(pind, outcome, i, maxrate, Δt)

Apply a `CollisionOutcome` to a population index `pind`.  `i` is the index
of the colliding particle, which is needed if it experiences a change.
We also need to pass `maxrate` and `Δt` because for ionization collisions,
they are needed to initialize the time-to-next collisions of the new
particle.
"""
function apply!(pind, outcome::VelocityChangeOutcome{Particle{sym}}, i,
                maxrate, Δt) where sym
    @unpack pop = pind[sym]
    @inbounds pop.v[i] = outcome.v
end

function apply!(pind, outcome::IonizationOutcome{Particle{sym}}, i,
                maxrate, Δt) where sym
    @unpack pop = pind[sym]
    @inbounds pop.v[i] = outcome.v1
    @inbounds newi = add_particle!(pop, outcome.v2, pop.x[i], pop.w[i])

    pop.s[newi] = Int(fld(nextcol(maxrate), Δt))
end

function apply!(pind, outcome::AttachmentOutcome{Particle{sym}}, i,
                maxrate, Δt) where sym
    @unpack pop = pind[sym]
    @inbounds pop.active[i] = false
end

#
# With collision tracker we can define a callback executed whenever a collision
# happens.
#
abstract type AbstractCollisionTracker end
struct VoidCollisionTracker end

track(::AbstractCollisionTracker, outcome::AbstractOutcome, x, v) = nothing



"""
Check for the particles that are set to collide and then perform a
(possibly null) collision
"""
function collisions!(pind, particle::Particle{sym}, Δt,
                     tracker=VoidCollisionTracker()) where sym
    @unpack pop, colls = pind[sym]

    #ncolls = zeros(Int, Threads.nthreads())
    
    @threads for i in eachindex(pop)
        @inbounds pop.active[i] || continue

        pop.s[i] -= 1

        if pop.s[i] < 0
            #ncolls[Threads.threadid()] += 1
            #@info "Particle $i collision"
            E = energy(pop.particle, pop.v[i])
            ξ = rand(typeof(E)) * colls.maxrate
            
            do_one_collision!(pind, particle, pop, colls, tracker, i, ξ, E, Δt)
            pop.s[i] = Int(fld(nextcol(colls.maxrate), Δt))
        end
    end
    
    #println("$ncolls of $(pop.n) particles (fraction $(ncolls / (pop.n)))")
    
end


@generated function do_one_collision!(pind, particle, pop, colls::CollisionTable{T, C},
                                     tracker, i, ξ, E, Δt) where {T, C}
    L = fieldcount(C)

    out = quote
        k, w = indweight(colls, E)
    end
    
    for j in 1:L
        push!(out.args,
              quote
              @inbounds ν = w * colls.rate[$j, k] + (1 - w) * colls.rate[$j, k + 1]
              if ν > ξ
                outcome = collide(colls.proc[$j], particle, pop.v[i], E)
                track(tracker, outcome, pop.x[i], pop.v[i], pop.w[i])
                apply!(pind, outcome, i, colls.maxrate, Δt)
                return
              else
                ξ -= ν
              end
              end)
    end
    
    push!(out.args,
          quote
          return nothing
          end)
          
    return out
end


"""
    collrates(pind, particle)

Return the collision rates for all processes for a given particle.
""" 
function collrates(pind, particle::Particle{sym}) where sym
    @unpack pop, colls = pind[sym]

    k = zeros(length(colls))

    for ip in eachindex(pop)
        E = energy(particle, pop.v[ip])
        i, w = indweight(colls, E)
        k += @. (w * colls.rate[:, i] + (1 - w) * colls.rate[:, i + 1])
    end

    k ./= pop.n[]
    return k
end
