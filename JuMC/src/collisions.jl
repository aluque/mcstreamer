abstract type CollisionProcess end

struct NullCollision <: CollisionProcess end

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

maxrate(c::CollisionTable) = c.maxrate

#
# Collision outcomes
#
# Each collision type must return always the same type of collision to ensure
# type stability.

abstract type AbstractOutcome end

# Nothing happens
struct NullOutcome <: AbstractOutcome end
    
# The colliding particle dissapears
struct RemoveParticleOutcome{PT} <: AbstractOutcome end

# The state of the colliding particle changes to p
struct StateChangeOutcome{PS} <: AbstractOutcome
    p::PS
end

# A new particle is created with state p2. The state of
# the colliding particle changes to p1
struct NewParticleOutcome{PS1, PS2} <: AbstractOutcome
    p1::PS1
    p2::PS2
end

@inline collide(c::NullCollision, k, energy) = NullOutcome()

"""
   appply!(population, outcome, i)

Apply a `CollisionOutcome` to a population `population`.  `i` is the index
of the colliding particle, which is needed if it experiences a change.
We delegate to `population` handling the creation of a new particle.
"""
@inline function apply!(population, outcome::NullOutcome, i)
    # DO-nothing collision
end

@inline function apply!(population, outcome::StateChangeOutcome{PS}, i) where PS
    set_particle_state!(population, particle_type(PS), i, outcome.p)
end

@inline function apply!(population, outcome::NewParticleOutcome{PS1, PS2}, i) where {PS1, PS2}
    w = get_super_particle_state(population, particle_type(PS1), i).w
    
    set_particle_state!(population, particle_type(PS1), i, outcome.p1)
    add_particle!(population, particle_type(PS2), outcome.p2, w)
end

@inline function apply!(population, outcome::RemoveParticleOutcome{PT}, i) where PT
    remove_particle!(population, PT, i)
end

#
# With collision tracker we can define a callback executed whenever a collision
# happens.
#
abstract type AbstractCollisionTracker end
struct VoidCollisionTracker end

track(::AbstractCollisionTracker, outcome::AbstractOutcome, p::SuperParticleState) = nothing


"""
Check for the particles that are set to collide and then perform a
(possibly null) collision
"""
function collisions!(population, particle_type, Δt, tracker=VoidCollisionTracker())
    n = get_particle_count(population, particle_type)
    
    @batch for i in 1:n        
        super_state = get_super_particle_state(population, particle_type, i)
        s = super_state.s
        s -= Δt
        if s > 0
            set_particle_super_state(population, particle_type, i,
                                     @set super_state.s = s)            
        else
            state = super_state.state
            
            νmax = maxrate(population, particle_type)
            
            E = energy(state)
            ξ = rand(typeof(E)) * νmax
            
            do_one_collision!(population, particle_type, super_state, ξ, i,
                              E, tracker)

            s = nextcol(νmax)
            set_super_particle_state(population, particle_type, i,
                                     @set super_state.s = s)            
        end
    end
    
    #println("$ncolls of $(pop.n) particles (fraction $(ncolls / (pop.n)))")
    
end

indweight(colls::CollisionTable, E) = indweight(colls.energy, E)

"""
    Sample one (possibly null) collision.
"""    
@generated function do_one_collision!(population, particle_type, super_state, ξ, i, energy,
                                      tracker)
    colls = get_collisions(population, particle_type)

    L = fieldcount(C)

    out = quote
        $(Expr(:meta, :inline))
        k, w = indweight(colls, energy)
    end
    
    for j in 1:L
        push!(out.args,
              quote
              @inbounds ν = w * colls.rate[$j, k] + (1 - w) * colls.rate[$j, k + 1]
              if ν > ξ                  
                  outcome = collide(colls.proc[$j], state, energy)
                  track(tracker, outcome, super_state)
                  apply!(population, outcome, i)
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
