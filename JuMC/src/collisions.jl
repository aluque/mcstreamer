abstract type CollisionProcess end

struct NullCollision <: CollisionProcess end

abstract type AbstractCollisionTable{T, C <: Tuple}; end
"""
    Struct with a set of collisions evaluated at the same energy grid.
"""
@with_kw struct CollisionTable{T, C <: Tuple, V <: AbstractRange{T},
                               A <: AbstractMatrix{T}} <: AbstractCollisionTable{T, C}
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
maxenergy(c::CollisionTable) = c.energy[end]

# Checking for collisions involves many tests but some computations are common
# common to all of them. Here we store them in a generic way. energy is passed
# as an optimization.
presample(c::CollisionTable, state, energy) = indweight(c.energy, energy)

# Obtain probability rate of process j
function rate(c::CollisionTable, j, pre)
    k, w = pre

    return w * c.rate[j, k] + (1 - w) * c.rate[j, k + 1]
end


#
# Collision outcomes
#
# Each collision type must return always the same type of collision to ensure
# type stability.

abstract type AbstractOutcome end

# Nothing happens
struct NullOutcome <: AbstractOutcome end
    
# The colliding particle dissapears
struct RemoveParticleOutcome{PS} <: AbstractOutcome
    state::PS
end

# The state of the colliding particle changes to p
struct StateChangeOutcome{PS} <: AbstractOutcome
    state::PS
end

# A new particle is created with state state2. The state of
# the colliding particle changes to state1
struct NewParticleOutcome{PS1, PS2} <: AbstractOutcome
    state1::PS1
    state2::PS2
end

# A new particle with state p2 replaces an existing particle with state state1.
# This is e.g. when a photon is absorbed and liberates an electron.
struct ReplaceParticleOutcome{PS1, PS2} <: AbstractOutcome
    state1::PS1
    state2::PS2
end

# For Poisson photon generations we may want to generate many particles in a single
# event. To sample them we create instances of this.
struct MultiplePhotonOutcome{PS1, C} <: AbstractOutcome
    # The new state of the electron producing the emission
    nphot::Int
    state1::PS1
    photoemit::C
end


@inline collide(c::NullCollision, k, energy) = NullOutcome()

"""
   appply!(population, outcome, i)

Apply a `CollisionOutcome` to a population `population`.  `i` is the index
of the colliding particle, which is needed if it experiences a change.
We delegate to `population` handling the creation of a new particle.
"""
@inline function apply!(mpopl, outcome::NullOutcome, i)
end

@inline function apply!(mpopl, outcome::StateChangeOutcome{PS}, i) where PS
    popl = get(mpopl, particle_type(PS))
    popl.particles[i] = outcome.state
end

@inline function apply!(mpopl, outcome::NewParticleOutcome{PS1, PS2}, i) where {PS1, PS2}
    popl1 = get(mpopl, particle_type(PS1))
    popl1.particles[i] = outcome.state1

    popl2 = get(mpopl, particle_type(PS2))
    add_particle!(popl2, outcome.state2)
end

@inline function apply!(mpopl, outcome::RemoveParticleOutcome{PS}, i) where PS
    popl = get(mpopl, particle_type(PS))
    remove_particle!(popl, i)
end

@inline function apply!(mpopl, outcome::ReplaceParticleOutcome{PS1, PS2}, i) where {PS1, PS2}
    popl1 = get(mpopl, particle_type(PS1))
    remove_particle!(popl1, i)

    popl2 = get(mpopl, particle_type(PS2))
    add_particle!(popl2, outcome.state2)    
end

@inline function apply!(mpopl, outcome::MultiplePhotonOutcome{PS1, C}, i) where {PS1, C}
    popl1 = get(mpopl, particle_type(PS1))
    popl1.particles[i] = outcome.state1
    T = eltype(outcome.state1)

    p = outcome.state1
    
    (;log_νmin, log_νmax, weight_scale) = outcome.photoemit
    photons = get(mpopl, Photon)

    for i in 1:outcome.nphot
        v = randsphere() .* co.c
        ν = exp(log_νmin + (log_νmax - log_νmin) * rand())
        p2 = PhotonState{T}(p.x, v, ν, 1.0, nextcoll(), p.active)
        add_particle!(photons, p2)    
    end
end


#
# With collision tracker we can define a callback executed whenever a collision
# happens.
#
abstract type AbstractCollisionTracker end
struct VoidCollisionTracker <: AbstractCollisionTracker end

track(::AbstractCollisionTracker, outcome::AbstractOutcome) = nothing


indweight(colls::CollisionTable, E) = indweight(colls.energy, E)

"""
    Sample one (possibly null) collision.
"""    
@generated function do_one_collision!(mpopl, colls::AbstractCollisionTable{T, C}, state, i, tracker) where {T, C}
    L = fieldcount(C)

    out = quote
        $(Expr(:meta, :inline))
        eng = energy(state)
        pre = presample(colls, state, eng)
        ξ = rand(T) * maxrate(colls)
        
        #k, w = indweight(colls, energy)
    end
    
    for j in 1:L
        push!(out.args,
              quote
              ν = rate(colls, $j, pre)
              if ν > ξ                  
                  outcome = collide(colls.proc[$j], state, eng)
                  track(tracker, outcome)
                  apply!(mpopl, outcome, i)
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
