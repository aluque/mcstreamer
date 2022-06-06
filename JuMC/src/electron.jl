#
# Non-relativistic dynamics
#
const Electron = ParticleType{:electron}

" Charge in units of the elementary charge. "
charge(Type{Electron}) = -1

struct ElectronState{T} <: ParticleState
    v::SVector{3, T}
    x::SVector{3, T}
end

particle_type(Type{ElectronState{T}}) where T = Electron
new_particle(Type{Electron}, x, v) = ElectronState(x, v)

mass(p::ElectronState) = co.electron_mass
energy(p::ElectronState) = 0.5 * mass(p) * p.v^2

@inline function advance_free(p::ElectronState, efield, Δt)
    # Leapfrog integration. Note that x and v are not synchronous.
    Δv = -(Δt * co.elementary_charge / mass(p)) .* efield(p.x)
    v1 = p.v + Δv

    ElectronState(p.x .+ Δt * v1, v1)
end


#
# Collisional processes and collision outcomes
#
struct Excitation{T} <: CollisionProcess
    threshold::T
end

struct Ionization{T} <: CollisionProcess
    threshold::T
end

struct Attachment{T} <: CollisionProcess
    threshold::T
end

struct Elastic{T} <: CollisionProcess
    mass_ratio::T
end

# Note: Here I am making copies of the particle's position at p.x. I hope
# that the compiler will be able to see them as no-ops.
function collide(c::Excitation, p::ElectronState{T}, energy) where T
    E1 = max(0, energy - c.threshold)
    vabs = sqrt(2 * E1 / mass(p))
    v1 = randsphere() .* vabs

    StateChangeOutcome(ElectronState{T}(v1, p.x))
end

function collide(c::Ionization, p::ElectronState{T}, energy) where T
    # Energy equipartitiom
    E1 = max(0, energy - c.threshold) / 2
    vabs = sqrt(2 * E1 / mass(p))

    v = randsphere() .* vabs
    v1 = randsphere() .* vabs

    p1 = ElectronState{T}(v, p.x)
    p2 = ElectronState{T}(v1, p.x)
    NewParticleOutcome(p1, p2)
end

function collide(c::Attachment, p::ElectronState, energy)
    RemoveParticleOutcome{Electron}()
end

function collide(c::Elastic, p::ElectronState{T}, energy) where T
    # Velocity in the center-of-mass frame
    v_cm = (c.mass_ratio / (1 + c.mass_ratio)) .* v
    v_final = randsphere() .* norm(v - v_cm) .+ v_cm

    StateChangeOutcome(ElectronState{T}(v_final, p.x))
end
