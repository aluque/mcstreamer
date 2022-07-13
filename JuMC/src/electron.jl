#
# Non-relativistic dynamics
#
const Electron = ParticleType{:electron}

" Charge in units of the elementary charge. "
charge(::Type{Electron}) = -1

struct ElectronState{T} <: ParticleState{T}
    x::SVector{3, T}
    v::SVector{3, T}
    w::T
    s::T    
    active::Bool
end

ElectronState(x, v, w=1.0, s=nextcoll(), active=true) = ElectronState(x, v, w, s, active)

particle_type(::Type{ElectronState{T}}) where T = Electron
new_particle(::Type{Electron}, x, v) = ElectronState(x, v, 1.0, nextcoll(), true)

mass(p::ElectronState) = co.electron_mass
mass(::Type{Electron}) = co.electron_mass
mass(::Electron) = co.electron_mass

energy(p::ElectronState) = 0.5 * mass(p) * (p.v[1]^2 + p.v[2]^2 + p.v[3]^2)

@inline function advance_free(p::ElectronState, efield, Δt)
    # Leapfrog integration. Note that x and v are not synchronous.
    Δv = -(Δt * co.elementary_charge / mass(p)) .* efield(p.x)
    v1 = p.v .+ Δv

    ElectronState(p.x .+ Δt * v1, v1, p.w, p.s, p.active)
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

# This photo-emission. The energy loss and ionization are already included in
# other processes, so this only creates a new photon.
struct PhotoEmission{T} <: CollisionProcess;
    log_νmin::T
    log_νmax::T

    function PhotoEmission(νmin, νmax)
        lmin, lmax = promote(log(νmin), log(νmax))
        new{typeof(lmin)}(lmin, lmax)
    end
end


function collide(c::Excitation, p::ElectronState{T}, energy) where T
    E1 = max(0, energy - c.threshold)
    vabs = sqrt(2 * E1 / mass(p))
    v1 = randsphere() .* vabs

    StateChangeOutcome(ElectronState{T}(p.x, v1, p.w, p.s, p.active))
end

function collide(c::Ionization, p::ElectronState{T}, energy) where T
    # Energy equipartitiom
    E1 = max(0, energy - c.threshold) / 2
    vabs = sqrt(2 * E1 / mass(p))

    v  = randsphere() .* vabs
    v1 = randsphere() .* vabs

    p1 = ElectronState{T}(p.x,  v, p.w, p.s, p.active)
    p2 = ElectronState{T}(p.x, v1, p.w, nextcoll(), p.active)
    NewParticleOutcome(p1, p2)
end

function collide(c::Attachment, p::ElectronState, energy)
    RemoveParticleOutcome(p)
end

function collide(c::Elastic, p::ElectronState{T}, energy) where T
    # Velocity in the center-of-mass frame
    v_cm = (c.mass_ratio / (1 + c.mass_ratio)) .* p.v
    v_final = randsphere() .* norm(p.v - v_cm) .+ v_cm

    StateChangeOutcome(ElectronState{T}(p.x, v_final, p.w, p.s, p.active))
end

function collide(c::PhotoEmission, p::ElectronState{T}, energy) where T
    v = randsphere() .* co.c
    (;log_νmin, log_νmax) = c

    ν = exp(log_νmin + (log_νmax - log_νmin) * rand())
    p2 = PhotonState{T}(p.x, v, ν, p.w, nextcoll(), p.active)

    NewParticleOutcome(p, p2)
end
