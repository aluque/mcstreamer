#
# Non-relativistic dynamics
#
mass(k::Electron) = co.electron_mass
energy(k::Electron, v2::Real) = 0.5 * mass(k) * v2
energy(k::Electron, v::AbstractVector) = energy(k, sum(v.^2))

function advance_free(k::Electron, x, v, efield, Δt)
    # Leapfrog integration. Note that x and v are not synchronous.
    Δv = -(Δt * co.elementary_charge / mass(k)) .* efield(x) 
    v1 = v + Δv

    x .+ Δt * v1, v1
end


#
# Collisional processes and collision outcomes
#
struct Excitation{T} <: CollisionProcess
    threshold::T
end

function collide(c::Excitation, k::Electron, v, energy)
    E1 = max(0.0, energy - c.threshold)
    vabs = sqrt(2 * E1 / mass(k))
    v1 = randsphere() .* vabs

    VelocityChangeOutcome{Electron, eltype(v1)}(v1)
end


struct Ionization{T} <: CollisionProcess
    threshold::T
end

function collide(c::Ionization, k::Electron, v, energy)
    # Energy equipartitiom
    E1 = 0.5 * max(0.0, energy - c.threshold)
    vabs = sqrt(2 * E1 / mass(k))

    v = randsphere() .* vabs
    v1 = randsphere() .* vabs

    IonizationOutcome{Electron, eltype(v)}(v, v1)
end


struct Attachment{T} <: CollisionProcess
    threshold::T
end


function collide(c::Attachment, k::Electron, p, energy)
    AttachmentOutcome{Electron}()
end


struct Elastic{T} <: CollisionProcess
    mass_ratio::T
end

function collide(c::Elastic, k::Electron, v, E)
    # Velocity in the center-of-mass frame
    v_cm = (c.mass_ratio / (1 + c.mass_ratio)) .* v
    v_final = randsphere() .* norm(v - v_cm) .+ v_cm

    VelocityChangeOutcome{Electron, eltype(v)}(v_final)
end
