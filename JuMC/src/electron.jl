#
# Relativistic dynamics
#
const m2c2 = co.electron_mass^2 * co.c^2
const mc2 = co.electron_mass * co.c^2

gamma(k::Electron, p2::Real) = sqrt(1 + p2 / m2c2)
gamma(k::Electron, p::AbstractVector) = gamma(k, sum(p.^2))


function gamma1(k::Electron, p2::Real)
    ϵ = 1e-4

    (p2 < 8ϵ * m2c2) && return 0.5 * p2 / m2c2
    return gamma(k, p2) - 1
end

gamma1(k::Electron, p::AbstractVector) = gamma1(k, sum(p.^2))

velocity(k::Electron, p) = p ./ (gamma(k, p) * co.electron_mass)
energy(k::Electron, p2::Real) = gamma1(k, p2) * mc2
energy(k::Electron, p::AbstractVector) = energy(k, sum(p.^2))

function advance_free(k::Electron, x, p, efield, Δt)
    Δp = -(Δt * co.elementary_charge) .* efield(x) 
    v1 = velocity(k, p .+ Δp / 2)

    x .+ Δt .* v1, p .+ Δp
end


#
# Collisional processes and collision outcomes
#
struct Excitation{T} <: CollisionProcess
    threshold::T
end

function collide(c::Excitation, k::Electron, p, energy)
    E1 = max(0.0, energy - c.threshold)
    pabs = sqrt(E1 * (E1 + 2 * mc2)) / co.c
    p = randsphere() .* pabs

    MomentumChangeOutcome{Electron, eltype(p)}(p)
end


struct Ionization{T} <: CollisionProcess
    threshold::T
end

function collide(c::Ionization, k::Electron, p, energy)
    # Energy equipartitiom
    E1 = 0.5 * max(0.0, energy - c.threshold)
    pabs = sqrt(E1 * (E1 + 2 * mc2)) / co.c

    p = randsphere() .* pabs
    p1 = randsphere() .* pabs

    IonizationOutcome{Electron, eltype(p)}(p, p1)
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

function collide(c::Elastic, k::Electron, p, E)
    v = velocity(k, p)

    # Velocity in the center-of-mass frame
    v_cm = (c.mass_ratio / (1 + c.mass_ratio)) .* v
    v_final = randsphere() .* norm(v - v_cm) .+ v_cm

    v2_final = sum(v.^2)
    γ = 1 / sqrt(1 -  v2_final/ co.c^2)
    pf = (co.electron_mass * γ) .* v_final

    MomentumChangeOutcome{Electron, eltype(p)}(pf)
end
