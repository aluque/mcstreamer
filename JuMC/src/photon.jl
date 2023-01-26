#=
  This is the implementation of a "Zhelezniak" photon: a photon with a random
  absorption rate between two maxima and a uniform probability in log space.
=#

const Photon = ParticleType{:photon}

" Charge in units of the elementary charge. "
charge(::Type{Photon}) = 0

struct PhotonState{T} <: ParticleState{T}
    x::SVector{3, T}
    v::SVector{3, T}
    ν::T
    w::T
    s::T    
    active::Bool
end

particle_type(::Type{PhotonState{T}}) where T = Photon

# Currently we do not use the energy anywhere
energy(::PhotonState) = nothing

struct PhotoIonization; end

struct ZhelezniakCollisions{T, C} <: AbstractCollisionTable{T, C}
    proc::C

    νmax::T

    function ZhelezniakCollisions(νmax::T) where T
        proc = (PhotoIonization(),)

        new{T, typeof(proc)}(proc, νmax)
    end
end

presample(c::ZhelezniakCollisions, state, energy) = state.ν
rate(c::ZhelezniakCollisions, j, ν) = ν

maxrate(c::ZhelezniakCollisions) = c.νmax


function collide(c::PhotoIonization, p::PhotonState{T}, energy) where T
    # The electron starts with 0 velocity
    v  = zero(SVector{3, Float64})

    pe = ElectronState{T}(p.x,  v, p.w, nextcoll(), p.active)
    ReplaceParticleOutcome(p, pe)
end


@inline function advance_free(p::PhotonState, efield, Δt)
    PhotonState(p.x .+ Δt .* p.v, p.v, p.ν, p.w, p.s, p.active)
end
