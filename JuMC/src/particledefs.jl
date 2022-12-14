#=
NOTES ON TYPES:

With each particle kind (electrons, photons...) we associate two julia types.
  1. A ParticleType (e.g. ParticleType{:electron}) does not contain any
     fields and is used only for dispatch.  For example when we want properties
     of a particle that are independent of its physical state (mass, charge)
     or when we want to obtain the population of a given particle out of a
     set of several particle populations. 
     We dispatch on the singleton type e.g. Type{Electron} where
     Electron = ParticleType{:electron} so the functions are called as
     f(Electron, ....).

  2. A ParticleState that contains the physical state of the particle
     (velocity, position or any other variable that we want to follow).

Finally we encapsulate ParticleStates into SuperParticleStates, wich also
contain simulation properties of the particle such as weight, whether it is
active etc.  The reason is that we can write functions that are independent of
the type of particle that is being handled.
=#
struct ParticleType{S}; end

const Positron = ParticleType{:positron}
const Photon = ParticleType{:photon}
const Electron = ParticleType{:electron}

@inline id(p::ParticleType{S}) where S = S
@inline id(sym::Symbol) = sym

name(p::ParticleType) = String(id(p))

"""
    This is the abstract type that is sub-classed by structs that store the
    state of a given particle.  For example for electrons we may use
    ElectronState <: ParticleState that contains position, velocity etc.
    All ParticleStates must contain at least these fields besides the particle
    variables: `w`: particle weight, `s` normalized time to next collision, 
    `active` whether the particle is active.
"""
abstract type ParticleState{T}; end

Base.eltype(p::ParticleState{T}) where {T} = T
