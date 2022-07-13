module MCStreamer

using StaticArrays 
using StatsBase: sample, pweights
using Constants: co
using UnPack
using Accessors
using StructArrays
using Multigrid
using OffsetArrays
import PyPlot as plt
using LaTeXStrings
using JLD2
using CodecZlib
using Formatting
using Logging
using Dates
using LinearAlgebra
using TOML

using JuMC: NewParticleOutcome, RemoveParticleOutcome, StateChangeOutcome,
    NullOutcome, ParticleType, shuffle!, repack!, load_lxcat, CollisionTable, Electron,
    Population, nextcoll, ElectronState, MultiPopulation,
    actives, weight, advance!, collisions!, AbstractCollisionTracker,
    eachparticle, remove_particle!, add_particle!, ZhelezniakCollisions, PhotonState
import JuMC

include("autoencoder.jl")
using .Autoencoder: Denoiser, denoise, NullDenoiser

include("timesteps.jl")

include("main.jl")
include("resample.jl")
include("grid.jl")
include("gridfields.jl")
include("poisson.jl")
include("plot.jl")


end # module
