module MCStreamer

using Random
using HDF5
using StaticArrays 
using Distributions: Poisson
using StatsBase: sample, pweights
using Constants: co
using UnPack
using Accessors
using StructArrays
using Multigrid
using FFTW
using SpecialFunctions
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
using Polyester

using JuMC: NewParticleOutcome, RemoveParticleOutcome, StateChangeOutcome, ReplaceParticleOutcome,
    NullOutcome, ParticleType, shuffle!, repack!, load_lxcat, CollisionTable, Electron, Photon,
    Population, nextcoll, ElectronState, MultiPopulation, nparticles,
    actives, weight, advance!, collisions!, AbstractCollisionTracker,
    eachparticle, remove_particle!, add_particle!, ZhelezniakCollisions, PhotonState
import JuMC

include("autoencoder.jl")
using .Autoencoder: Denoiser, denoise, CutoutDenoiser, NullDenoiser

include("timesteps.jl")

include("main.jl")
include("resample.jl")
include("grid.jl")
include("gridfields.jl")
include("freebc.jl")
include("poisson.jl")
include("fluid.jl")
include("io.jl")
include("initdens.jl")
include("plot.jl")

end # module
