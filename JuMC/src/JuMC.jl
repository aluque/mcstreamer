module JuMC

using Base.Threads: Atomic, @threads, atomic_add!
using Parameters
using LinearAlgebra
using StaticArrays
using Interpolations
using Polyester
using LoopVectorization
using StructArrays
using DocStringExtensions

import JSON

include("constants.jl")
const co = Constants

@template DEFAULT =
    """
    $(SIGNATURES)
    $(DOCSTRING)
    """

include("util.jl")
include("particledefs.jl")
include("population.jl")
include("mixed_population.jl")
include("collisions.jl")
#include("collpop.jl")
include("electron.jl")
include("photon.jl")
#include("electron_population.jl")
include("lxcat.jl")

end
