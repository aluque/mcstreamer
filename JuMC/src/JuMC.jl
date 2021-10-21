module JuMC

using Base.Threads: Atomic, @threads, atomic_add!
using Parameters
using LinearAlgebra
using StaticArrays
using Interpolations

import JSON

include("constants.jl")
const co = Constants

include("util.jl")
include("particledefs.jl")
include("population.jl")
include("collisions.jl")
include("collpop.jl")
include("electron.jl")
include("lxcat.jl")

end