## autoencoder.jl : Thu Nov 25 11:29:27 2021
module Autoencoder

using PyCall
using HDF5
import PyPlot as plt

# Python modules; initialized later
models = PyNULL() # tensorflow.keras.models


"""
A struct with all the data required to de-noise an array with charge
density.
"""
struct Denoiser{T}
    model::PyObject

    nn_range::NTuple{2, T}
    q_range::NTuple{2, T}
end


""" 
    Denoiser(model_name::String, patch_size, step, nn_range, logdens_range)

Init a Denoiser struct wreading a saved keras model called `model_name`. 
"""
function Denoiser(model_name::String, args...)
    model = models.load_model(model_name;
                              custom_objects=Dict("custom_loss_4" => x -> nothing))

    return Denoiser{Float32}(model, args...)
end


"""
    denoise(d::Denoiser, q)

Use the denoiser `d` to remove noise from the charge density `q`.
"""
function denoise(d::Denoiser{T}, q) where T
    q1 = reshape(q, (1, size(q)..., 1))
    
    normq = rescale.(q1, Ref(d.q_range), Ref(d.nn_range))

    pypred = pycall(d.model.predict, PyArray, PyReverseDims(normq))
    pypred .= rescale.(pypred, Ref(d.nn_range), Ref(d.q_range))

    # move to julia Array (with copy)
    pred = copy((@view pypred[1, :, :, 1])')

    return pred
end


""" 
    rescale(x, (a1, b1), (a2, b2))

Rescale a number x such that the interval (a1, b1) is mapped to (a2, b2). 
"""
rescale(x, (a1, b1), (a2, b2)) = a2 + (x - a1) * (b2 - a2) / (b1 - a1)
    

function load_density(fname)
    local ne
    h5open(fname, "r") do fid
        # First dataset from first group.
        group = fid[keys(fid)[1]]
        ne = Array(group[keys(group)[1]])
    end

    
    return ne
end

"""
    A Null denoiser that does nothing.
"""
struct NullDenoiser; end

denoise(d::NullDenoiser, ne) = ne


function __init__()    
    copy!(models, pyimport("tensorflow.keras.models"))
    @info "tensorflow.keras.models imported"
end

end

