## autoencoder.jl : Thu Nov 25 11:29:27 2021
module Autoencoder

using PyCall

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

    # The denoising will be active only after this time
    activ_time::T
end


Denoiser(model, nn_range, q_range) = Denoiser(model, nn_range, q_range, 0)

""" 
    Denoiser(model_name::String, args...)

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

    # move to julia Array (with copy)
    pred = copy((@view pypred[1, :, :, 1])')

    return pred
end

function denoise(d::Denoiser, q, t)
    if t >= d.activ_time
        return denoise(d, q)
    else
        return q
    end
end


""" 
    rescale(x, (a1, b1), (a2, b2))

Rescale a number x such that the interval (a1, b1) is mapped to (a2, b2). 
"""
rescale(x, (a1, b1), (a2, b2)) = a2 + (x - a1) * (b2 - a2) / (b1 - a1)
    

"""
    A Null denoiser that does nothing.
"""
struct NullDenoiser; end

denoise(d::NullDenoiser, ne) = ne
denoise(d::NullDenoiser, ne, t) = denoise(d, ne)


function __init__()    
    copy!(models, pyimport("tensorflow.keras.models"))
    @info "tensorflow.keras.models imported"
end

end

