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

    rscale::Bool
    
    # The denoising will be active only after this time
    activ_time::T

    # A hack to fiz boundary issues at r = R.
    fix_bnd_hack::Bool
end


Denoiser{T}(model::PyObject, nn_range, q_range, rscale) where {T} = Denoiser(model, nn_range, q_range, rscale, 0, true)
Denoiser{T}(model::PyObject, nn_range, q_range, rscale, activ_time) where {T} = Denoiser(model, nn_range, q_range, rscale, activ_time, true)

""" 
    Denoiser(model_name::String, args...)

Init a Denoiser struct wreading a saved keras model called `model_name`. 
"""
function Denoiser(model_name::String, args...)
    model = models.load_model(model_name;
                              compile=false,
                              custom_objects=Dict("custom_loss_4" => x -> nothing,
                                                  "cc_loss" => x -> nothing))
    model.compile()
    return Denoiser{Float32}(model, args...)
end


"""
    denoise(d::Denoiser, q)

Use the denoiser `d` to remove noise from the charge density `q`.
"""
function denoise(d::Denoiser{T}, q) where T
    q1 = reshape(q, (1, size(q)..., 1))
    
    normq = rescale.(q1, Ref(d.q_range), Ref(d.nn_range))
    d.rscale && addrscale(normq)
    
    pypred = pycall(d.model.predict, PyArray, PyReverseDims(normq))

    # move to julia Array (with copy)
    pred = copy((@view pypred[1, :, :, 1])')

    if d.fix_bnd_hack
        pred[end - 1, :] .= 0
    end
    
    d.rscale && rmrscale(pred)

    return pred
end

function denoise(d::Denoiser, q, t)
    if t >= d.activ_time
        return denoise(d, q)
    else
        return q
    end
end

function addrscale(q)
    for j in size(q, 3)
        for i in size(q, 2)
            q[1, i, j, 1] *= (i - 0.5)
        end
    end
end

function rmrscale(q)
    for j in size(q, 2)
        for i in size(q, 1)
            q[i, j] /= (i - 0.5)
        end
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

