# Support for dispatching to different devices.
# To dispatch to CPU or GPU we use one of these types
abstract type AbstractDevice end

struct CPU <: AbstractDevice end
struct GPU <: AbstractDevice end

dzeros(dev::CPU, args...) = zeros(args...)
dzeros(dev::GPU, args...) = CUDA.zeros(args...)

dfill(dev::CPU, args...) = fill(args...)
dfill(dev::GPU, args...) = CUDA.fill(args...)

dpow(dev::CPU, a, b) = a^b
dpow(dev::GPU, a, b) = CUDA.pow(a, b)

initdevice(dev::CPU) = nothing
function initdevice(dev::GPU)    
    idev = parse(Int, get(ENV, "CUDA_DEVICE", "0"))
    @info "Selecting CUDA device $(idev)"
    CUDA.device!(idev)
end


@inline function applykern(dev::CPU, f, g, lims)
    Threads.@threads for j in (g + 1):(g + lims[2])
        for i in (g + 1):(g + lims[1])
            f(i, j)
        end
    end
end

@inline function applykern(dev::GPU, f, g, lims)
    @cuda(threads=BLCK,
          blocks=blocks.(lims, BLCK),
          cudaapply(f, g, lims))        
end

function blocks(n, bsize)
    return div(n, bsize, RoundUp)
end

function cudaapply(f, g, lims)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (1 <= i <= lims[1] && 1 <= i <= lims[2])
        f(g + i, g + j)
    end

    nothing
end

# Atomics
" Allocate space for an atomic element with type and initial value from `x`."
datomic(::CPU, x) = Threads.Atomic{typeof(x)}(x)
datomic(::GPU, x) = CUDA.fill(x, 1)

# Set outside kernels.  Not really needed
# datomicset!(::CPU, a, val) = a[] = val
# datomicset!(::GPU, a, val) = a[] = val

# Set in kernels
@inline datomicmin!(::CPU, a, val) = a[] = min(a[], val)
@inline datomicmin!(::GPU, a, val) = @atomic a[] = min(a[], val)
