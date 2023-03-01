#=
I/O operations.
=#

"""
A container for saving the data. Compared with GridFields this allows us to save disk space and
transfer time.
"""
struct SavedGridFields{T,A<:AbstractArray{T},AI<:AbstractArray{Int}}
    grid::Grid{T}

    # Fixed charges
    qfixed::A
    
    # Charges associated with moving particles
    qpart::A
    
    # Pre-computed charge density.  Differs from q only without denoising
    q0::A
    
    # Charge density
    q::A

    # Electrostatic potential
    u::A
    
    # r-component of the electric field
    er::A

    # z-component of the electric field
    ez::A

    # For Russian roulette; counter of particles inside each cell.
    p::AI

    # For Russian roulette: total weight inside a cell
    wtotal::A
end

function SavedGridFields(gf::GridFields)
    args = map(fieldnames(SavedGridFields)) do name
        v = getfield(gf, name)
        name == :grid && (return v)
        
        if ndims(v) == 2
            return v
        else
            return dropdims(sum(v, dims=3), dims=3)
        end
    end
    return SavedGridFields(args...)
end


function savefields(fname, fields)
    jldsave(fname, false; iotype=IOStream, fields=SavedGridFields(fields))
    @info "Saved file" fname
end
