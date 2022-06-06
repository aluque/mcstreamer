""" 
    randsphere()

    Sample points from the unitary sphere. 
"""
function randsphere()
    ϕ = 2π * rand()
    sinϕ, cosϕ = sincos(ϕ)
    
    u = 2 * rand() - 1
    v = sqrt(1 - u^2)

    @SVector [v * cosϕ, v * sinϕ, u]
end

"""
    nexcol(ν)

    Sample the next collision time for a particle with a collision rate `ν`.
"""
nextcol(ν) = -log(rand()) / ν


"""
    Return the index and weight of the item before x in a range.
"""
function indweight(r::AbstractRange, x)
    #i = searchsortedlast(r, x)
    i = Int(fld(x - first(r), step(r))) + 1
    #@assert ip == i "$ip != $i"
    w = (r[i + 1] - x) / step(r)
    @assert 0 <= w <= 1
    return (i, w)
end


