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


