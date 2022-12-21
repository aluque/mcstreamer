#=
Init densities from files.
=#

"""
    Load the corresponding files from `nefile` and `qfile` and sample electrons with them, using a maximum
    of particles per cell ncellmax.
"""
function initfromfiles!(fields, nefile, qfile, ncellmax::Int)
    (;grid, qfixed) = fields
    (;M, N) = grid

    hne = h5open(nefile)
    hq = h5open(qfile)

    ne = Array(hne["group1/electron_density"])
    q = Array(hq["group1/charge_density"])
    att = attrs(hq["group1"])

    @assert size(ne) == (M, N) "Size $(size(ne)) != (M=$M, ,N=$N)"
    @assert isapprox(att["x.delta"], dr(grid))
    @assert isapprox(att["y.delta"], dz(grid))
    
    s = ElectronState{Float64}[]
    v0 = @SVector zeros(Float64, 3)

    for j in 1:N, i in 1:M
        V = dV(grid, i)
        n = ne[i, j] * V
        
        if n < 1
            nn = rand(Poisson(n))
            w = 1.0
        elseif n < ncellmax
            nn = randround(n)
            w = 1.0
        else
            nn = ncellmax
            w = n / ncellmax
        end
        qfixed[i, j, 1] = q[i, j] / co.elementary_charge + w * nn / V
        
        for p in 1:nn
            r = dr(grid) * rand() + grid.rf[i]
            z = dz(grid) * rand() + grid.zf[j]
            phi = 2π * rand()
            x = @SVector [r * cos(phi), r * sin(phi), z]
            push!(s, ElectronState(x, v0, w))
        end
        if length(s) > 10^9
            @info "Particle limit reached" i j
            break 
        end
    end
    
    return s
end

"""
    Randomly rounds a number `x` to `floor(x)` or `ceil(x)` depending on the fractional part of
    x (e.g. 3.1 has 90% chance of giving 3, 10% of giving 4).
"""
function randround(x)
    m = floor(x)
    mint = convert(Int, m)
    
    f = x - m
    if rand() > f
        return mint
    else
        return mint + 1
    end
end

