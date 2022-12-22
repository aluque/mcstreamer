#=
Init densities from files.
=#

struct NoiselessInitSampling
    p::Int
end

struct PoissonInitSampling
    p::Int
end

"""
    Load the corresponding files from `nefile` and `qfile` and sample electrons with them, using a maximum
    of particles per cell ncellmax.
"""
function initfromfiles!(method, fields, nefile, qfile)
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
        nume, qf, w = nsample(method, ne[i, j], q[i, j], V)
        qfixed[i, j, 1] = qf
        
        for p in 1:nume
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
    Sample a number of electrons and weight from electron and ion densities `ne` and `ni` in a
    volume `V`.
"""
function nsample(method::NoiselessInitSampling, ne, q, V)
    p = method.p
    n = ne * V
        
    if n < 1
        nume = rand(Poisson(n))
        w = 1.0
    elseif n < p
        nume = randround(n)
        w = 1.0
    else
        nume = p
        w = n / p
    end
    qf = q / co.elementary_charge + w * nume / V

        
    return nume, qf, w
end


function nsample(method::PoissonInitSampling, ne, q, V)
    p = method.p
    
    w = max(1, ne * V / p)
    nume = rand(Poisson(ne * V / w))

    ni = ne + q / co.elementary_charge
    wi = max(1, abs(ni) * V / p)

    numi = rand(Poisson(abs(ni) * V / wi))
        
    qf = sign(ni) * numi * wi / V
        
    return nume, qf, w
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

