"""
    Resample the population `popl` using  Russian roulette and splitting to 
    ensure that in each cell no more than nmax super-particles remain.  
    The weights of the removed particles are distributed among all remaining 
    particles cell-wise.
"""
function resample!(popl, fields, nmax)
    (;grid, p, w) = fields

    w .= 0.0
    p .= 0
    
    for i in 1:popl.n[]
        k = LazyRow(popl.particles, i)
        k.active || continue

        I = cellindex(grid, k.x)
        checkbounds(Bool, p, I) || continue

        p[I] += 1
        
        if p[I] > nmax
            w[I] += k.w
            remove_particle!(popl, i)
        end
    end

    for i in 1:popl.n[]
        k = LazyRow(popl.particles, i)
        k.active || continue

        I = cellindex(grid, k.x)
        checkbounds(Bool, p, I) || continue

        if p[I] > nmax
            # Russian roulette
            wnew = k.w + w[I] / nmax
            k.w = wnew
        elseif p[I] < nmax && k.w > 2
            # Splitting
            wnew = round(k.w / 2)
            index = add_particle!(popl, popl.particles[i])
            popl.particles.w[index] = wnew
            k.w = k.w - wnew
            p[I] += 1
        end
    end
end


""" 
    Average the cylindrical coordinates of two position given by cartesian
    coordinates `x1` and `x2` using normalized weights `w1` and `w2`.
"""
@inline function cylavg(x1, x2, nw1, nw2)
    r1, r2 = rcyl(x1), rcyl(x2)
    θ1, θ2 = atan(x1[2], x1[1]), atan(x2[2], x2[1])

    rm = nw1 * r1 + nw2 * r2
    θm = nw1 * θ1 + nw2 * θ2
    zm = nw1 * x1[3] + nw2 * x2[3]
    sinθm, cosθm = sincos(θm)

    @SVector [rm * cosθm, rm * sinθm, zm]
end

"""
    Randomly choose a particle depending on normalized weights nw1, nw2
    (normalized such that nw1 + nw2 = 1)
"""
@inline function choose(x1, x2, p1, p2, nw1, nw2)
    if rand() < nw1
        return (x1, p1)
    else
        return (x2, p2)
    end
end