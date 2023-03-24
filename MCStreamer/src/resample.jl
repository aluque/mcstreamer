"""
    Resample the population `popl` using  Russian roulette and splitting to 
    ensure that in each cell no more than nmax super-particles remain.  
    The weights of the removed particles are distributed among all remaining 
    particles cell-wise.
"""
function resample!(popl, fields, ntarget, nmax)
    (;grid, x, p, pk, wtotal, wcum, locks) = fields
    wcum .= 0.0
    wtotal .= 0.0
    x .= 0
    
    p .= 0
    pk .= 0
    # First pass: compute p, wtotal, wmax for each cell
    for i in 1:popl.n[]
        k = LazyRow(popl.particles, i)
        k.active || continue

        I = cellindex(grid, k.x)
        checkbounds(Bool, p, I) || continue

        p[I] += 1
        wtotal[I] += k.w

        if p[I] == nmax
            x[I] = samplemin(ntarget)
        end
    end
    
    @batch for i in 1:popl.n[]
        k = LazyRow(popl.particles, i)
        k.active || continue
        
        I = cellindex(grid, k.x)
        checkbounds(Bool, p, I) || continue
        
        
        lock(locks[I])
        #= 
        A short description of the Russian roulette resampling that we do here:
        When the number of particles in a cell exceeds nmax we resample them into ntarget 
        particles of equal weight. What we do is equivalent to the following: arrange the 
        weights sequentially and normalize them such that their sum adds to 1. Then draw
        ntarget samples from a uniform distribution in [0, 1].  The chosen particles are those
        where the samples fall and a particle may be chosen more than once. In this manner the
        number of times that a particle is selected is proportional to its weight.
        For performance reasons we do not draw all the samples at once but consecutively;
        this is achieved with `samplemin(m)`, which samples from the distribution of the minimum
        of `m` uniformly-distributed, independent samples.
        =#
        @inbounds begin
            if p[I] > nmax 
                while (wtotal[I] * x[I] < wcum[I] + k.w) && pk[I] < ntarget
                    pk[I] += 1
                    index = add_particle!(popl, popl.particles[i])
                    popl.particles.w[index] = wtotal[I] / ntarget
                    if pk[I] < ntarget
                        x[I] += samplemin(ntarget - pk[I]) * (1 - x[I])
                    end
                end
                wcum[I] += k.w
                remove_particle!(popl, i)
                
            elseif p[I] < ntarget && k.w > 2
                # Splitting
                wnew = round(k.w / 2)
                index = add_particle!(popl, popl.particles[i])
                popl.particles.w[index] = wnew
                k.w = k.w - wnew
                
                p[I] += 1
            end
        end
        unlock(locks[I])
    end
end


"""
Sample from the probability distribution of the minimum of `m` draws from the uniform
distribution in (0, 1).
"""
samplemin(m) = 1 - (1 - rand())^(1 / m)


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
