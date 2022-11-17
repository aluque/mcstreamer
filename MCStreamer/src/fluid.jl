# Routines for handling fluid electrons.

function setneedle!(fields, sigma_z, sigma_r, position_z, position_r, ne0)
    (;ne, qfixed, grid) = fields
    (;rc, zc, M, N) = grid
    
    @batch for j in 1:N, i in 1:M
        z1 = max(0, zc[j] - position_z)
        ne[i, j] += ne0 * exp(-rc[i]^2 / sigma_r^2 - z1^2 / sigma_z^2)
        qfixed[i, j, 1] += ne[i, j]
    end
end


"""
    Impose fluid boundary conditions.
"""
function setnebc!(fields)
    (;ne, grid) = fields
    (;M, N) = grid
    
    G = 2

    # Currently only Neumann bc implemented
    @views @inbounds begin
        ne[0,  :] = ne[1, :]
        ne[-1, :] = ne[1, :]

        ne[:,  0] = ne[:, 1]
        ne[:, -1] = ne[:, 1]

        ne[M + 1, :] = ne[M, :]
        ne[M + 2, :] = ne[M, :]

        ne[:, N + 1] = ne[:, N]
        ne[:, N + 2] = ne[:, N]
    end    
end


"""
    Compute the derivatives of the electron fluid density.
"""
function setdne!(fields, mobility=1e24 / co.nair)
    (;ne, dne, er, ez, grid) = fields
    (;M, N) = grid
    G = 2

    setnebc!(fields)
    
    dne .= 0
    
    # flux in z; i's can be parallelized
    for j in 1:N + 1
        @batch for i in 1:M
            v = -ez[i, j] * mobility
            sv = signbit(v) ? -1 : 1
            
            # The downstream, upstream and twice-upstream indices
            jd  = j + div(sv - 1, 2)
            ju  = j + div(-sv - 1, 2)
            ju2 = j + div(-3sv - 1, 2)
            
            θ = (ne[i, ju] - ne[i, ju2]) / (ne[i, jd] - ne[i, ju])
            F = v * (ne[i, ju] + koren_limiter(θ) * (ne[i, jd] - ne[i, ju]))

            dne[i, j] += F / dz(grid)
            dne[i, j - 1] -= F / dz(grid)
        end
    end

    # flux in r; j's can be parallelized
    for i in 1:M + 1
        @batch for j in 1:N
            v = -er[i, j] * mobility
            sv = signbit(v) ? -1 : 1
            
            # The downstream, upstream and twice-upstream indices
            id  = i + div(sv - 1, 2)
            iu  = i + div(-sv - 1, 2)
            iu2 = i + div(-3sv - 1, 2)
            
            θ = (ne[iu, j] - ne[iu2, j]) / (ne[id, j] - ne[iu, j])
            F = v * (ne[iu, j] + koren_limiter(θ) * (ne[id, j] - ne[iu, j]))

            dne[i, j] += F / dr(grid)
            dne[i - 1, j] -= F / dr(grid)
        end
    end    
end

"""
    Update the electron fluid density according to the derivatives previously computed
"""
function euler!(fields, dt)
    (;ne, dne, grid) = fields
    (;M, N) = grid

    @batch for j in 1:N
        for i in 1:M
            ne[i, j] += dt * dne[i, j]
        end
    end
end


""" The Koren limiter. """        
@inline function koren_limiter(theta)
    (theta >= 4.0) && return 1.0
    (theta > 0.4) && return 1.0 / 3.0 + theta / 6.0
    (theta > 0.0) && return theta
    0.0
end
    
