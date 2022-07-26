const poisson_save_n = Ref(0)

"""
    Solve Poisson equation and compute the electrostatic fields.
"""
function poisson!(fields, mpopl, eb, mg, ws, denoiser, outpath)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! remove outpath when finished debugging
    g = mg.g
    @unpack grid, u, er, ez = fields
    @unpack M, N = grid
    
    fields.qpart .= 0
    
    density!(grid, fields.qpart, mpopl, Electron, -1.0)

    # Summing along threads: TODO: pre-allocate
    qfixed = zeros(Float32, M, N)
    qpart = zeros(Float32, M, N)
    
    ci = CartesianIndices(axes(fields.q))
    eps = 1e13
    
    for j in 1:N
        for i in 1:M
            # Adding a fixed density of 1e13 as a hack
            qfixed[i, j] = reduce(+, @view(fields.qfixed[i, j, :])) + eps
            qpart[i, j] = -reduce(+, @view(fields.qpart[i, j, :])) + eps

            # Until we implement using the charges as denoising input we use
            # yet another horrible hack to prevent negative densities in the fixed
            # charges, which are created by attachment.
            if qfixed[i, j] < 0
                qpart[i, j] -= qfixed[i, j] - eps
                qfixed[i, j] = eps
            end
        end
    end

    
    # De-noising
    qfixed1 = denoise(denoiser, qfixed)
    qpart1 = denoise(denoiser, qpart)

    # if (poisson_save_n[] % 50) == 0
    #     ofile = joinpath(outpath, "denoise_" * fmt("04d", poisson_save_n[]) * ".jld")
    
    #     jldsave(ofile, true; grid, qfixed, qfixed1, qpart, qpart1)
    #     @info "Saved file" ofile
    # end
    
    # poisson_save_n[] += 1


    fields.q[1:M, 1:N] .= qfixed1 .- qpart1
    
    
    Multigrid.solve(mg, parent(fields.u), parent(fields.q), ws)

    Threads.@threads for i in 1:(M + 1)
        for j in 1:(N + 1)
            er[i, j] = (u[i - 1, j] - u[i, j]) / dr(grid)
            ez[i, j] = eb + (u[i, j - 1] - u[i, j]) / dz(grid)
        end
    end

    ez[0,             :] .= ez[1,     :]
    ez[M + 1,         :] .= ez[M,     :]
    er[:,             0] .= er[:,     1]
    er[:,         N + 1] .= er[:,     N]
end


"""
    Update array `arr` with the densities of particles of type 
    `ptype` contained in `mpopl`.
"""
function density!(grid, arr, mpopl, ptype, val=1.0) where sym
    popl = get(mpopl, ptype)
    
    @inbounds Threads.@threads for p in eachparticle(popl)
        p.active || continue
        I = cellindext(grid, p.x)

        checkbounds(Bool, arr, I) || continue

        # arr is slightly larger because of padding so we have also to ckeck
        # the rc field for dV.
        checkbounds(Bool, grid.rc, I[1]) || continue
        
        arr[I] += p.w * val / dV(grid, I[1])
    end
end
