const poisson_save_n = Ref(0)

"""
    Solve Poisson equation and compute the electrostatic fields.
"""
function poisson!(fields, mpopl, eb, mg, ws, denoiser, t, outpath)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! remove outpath when finished debugging
    g = mg.g
    @unpack grid, u, er, ez, q0, q = fields
    @unpack M, N = grid
    
    fields.qpart .= 0
    
    density!(grid, fields.qpart, mpopl, Electron, fields, -1.0)

    # Summing along threads
    q0 .= 0
    
    for j in 1:N
        for i in 1:M
            q0[i, j]  = reduce(+, @view(fields.qfixed[i, j, :]))
            q0[i, j] += reduce(+, @view(fields.qpart[i, j, :]))
            q0[i, j] -= fields.ne[i, j]
            q0[i, j] *= co.elementary_charge
        end
    end

    
    # De-noising.  We have to do this inefficient array allocations for compatibility
    # with the tensorflow api.   
    q1 = zeros(Float32, M, N)
    q1 .= @view q0[1:M, 1:N]
    
    q[1:M, 1:N] .= denoise(denoiser, q1, t)

    # if (poisson_save_n[] % 50) == 0
    #     ofile = joinpath(outpath, "denoise_" * fmt("04d", poisson_save_n[]) * ".jld")
    
    #     jldsave(ofile, true; grid, qfixed, qfixed1, qpart, qpart1)
    #     @info "Saved file" ofile
    # end
    
    # poisson_save_n[] += 1
    
    solve_freebc!(parent(fields.u), parent(fields.q), grid, mg, ws)
    
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
function density!(grid, arr, mpopl, ptype, fields, val=1.0) where sym
    popl = get(mpopl, ptype)
    
    @inbounds Threads.@threads for p in eachparticle(popl)
        p.active || continue

        I = cellindext(grid, p.x)

        checkbounds(Bool, arr, I) || continue

        # arr is slightly larger because of padding so we have also to ckeck
        # the rc field for dV.
        checkbounds(Bool, grid.rc, I[1]) || continue
        
        lock(fields.thread_locks[I[length(I)]])
        arr[I] += p.w * val / dV(grid, I[1])
        unlock(fields.thread_locks[I[length(I)]])

    end
end
