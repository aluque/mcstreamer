#=
  Plotting functions.
=#

function plot(fields; rlim=nothing, zlim=nothing, kw...)
    plt.matplotlib.pyplot.style.use("granada")
    M = fields.grid.M
    N = fields.grid.N
    # qfixed_t = dropdims(sum(@view(fields.qfixed[1:M, 1:N, :]), dims=3), dims=3)
    # qpart_t = dropdims(sum(@view(fields.qpart[1:M, 1:N, :]), dims=3), dims=3)

    # q = qfixed_t .- qpart_t
    @unpack rf, zf, rc, zc = fields.grid
    
    plt.figure("Charge density")
    q = @view(fields.q[1:M, 1:N])
    qmax, qmin = extrema(q)
    absmax = max(abs(qmax), abs(qmin)) * co.elementary_charge
    
    plt.pcolormesh(zf, rf, q .* co.elementary_charge;
                   cmap="seismic", vmin=-absmax, vmax=absmax, kw...)
    cbar = plt.colorbar(label=L"Charge density (C/m$^3$)")
    setlims(rlim, zlim)
    
    plt.figure("Electric field")
    eabs = @. @views sqrt(0.25 * (fields.er[1:M, 1:N] + fields.er[2:(M + 1), 1:N])^2 +
                          0.25 * (fields.ez[1:M, 1:N] + fields.ez[1:M, 2:(N + 1)])^2)

    plt.pcolormesh(zf, rf, eabs, cmap="gnuplot2"; kw...)
    cbar = plt.colorbar(label="Electric field (V/m)")
    setlims(rlim, zlim)

    plt.figure("Electron density")
    ne = @views -dropdims(sum(fields.qpart[1:M, 1:N, begin:end], dims=3), dims=3)
    lognorm = plt.matplotlib.colors.LogNorm()
    plt.pcolormesh(zf, rf, ne, cmap="gnuplot2", vmin=1e15, vmax=1e21, norm=lognorm; kw...)
    cbar = plt.colorbar(label="Electron density (m\$^{-3}\$)")
    setlims(rlim, zlim)

    setlims(rlim, zlim)

    
    plt.show()
end

function setlims(rlim, zlim)
    setrlim(rlim)
    setzlim(zlim)
end

setrlim(rlim::Nothing) = nothing
setrlim(rlim::AbstractVector) = plt.ylim(rlim)
setzlim(zlim::Nothing) = nothing
setzlim(zlim::AbstractVector) = plt.xlim(zlim)


