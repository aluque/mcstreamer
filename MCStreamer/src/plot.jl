#=
  Plotting functions.
=#

function plot(fields::GridFields; rlim=nothing, zlim=nothing, kw...)
    plt.matplotlib.pyplot.style.use("granada")
    M = fields.grid.M
    N = fields.grid.N
    # qfixed_t = dropdims(sum(@view(fields.qfixed[1:M, 1:N, :]), dims=3), dims=3)
    # qpart_t = dropdims(sum(@view(fields.qpart[1:M, 1:N, :]), dims=3), dims=3)

    # q = qfixed_t .- qpart_t
    @unpack rf, zf, rc, zc = fields.grid
    
    (i1, i2) = indexlims(rlim, rc, M)
    (j1, j2) = indexlims(zlim, zc, N)

    plt.figure("Charge density")
    q = @view(fields.q[i1:i2, j1:j2])
    qmax, qmin = extrema(q)
    absmax = max(abs(qmax), abs(qmin)) * co.elementary_charge
    
    plt.pcolormesh(zf[j1:(j2 + 1)], rf[i1:(i2 + 1)], q .* co.elementary_charge;
                   cmap="seismic", vmin=-absmax, vmax=absmax, kw...)
    cbar = plt.colorbar(label=L"Charge density (C/m$^3$)")
    setlims(rlim, zlim)
    
    plt.figure("Electric field")
    eabs = @. @views sqrt(0.25 * (fields.er[i1:i2, j1:j2] + fields.er[(i1 + 1):(i2 + 1), j1:j2])^2 +
                          0.25 * (fields.ez[i1:i2, j1:j2] + fields.ez[i1:i2, (j1 + 1):(j2 + 1)])^2)

    plt.pcolormesh(zf[j1:(j2 + 1)], rf[i1:(i2 + 1)], eabs, cmap="gnuplot2"; kw...)
    cbar = plt.colorbar(label="Electric field (V/m)")
    setlims(rlim, zlim)

    plt.figure("Electron density")
    ne = @views -dropdims(sum(fields.qpart[i1:i2, j1:j2, begin:end], dims=3), dims=3)
    lognorm = plt.matplotlib.colors.LogNorm(vmin=1e15, vmax=1e21)
    plt.pcolormesh(zf[j1:(j2 + 1)], rf[i1:(i2 + 1)], ne, cmap="gnuplot2", norm=lognorm; kw...)
    cbar = plt.colorbar(label="Electron density (m\$^{-3}\$)")
    setlims(rlim, zlim)

    setlims(rlim, zlim)

    
    plt.show()
end

function indexlims(lim::AbstractVector, x, n)
    i1 = searchsortedfirst(x, lim[1])
    i2 = searchsortedlast(x, lim[2])
    return (i1, i2)
end

function indexlims(lim::Nothing, x, n)
    return (1, n)
end


function plot(fname::String; kw...)
    fields = load(fname, "fields");
    plot(fields; kw...)
end

function setlims(rlim, zlim)
    setrlim(rlim)
    setzlim(zlim)
end

setrlim(rlim::Nothing) = nothing
setrlim(rlim::AbstractVector) = plt.ylim(rlim)
setzlim(zlim::Nothing) = nothing
setzlim(zlim::AbstractVector) = plt.xlim(zlim)


