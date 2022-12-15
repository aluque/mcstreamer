#=
  Plotting functions.
=#

function plot(fields; titleprefix="", rlim=nothing, zlim=nothing,
              savedir=nothing, charge_scale=co.elementary_charge, kw...)
    plt.matplotlib.pyplot.style.use("granada")
    if !isnothing(savedir)
        isdir(savedir) || mkpath(savedir)
    end

    M = fields.grid.M
    N = fields.grid.N
    # qfixed_t = dropdims(sum(@view(fields.qfixed[1:M, 1:N, :]), dims=3), dims=3)
    # qpart_t = dropdims(sum(@view(fields.qpart[1:M, 1:N, :]), dims=3), dims=3)

    # q = qfixed_t .- qpart_t
    @unpack rf, zf, rc, zc = fields.grid
    
    function xylabel()
        plt.xlabel("z (mm)")
        plt.ylabel("r (mm)")
    end
    
    (i1, i2) = indexlims(rlim, rc, M)
    (j1, j2) = indexlims(zlim, zc, N)

    plt.figure("$titleprefix Charge density")
    plt.clf()
    q = @view(fields.q[i1:i2, j1:j2])
    qmax, qmin = extrema(q)
    absmax = max(abs(qmax), abs(qmin)) * charge_scale
    
    plt.pcolormesh(zf[j1:(j2 + 1)] ./ co.milli,
                   rf[i1:(i2 + 1)] ./ co.milli,
                   q .* charge_scale;
                   cmap="seismic", vmin=-absmax, vmax=absmax, kw...)
    cbar = plt.colorbar(label=L"Charge density (C/m$^3$)")

    if rlim != nothing && zlim!= nothing
        setlims(rlim / co.milli, zlim / co.milli)
    end
    
    xylabel()
    
    if !isnothing(savedir)
        fname = joinpath(savedir, "charge.png")
        @info "Saving plot to" fname
        plt.savefig(fname, dpi=600)
    end
    
    plt.figure("$titleprefix Electric field")
    plt.clf()
    eabs = @. @views sqrt(0.25 * (fields.er[i1:i2, j1:j2] + fields.er[(i1 + 1):(i2 + 1), j1:j2])^2 +
                          0.25 * (fields.ez[i1:i2, j1:j2] + fields.ez[i1:i2, (j1 + 1):(j2 + 1)])^2)

    plt.pcolormesh(zf[j1:(j2 + 1)] ./ co.milli,
                   rf[i1:(i2 + 1)] ./ co.milli,
                   eabs, cmap="gnuplot2"; kw...)
    cbar = plt.colorbar(label="Electric field (V/m)")
    if rlim != nothing && zlim!= nothing
        setlims(rlim / co.milli, zlim / co.milli)
    end
    xylabel()

    if !isnothing(savedir)
        fname = joinpath(savedir, "efield.png")
        @info "Saving plot to" fname
        plt.savefig(fname, dpi=600)
    end

    plt.figure("$titleprefix Electron density")
    plt.clf()
    ne = @views -dropdims(sum(fields.qpart[i1:i2, j1:j2, begin:end], dims=3), dims=3)
    lognorm = plt.matplotlib.colors.LogNorm(vmin=1e15, vmax=1e21)
    plt.pcolormesh(zf[j1:(j2 + 1)] ./ co.milli,
                   rf[i1:(i2 + 1)] ./ co.milli,
                   ne, cmap="gnuplot2", norm=lognorm; kw...)
    cbar = plt.colorbar(label="Electron density (m\$^{-3}\$)")
    if rlim != nothing && zlim!= nothing
        setlims(rlim / co.milli, zlim / co.milli)
    end
    xylabel()
    
    
    if !isnothing(savedir)
        fname = joinpath(savedir, "edensity.png")
        @info "Saving plot to" fname
        plt.savefig(fname, dpi=600)
    end
    
    if isnothing(savedir)
        plt.show()
    end
end

function indexlims(lim::AbstractVector, x, n)
    i1 = searchsortedfirst(x, lim[1])
    i2 = searchsortedlast(x, lim[2])
    return (i1, i2)
end

function indexlims(lim::Nothing, x, n)
    return (1, n)
end


function plot(fname::String; save=false, kw...)
    fields = load(fname, "fields");

    savedir = save ? splitext(fname)[1] : nothing

    plot(fields; savedir, kw...)
end

function setlims(rlim, zlim)
    setrlim(rlim)
    setzlim(zlim)
end

setrlim(rlim::Nothing) = nothing
setrlim(rlim::AbstractVector) = plt.ylim(rlim)
setzlim(zlim::Nothing) = nothing
setzlim(zlim::AbstractVector) = plt.xlim(zlim)


