#=
  Plotting functions.
=#

function plot1(fields, var::String; titleprefix="", rlim=nothing, zlim=nothing,
               clim=nothing, savedir=nothing, charge_scale=1, kw...)
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


    function f_efield()
        eabs = @. @views sqrt(0.25 * (fields.er[i1:i2, j1:j2] + fields.er[(i1 + 1):(i2 + 1), j1:j2])^2 +
                              0.25 * (fields.ez[i1:i2, j1:j2] + fields.ez[i1:i2, (j1 + 1):(j2 + 1)])^2)


        plt.figure("$titleprefix Electric field")
        plt.clf()
        (vmin, vmax) = _vlim((nothing, nothing), clim)
        plt.pcolormesh(zf[j1:(j2 + 1)] ./ co.milli,
                       rf[i1:(i2 + 1)] ./ co.milli,
                       eabs, cmap="gnuplot2"; vmin, vmax, kw...)
        cbar = plt.colorbar(label="Electric field (V/m)")
        
    end

    function f_edensity()
        ne = @views -fields.qpart[i1:i2, j1:j2, begin:end]
        (vmin, vmax) = _vlim((1e15, 1e21), clim)
        lognorm = plt.matplotlib.colors.LogNorm(;vmin, vmax)
        plt.figure("$titleprefix Electron density")
        plt.clf()
        plt.pcolormesh(zf[j1:(j2 + 1)] ./ co.milli,
                       rf[i1:(i2 + 1)] ./ co.milli,
                       ne, cmap="gnuplot2", norm=lognorm; kw...)
        cbar = plt.colorbar(label="Electron density (m\$^{-3}\$)")
    end
    
    function f_charge()
        q = @view(fields.q[i1:i2, j1:j2])
        q .*= charge_scale
        qmax, qmin = extrema(q)
        absmax = max(abs(qmax), abs(qmin)) * charge_scale
        (vmin, vmax) = _vlim((-absmax, absmax), clim)
        
        plt.figure("$titleprefix Charge density")
        plt.clf()
        plt.pcolormesh(zf[j1:(j2 + 1)] ./ co.milli,
                       rf[i1:(i2 + 1)] ./ co.milli, q;
                       cmap="seismic", vmin=vmin, vmax=vmax, kw...)
        cbar = plt.colorbar(label=L"Charge density (C/m$^3$)")        
    end

    function f_charge0()
        q = @view(fields.q0[i1:i2, j1:j2])
        q .*= charge_scale
        qmax, qmin = extrema(q)
        absmax = max(abs(qmax), abs(qmin)) * charge_scale
        (vmin, vmax) = _vlim((-absmax, absmax), clim)

        plt.figure("$titleprefix Noisy charge density")
        plt.clf()
        plt.pcolormesh(zf[j1:(j2 + 1)] ./ co.milli,
                       rf[i1:(i2 + 1)] ./ co.milli, q,
                       cmap="seismic", vmin=vmin, vmax=vmax, kw...)
        cbar = plt.colorbar(label=L"Charge density (C/m$^3$)")        
    end

    
    Dict(["edensity" => f_edensity,
          "efield" => f_efield,
          "charge" => f_charge,
          "charge0" => f_charge0])[var]()
    
    if rlim != nothing && zlim!= nothing
        setlims(rlim / co.milli, zlim / co.milli)
    end
    xylabel()

    if !isnothing(savedir)
        fname = joinpath(savedir, "$(var).png")
        @info "Saving plot to" fname
        plt.savefig(fname, dpi=600)
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


function plot(fname::String; save=false, vars=["edensity", "efield", "charge"], kw...)
    fields = load(fname, "fields");
    savedir = save ? splitext(fname)[1] : nothing
    for var in vars
        plot1(fields, var; savedir, kw...)
    end
end

function plot1(fname::String, var::String; save=false, kw...)
    fields = load(fname, "fields");

    savedir = save ? splitext(fname)[1] : nothing

    plot1(fields, var; savedir, kw...)
end

function setlims(rlim, zlim)
    setrlim(rlim)
    setzlim(zlim)
end

setrlim(rlim::Nothing) = nothing
setrlim(rlim::AbstractVector) = plt.ylim(rlim)
setzlim(zlim::Nothing) = nothing
setzlim(zlim::AbstractVector) = plt.xlim(zlim)

_vlim(vlim1, vlim2::Nothing) = vlim1
_vlim(vlim1, vlim2::AbstractVector) = vlim2
_vlim(vlim1, vlim2::Tuple) = vlim2


