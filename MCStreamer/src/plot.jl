#=
  Plotting functions.
=#

function plot1(fields, var::String; titleprefix="", rlim=nothing, zlim=nothing,
               clim=nothing, savedir=nothing, charge_scale=1, grid=fields.grid, kw...)
    plt.matplotlib.pyplot.style.use("granada")
    if !isnothing(savedir)
        isdir(savedir) || mkpath(savedir)
    end

    M = grid.M
    N = grid.N
    # qfixed_t = dropdims(sum(@view(fields.qfixed[1:M, 1:N, :]), dims=3), dims=3)
    # qpart_t = dropdims(sum(@view(fields.qpart[1:M, 1:N, :]), dims=3), dims=3)

    # q = qfixed_t .- qpart_t
    @unpack rf, zf, rc, zc = grid
    
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
        ne = @views -fields.qpart[i1:i2, j1:j2]
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


function h5export(fields::SavedGridFields, filename::String)
    h5file = h5open(filename, "w")

    # Save Grid
    g_grid = create_group(h5file, "grid")
    write(g_grid, "R", fields.grid.R)
    write(g_grid, "L", fields.grid.L)
    write(g_grid, "M", fields.grid.M)
    write(g_grid, "N", fields.grid.N)
    write(g_grid, "rc", collect(fields.grid.rc))
    write(g_grid, "rf", collect(fields.grid.rf))
    write(g_grid, "zc", collect(fields.grid.zc))
    write(g_grid, "zf", collect(fields.grid.zf))

    (;M, N) = fields.grid
    
    # Save OffsetMatrix arrays
    write(h5file, "qfixed", fields.qfixed[1:M, 1:N])
    write(h5file, "qpart", fields.qpart[1:M, 1:N])
    write(h5file, "q0", fields.q0[1:M, 1:N])
    write(h5file, "q", fields.q[1:M, 1:N])
    write(h5file, "u", fields.u[1:M, 1:N])
    write(h5file, "er", fields.er[1:M + 1, 1:N])
    write(h5file, "ez", fields.ez[1:M, 1:N + 1])
    write(h5file, "p", fields.p[1:M, 1:N])
    write(h5file, "wtotal", fields.wtotal[1:M, 1:N])

    close(h5file)
end

h5export(from::String, to::String) = h5export(load(from, "fields"), to)

setrlim(rlim::Nothing) = nothing
setrlim(rlim::AbstractVector) = plt.ylim(rlim)
setzlim(zlim::Nothing) = nothing
setzlim(zlim::AbstractVector) = plt.xlim(zlim)

_vlim(vlim1, vlim2::Nothing) = vlim1
_vlim(vlim1, vlim2::AbstractVector) = vlim2
_vlim(vlim1, vlim2::Tuple) = vlim2


