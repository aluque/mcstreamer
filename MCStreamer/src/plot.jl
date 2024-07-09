#=
  Plotting functions.
=#

function plot1(fields, var::String; titleprefix="", rlim=nothing, zlim=nothing,
               clim=nothing, savedir=nothing, savename=nothing, charge_scale=1, grid=fields.grid,
               resample=1, bin=nothing, log=false, kw...)
    plt.matplotlib.pyplot.style.use("granada")
    if !isnothing(savedir)
        isdir(savedir) || mkpath(savedir)
    end

    M = grid.M
    N = grid.N
    # qfixed_t = dropdims(sum(@view(fields.qfixed[1:M, 1:N, :]), dims=3), dims=3)
    # qpart_t = dropdims(sum(@view(fields.qpart[1:M, 1:N, :]), dims=3), dims=3)

    # q = qfixed_t .- qpart_t
    (;rf, zf, rc, zc) = grid

    function xylabel()
        plt.xlabel("z (mm)")
        plt.ylabel("r (mm)")
    end
    
    (i1, i2) = indexlims(rlim, rc, M)
    (j1, j2) = indexlims(zlim, zc, N)
    s = resample

    function f_efield(f)
        eabs = @. @views sqrt(0.25 * (f.er[i1:s:i2, j1:s:j2] +
                                      f.er[(i1 + 1):s:(i2 + 1), j1:s:j2])^2 +
                              0.25 * (f.ez[i1:s:i2, j1:s:j2] +
                                      f.ez[i1:s:i2, (j1 + 1):s:(j2 + 1)])^2)


        plt.figure("$titleprefix Electric field")
        plt.clf()
        (vmin, vmax) = _vlim((nothing, nothing), clim)
        if log
            vmin = max(vmin, vmax / 1e2)
            lognorm = plt.matplotlib.colors.LogNorm(;vmin, vmax)
            kw = (;norm=lognorm, kw...)
        else
            kw = (;vmin, vmax, kw...)
        end
        
        plt.pcolormesh(zf[j1:s:(j2 + 1)] ./ co.milli,
                       rf[i1:s:(i2 + 1)] ./ co.milli,
                       eabs, cmap="gnuplot2"; kw...)
        cbar = plt.colorbar(label="Electric field (V/m)")
        
    end

    function f_edensity(f)
        ne = @views -f.qpart[i1:s:i2, j1:s:j2]
        (vmin, vmax) = _vlim((1e15, 1e21), clim)
        lognorm = plt.matplotlib.colors.LogNorm(;vmin, vmax)
        plt.figure("$titleprefix Electron density")
        plt.clf()
        plt.pcolormesh(zf[j1:s:(j2 + 1)] ./ co.milli,
                       rf[i1:s:(i2 + 1)] ./ co.milli,
                       ne, cmap="gnuplot2", norm=lognorm; kw...)
        cbar = plt.colorbar(label="Electron density (m\$^{-3}\$)")
    end
    
    function f_charge(f)
        q = @view(f.q[i1:s:i2, j1:s:j2])
        q .*= charge_scale
        qmax, qmin = extrema(q)
        absmax = max(abs(qmax), abs(qmin)) * charge_scale
        (vmin, vmax) = _vlim((-absmax, absmax), clim)
        
        plt.figure("$titleprefix Charge density")
        plt.clf()
        plt.pcolormesh(zf[j1:s:(j2 + 1)] ./ co.milli,
                       rf[i1:s:(i2 + 1)] ./ co.milli, q;
                       cmap="bwr", vmin=vmin, vmax=vmax, kw...)
        cbar = plt.colorbar(label=L"Charge density (C/m$^3$)")        
    end

    function f_charge0(f)
        q = @view(f.q0[i1:s:i2, j1:s:j2])
        q .*= charge_scale
        qmax, qmin = extrema(q)
        absmax = max(abs(qmax), abs(qmin)) * charge_scale
        (vmin, vmax) = _vlim((-absmax, absmax), clim)

        plt.figure("$titleprefix Noisy charge density")
        plt.clf()
        plt.pcolormesh(zf[j1:s:(j2 + 1)] ./ co.milli,
                       rf[i1:s:(i2 + 1)] ./ co.milli, q,
                       cmap="bwr", vmin=vmin, vmax=vmax, kw...)
        cbar = plt.colorbar(label=L"Charge density (C/m$^3$)")        
    end

    function f_particles(f)
        p = @view(f.p[i1:s:i2, j1:s:j2])
        plt.figure("$titleprefix Super-particle number")
        plt.clf()
        plt.pcolormesh(zf[j1:s:(j2 + 1)] ./ co.milli,
                       rf[i1:s:(i2 + 1)] ./ co.milli, p;
                       cmap="gnuplot2", kw...)
        cbar = plt.colorbar(label="Number of super-particles per cell")
    end
    
    
    Dict("edensity" => f_edensity,
         "efield" => f_efield,
         "charge" => f_charge,
         "charge0" => f_charge0,
         "p" => f_particles,
         "particles" => f_particles)[var](fields)
    
    if rlim != nothing && zlim!= nothing
        setlims(rlim / co.milli, zlim / co.milli)
    end
    xylabel()

    if !isnothing(savedir)
        if isnothing(savename)
            savename = "$(var).png"
        end
        
        fname = joinpath(savedir, savename)
        @info "Saving plot to" fname
        plt.savefig(fname, dpi=600)
    end
end

function ndbin(a::AbstractArray{T, D}, s) where {T, D}
    ranges = ntuple(d -> first(axes(a, d)):s:last(axes(a, d)) - s, Val(D))
    @show ranges
    
    subrange = ntuple(d -> 0:(s - 1), Val(D))

    glb = CartesianIndices(ranges)
    loc = CartesianIndices(subrange)

    a1 = zeros(eltype(a), axes(glb))
    for I1 in CartesianIndices(axes(glb))
        I = glb[I1]
        for J in loc
            a1[I1] += a[I + J]
        end
        a1[I1] /= length(loc)
    end

    return a1
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

function plot1(path::AbstractString, step::Int, var::String;
               root=expanduser("~/data/denoise/final/"), save=false, titleprefix="",
               subtract=nothing, kw...)
    fstep = format("{:04d}.jld", step)
    if path[1] != "/"
        path = joinpath(root, path, fstep)
    end

    titleprefix = "[$(splitpath(path)[end - 1]): $fstep] $titleprefix"
    
    fields = load(joinpath(root, path), "fields");
    if !isnothing(subtract)
        path1 = joinpath(root, subtract, fstep)
        fields1 = load(joinpath(root, path1), "fields");
        subtract!(fields, fields1)
        titleprefix = "diff: $titleprefix"
    end
    
    savedir = save ? splitext(path)[1] : nothing

    plot1(fields, var; savedir, titleprefix, kw...)
end

function setlims(rlim, zlim)
    setrlim(rlim)
    setzlim(zlim)
end

function subtract!(f1, f2)
    for name in fieldnames(SavedGridFields)
        name == :grid && continue
        v1 = getfield(f1, name)
        v2 = getfield(f2, name)
        v1 .-= v2
    end
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


# Line plots
function plotaxis(id, istep, var; prefactor=1, root=expanduser("~/data/denoise/init/"), kwargs...)
    step = fmt("04d", istep)
    f = load(joinpath(root, "$(id)/$(step).jld"), "fields")
    v = getfield(f, var)
    plt.plot(prefactor .* dropdims(sum(v[1:1, :], dims=1), dims=1); kwargs...)    
end

function plotint(id, istep, var; root=expanduser("~/data/denoise/init/"), bndhack=true, kwargs...)
    step = fmt("04d", istep)
    f = load(joinpath(root, "$(id)/$(step).jld"), "fields")
    v = getfield(f, var)
    (;L, N, M, R, rc, zc) = f.grid
    dr = R / M
    dz = L / N
    
    if bndhack
        v[end - 1, :] .= 0
    end

    cs = dropdims(sum(@.(v[begin+1:end-1, begin+1:end-1] * rc * dr), dims=1), dims=1)
    total = sum(cs) * dz

    
    @show total
    
    @views plt.plot(zc, cs; kwargs...)    
end
