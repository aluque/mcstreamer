function logfmt(level, _module, group, id, file, line)
    return (:blue, format("{:<23}:", string(Dates.now())), "")
end

function start()
    logger = ConsoleLogger(meta_formatter=logfmt)
    with_logger(logger) do
        main()
    end
end

"""
    Read parameters from a file in `finput` and start the simulation.    
"""
function main(finput=ARGS[1]; debug=false, tmax=nothing, run=true)
    poisson_save_n[] = 0
    
    input = TOML.parsefile(finput)
    io = IOBuffer()
    MCStreamer.pretty_print(IOContext(io, :color => true), input)
    
    @info "Input read from $finput" * "\n" * String(take!(io))
    
    L::Float64 = input["domain"]["L"]
    R::Float64 = input["domain"]["R"]
    
    M::Int = input["domain"]["M"]
    N::Int = input["domain"]["N"]
    
    maxc::Int = input["maxc"]
    maxp::Int = input["maxp"]

    n::Int = input["n"]
    
    grid = Grid(R, L, M, N)
    fields = GridFields(grid)

    densities = Dict("N2" => co.nair * 0.79,
                     "O2" => co.nair * 0.21)
    
    # Create the energy grid in eV; all cross-sections are interpolated into
    # this grid
    #energy = (0:0.01:1000) .* co.eV
    energy = LinRange(0, 1000 * co.eV, 100_000)

    #datapath = normpath(joinpath(@__DIR__, "../data"))
    #csfile = joinpath(datapath, "LXCat-June2013-photo.json")
    csfiles = map(fname -> joinpath(dirname(finput), fname),
                  input["cross_sections"])

    proc, rate, maxrate = load_lxcat(csfiles, densities, energy)
    ecolls = CollisionTable(proc, energy, rate, maxrate)
        
    Δt = input["dt"]
    
    @info "max collision rate" maxrate
    @info "Time step" Δt

    # For debug: limit tmax by hand
    tmax = isnothing(tmax) ? input["tmax"] : tmax

    eb::Float64 = getfield(input)
    
    T = Float64
    
    # Initialize the electron population
    w::Float64 = input["seed"]["w"]
    z0::Float64 = input["seed"]["z0"]
    z1::Float64 = get(input["seed"], "z1", z0)
    
    v0 = @SVector zeros(T, 3)
    weight::Float64 = get(input, "initial_weight", 1.0)
    if !("init_files" in keys(input))
        init_particles = map(1:n) do _
            x = sampseg(z0, z1, w)
            ElectronState(x, v0, weight)
        end

    else
        nefile = joinpath(dirname(finput), input["init_files"]["ne_file"])
        qfile = joinpath(dirname(finput), input["init_files"]["q_file"])
        methoddesc = get(input["init_files"], "method", "noiseless")
        m = Dict("poisson" => PoissonInitSampling(maxc),
                 "noiseless" => NoiselessInitSampling(maxc))[methoddesc]
        
        init_particles = initfromfiles!(m, fields, nefile, qfile)
    end
    population_index = Pair{Symbol, Any}[:electron => Population(maxp, init_particles, ecolls)]
        
    # See if there is photo-emission; only in that case we include
    # photo-ionization
    ipe = findfirst(p -> p isa JuMC.PhotoEmission, ecolls.proc)
    if !isnothing(ipe)
        (;log_νmax) = ecolls.proc[ipe]
        pcolls = ZhelezniakCollisions(exp(log_νmax))
        init_photons = PhotonState{Float64}[]
        photon_population = Population(maxp, init_photons, pcolls)
        push!(population_index, :photon => photon_population)
    end
    
    mpopl = MultiPopulation(population_index...)
                            
    if !("init_files" in keys(input))
        density!(grid, fields.qfixed, mpopl, Electron, 1.0)
    end
    
    # Poisson
    bc = ((LeftBnd(), Dirichlet()),
          (RightBnd(), Dirichlet()),
          (TopBnd(), Neumann()),
          (BottomBnd(), Neumann()))

    poisson_levels = get(input, "poisson_levels", 9)
    
    mg = MGConfig(bc=bc, s=dz(grid)^2 / co.ϵ0,
                  conn=Multigrid.CylindricalConnector{1}(),
                  levels=poisson_levels,
                  tolerance=1.0e8 * co.elementary_charge,
                  smooth1=2,
                  smooth2=2,
                  verbosity=0,
                  g=1)
    
    ws = Multigrid.allocate(mg, parent(fields.q));
    
    efield = FieldInterp(fields)
    Δt_poisson, Δt_output, Δt_resample = map(v -> input["interval"][v],
                                             ("poisson", "output", "resample"))
    
    outpath = splitext(abspath(finput))[1]
    isdir(outpath) || mkdir(outpath)


    if haskey(input, "denoise")
        modelfname = joinpath(dirname(finput), input["denoise"]["model"])
        activ_time = get(input["denoise"], "activ_time", 0.0)
        denoiser = Denoiser(modelfname, (-1.0, 1.0),
                            Tuple(input["denoise"]["q_range"]), activ_time)
    else
        denoiser = NullDenoiser()
    end

    # Fluid
    if haskey(input, "needle")
        needle = input["needle"]
        setneedle!(fields, needle["sigma_z"], needle["sigma_r"],
                   needle["position_z"], needle["position_r"], needle["ne0"])
        hasfluid = true
    else
        hasfluid = false
    end
    
    
    if debug
        return (;mpopl, efield, eb, Δt, fields, mg, ws, grid, input)
    end

    if run
        nsteps(mpopl, Int(fld(tmax, Δt)), maxc, efield, eb, Δt,
               Δt_poisson, Δt_output, Δt_resample,
               fields, mg, ws, denoiser, hasfluid; outpath)
    end
    
    # k = collrates(pind, Electron())
    density!(grid, fields.qpart, mpopl, Electron, -1.0)

    (;grid, mpopl, efield, Δt, fields)
end


"""
    Advance the simulation by `n` steps.

    Parameters:
    * `mpopl` a (multi-)particle population.
    * `n`: number of steps.
    * `maxc`: Max. particles per cell.
    * `efield`: Electric field interpolator.    
    * `eb`: Background electric field.
    * `Δt`: Time-step at which the particles are updated.
    * `Δt_poisson`, `Δt_output`, `Δt_resample`: Time steps for updatting
      the electrostatic field, outputting data and resampling particles.
    * `fields`: A GridFields with space for all fields.
    * `mg`: A Multigrid (MG) solver.
    * `ws`: Working space for the MG solver.
    * `denoiser`: a Denoiser
"""
function nsteps(mpopl, n, maxc, efield, eb, Δt, Δt_poisson, Δt_output, Δt_resample,
                fields, mg, ws, denoiser, hasfluid; outpath="")

    popl = get(mpopl, Electron)
    ecolls = popl.collisions
    tracker = CollisionTracker(fields)

    poisson = TimeStepper(Δt_poisson)
    output = TimeStepper(Δt_output)
    resample = TimeStepper(Δt, Δt_resample)

    # Measure time used in different work
    elapsed_poisson = 0.0
    elapsed_advance = 0.0
    elapsed_resample = 0.0
        
    for i in 1:n
        t = (i - 1) * Δt
        
        atstep(poisson, t) do _
            elapsed_poisson += @elapsed poisson!(fields, mpopl, eb, mg, ws,
                                                 denoiser, t, outpath)
        end
        
        atstep(output, t) do j
            # Use true here to enable compression. Add electron=popl to save all
            # electron states.

            jldsave(joinpath(outpath, fmt("04d", j) * ".jld"), false;
                    iotype=IOStream, fields);
            
            active_superparticles = actives(popl)
            physical_particles = weight(popl)
            @info "$(j * Δt_output * 1e9) ns"  active_superparticles physical_particles
            @info "Elapsed times" elapsed_poisson elapsed_advance elapsed_resample
        end

        elapsed_advance += @elapsed advance!(mpopl, efield, Δt, tracker)

        if iszero(i % 10000)
            mean_energy = JuMC.meanenergy(popl)
            max_energy = JuMC.maxenergy(popl)
            active_superparticles = actives(popl)
            physical_particles = weight(popl)
            @info("t = $t", active_superparticles, physical_particles,
                  mean_energy / co.eV,
                  max_energy / co.eV)
        end
        
        atstep(resample, t) do _
            elapsed_resample += @elapsed begin
                pre_total_weight = weight(popl)
                repack!(popl)
                shuffle!(popl)
                resample!(popl, fields, maxc)
                post_total_weight = weight(popl)
                @assert(isapprox(pre_total_weight, post_total_weight, rtol=1e-5),
                        "repackaging is not conserving the number of particles")
                
            end
        end

        if hasfluid
            setdne!(fields)
            euler!(fields, Δt)
        end
        
        repack!(popl)
    end
end


"""
    Sample from a spatial distribution consisting in a vertical segment with a gaussian blur.
"""
function sampseg(z0, z1, w)
    @assert z1 >= z0
    l = z1 - z0

    # prob. of being closest to a point in the segment
    p = l / (l + sqrt(2π) * w)
    
    if (rand() < p)
        # closest to a point in the segment
        u = rand()
        return @SVector [w * randn(), w * randn(), z0 + u * l]
    else
        s = @SVector [w * randn(), w * randn(), w * randn()]
        if s[3] > 0
            return s + @SVector [0, 0, z1]
        else
            return s + @SVector [0, 0, z0]
        end
    end    
end


"""
    Read the electric field from input, allowing for units to be specified.
"""
function getfield(input)
    units = get(input, "field_units", "Td")
    scale = Dict("Td" => co.Td * co.nair,
                 "kV/cm" => 1e5,
                 "MV/m" => 1e6,
                 "V/m" => 1.0,
                 "" => 1.0)[units]
    
    -input["eb"] * scale
end

function pretty_print(io::IO, d::Dict, level=0)
    for (k,v) in d
        if typeof(v) <: Dict
            print(io, join(fill(" ", level * 8)))
            printstyled(io, k, color=:light_yellow, bold=true)            
            printstyled(io, " => \n", color=:light_black)            
            pretty_print(io, v, level + 1)
        else
            print(io, join(fill(" ", level * 8)))
            printstyled(io, k, color=:light_yellow, bold=true)
            printstyled(io, " => ", color=:light_black)
            printstyled(io, repr(v) * "\n", color=:blue) 
        end
    end
    nothing
end

pretty_print(d::Dict, kw...) = pretty_print(stdout, d, kw...)
