function logfmt(level, _module, group, id, file, line)
    return (:blue, format("{:<23}:", string(Dates.now())), "")
end

function start()
    logger = ConsoleLogger(meta_formatter=logfmt)
    with_logger(logger) do
        main()
    end
end

function main(finput=ARGS[1]; debug=false, tmax=nothing, run=true)
    poisson_save_n[] = 0
    
    input = TOML.parsefile(finput)
    @info "Input read from $finput" input
    
    L::Float64 = input["domain"]["L"]
    R::Float64 = input["domain"]["R"]
    
    n::Int = input["n"]
    
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
    
    
    Δt = 10^floor(log10(1 / (2 * ecolls.maxrate))) / 4
    @info "max collision rate" maxrate
    @info "Time step derived from collision rate" Δt

    # For debug: limit tmax by hand
    tmax = isnothing(tmax) ? input["tmax"] : tmax

    eb::Float64 = -input["eb"] * co.Td * co.nair
    
    T = Float64
    maxp::Int = input["maxp"]
    
    # Initialize the electron population
    w::Float64 = input["seed"]["w"]
    z0::Float64 = input["seed"]["z0"]
    v0 = @SVector zeros(T, 3)
    #v0 = @SVector [0, 0, sqrt(2 * 100 * co.eV / co.electron_mass)]
    
    init_particles = map(1:n) do _
        x = @SVector [w * randn(), w * randn(), z0 + w * randn()]
        ElectronState(x, v0)
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
                            
    M::Int = input["domain"]["M"]
    N::Int = input["domain"]["N"]
    
    grid = Grid(R, L, M, N)
    fields = GridFields(grid)
    density!(grid, fields.qfixed, mpopl, Electron, 1.0)
    
    
    # Poisson
    bc = ((LeftBnd(), Dirichlet()),
          (RightBnd(), Dirichlet()),
          (TopBnd(), Neumann()),
          (BottomBnd(), Neumann()))
    
    mg = MGConfig(bc=bc, s=co.elementary_charge * dz(grid)^2 / co.ϵ0,
                  conn=Multigrid.CylindricalConnector{1}(),
                  levels=9,
                  tolerance=1.0e8,
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

    maxc::Int = input["maxc"]

    if haskey(input, "denoise")
        modelfname = joinpath(dirname(finput), input["denoise"]["model"])
        
        denoiser = Denoiser(modelfname, (0.0, 1.0),
                            Tuple(input["denoise"]["log10range"]))
    else
        denoiser = NullDenoiser()
    end

    
    if debug
        return (;mpopl, efield, eb, Δt, fields, mg, ws)
    end

    # Debuggin help
    if run
        nsteps(mpopl, Int(fld(tmax, Δt)), maxc, efield, eb, Δt,
               Δt_poisson, Δt_output, Δt_resample,
               fields, mg, ws, denoiser; outpath)
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
                fields, mg, ws, denoiser; outpath="")

    popl = get(mpopl, Electron)
    ecolls = popl.collisions
    tracker = CollisionTracker(fields)

    poisson = TimeStepper(Δt_poisson)
    output = TimeStepper(Δt_output)
    resample = TimeStepper(Δt, Δt_resample)

    # Measure time used in different work
    elapsed_poisson = 0.0
    elapsed_advance = 0.0
    elapsed_collisions = 0.0
    elapsed_resample = 0.0
        
    for i in 1:n
        t = (i - 1) * Δt
        
        atstep(poisson, t) do _
            elapsed_poisson += @elapsed poisson!(fields, mpopl, eb, mg, ws,
                                                 denoiser, outpath)
        end
        
        atstep(output, t) do j
            # Use true here to enable compression.
            jldsave(joinpath(outpath, fmt("04d", j) * ".jld"), false; fields,
                    electron=popl)
            
            active_superparticles = actives(popl)
            physical_particles = weight(popl)
            @info "$(j * Δt_output * 1e9) ns"  active_superparticles physical_particles
            @info "Elapsed times" elapsed_poisson elapsed_advance elapsed_collisions elapsed_resample
        end

        elapsed_advance += @elapsed advance!(mpopl, efield, Δt)
        elapsed_collisions += @elapsed collisions!(mpopl, Δt, tracker)

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
        
        repack!(popl)
    end
end
