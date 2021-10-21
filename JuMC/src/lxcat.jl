##
##  DATA INPUT AND PRE-PROCESSING
##
"""
Reads a collision database from a .json file and builds a collision
table with cumulative cross sections scaled by the densities contained
in the densities dict and evaluated at the grid eng. 
"""
function load_lxcat(fname, densities, energy)
    T = eltype(energy)

    local db    
    open(fname, "r") do fd
        db = JSON.parse(read(fd, String))
    end
    
    # We first build a dictionary with the targets to later correct for the
    # EFFECTIVE / ELASTIC problem
    targets = Dict()

    γ = @. 1 + energy / mc2
    β = @. sqrt(1 - 1 / (γ^2))
    v = @. co.c * β

    nprocs = 0
    for item in db
        dens::Float64 = get(densities, item["target"], 0.0)
        dens == 0.0 && continue
        target = get!(targets, item["target"], Vector())

        energy0 = [v[1] * co.eV for v in item["data"]]
        cs0 = [v[2] for v in item["data"]]

        if occursin("3-body", item["comment"])
            cs0 .*= densities[item["target"]] ./ co.centi^-3
        end
        
        # For performance first we interpolate to an uniform grid
        itp = extrapolate(interpolate((energy0,), cs0, Gridded(Linear())),
                          Flat())
        
        item["nu"] = @. dens * v * itp.(energy)

        push!(target, item)
        nprocs += 1
    end

    proc = Vector{CollisionProcess}()
    
    dkinds = Dict("ATTACHMENT" =>
                  (itm) -> Attachment(itm["threshold"] * co.eV),
                  "EXCITATION" =>
                  (itm) -> Excitation(itm["threshold"] * co.eV),
                  "IONIZATION" =>
                  (itm) -> Ionization(itm["threshold"] * co.eV),
                  "ELASTIC" =>
                  (itm) -> Elastic(itm["mass_ratio"]))


    νrun = zeros(T, length(energy))
    rate = zeros(T, (nprocs, length(energy)))
    i = 1

    for (name, ps) in targets
        # Ensure that we have elastic collisions and not effective
        ensure_elastic(ps)
        
        for item in ps
            push!(proc, dkinds[item["kind"]](item))
            rate[i, :] .= item["nu"]
            # This is to compute the max. coll rate
            νrun .+= item["nu"]          
            i += 1
        end
    end

    # Find also the max. of the collision rate
    maxrate = maximum(νrun)
    proc = tuple(proc...)
    
    (;proc, rate, maxrate)
end


""" 
Make sure that we have elastic collisions in the collision set for a target.
"""
function ensure_elastic(procs)
    for p in procs
        p["kind"] == "EFFECTIVE" || continue
        for p2 in procs
            if p != p2 && (p2["kind"] == "EXCITATION" || p2["kind"] == "IONIZATION")
                p["nu"] .-= p2["nu"]
            end
        end
        p["kind"] = "ELASTIC"
    end
end

    
