##
##  DATA INPUT AND PRE-PROCESSING
##
"""
Reads a collision database from a .json file and builds a collision
table with cumulative cross sections scaled by the densities contained
in the densities dict and evaluated at the grid eng. 
"""
function load_lxcat(fnames, densities, energy)
    T = eltype(energy)
    
    db = mapreduce(vcat, fnames) do fname
        local db1    
        open(fname, "r") do fd
            db1 = JSON.parse(read(fd, String))
        end
        return db1
    end
    
    # We first build a dictionary with the targets to later correct for the
    # EFFECTIVE / ELASTIC problem
    targets = Dict()

    v = @. sqrt(2 * energy / mass(Electron()))

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

        if haskey(item, "rescale")
            cs0 .*= item["rescale"]
        end

        if haskey(item, "weight_scale")
            cs0 .*= item["weight_scale"]
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
                  (itm, dens) -> Attachment(itm["threshold"] * co.eV),

                  "EXCITATION" =>
                  (itm, dens) -> Excitation(itm["threshold"] * co.eV),

                  "IONIZATION" =>
                  (itm, dens) -> Ionization(itm["threshold"] * co.eV),

                  "ELASTIC" =>
                  (itm, dens) -> Elastic(itm["mass_ratio"]),

                  "PHOTOEMISSION" => (itm, dens) -> begin
                  # The absorption is rescaled according to the density in the
                  # "absorber" field and transformed from 1/m to 1/s.
                  νmin = co.c * dens[itm["absorber"]] * itm["chi_min"]
                  νmax = co.c * dens[itm["absorber"]] * itm["chi_max"]
                  
                  PhotoEmission(νmin, νmax, get(itm, "weight_scale", 1))
                  end

                  )


    νrun = zeros(T, length(energy))
    rate = zeros(T, (nprocs + 1, length(energy)))
    i = 1

    for (name, ps) in targets
        # Ensure that we have elastic collisions and not effective
        ensure_elastic(ps)
        
        for item in ps
            push!(proc, dkinds[item["kind"]](item, densities))
            rate[i, :] .= item["nu"]
            @info "New process: " * signature(item)
            i += 1
        end
    end

    νtotal = dropdims(sum(rate, dims=1), dims=1)
    
    # Find also the max. of the collision rate
    maxrate = maximum(νtotal)

    nullrate = maxrate .- νtotal
    rate[nprocs + 1, :] = nullrate
    push!(proc, NullCollision())
    
    # Order collisions according to the integral (sum) over energies
    inteng = dropdims(sum(rate, dims=2), dims=2)
    perm = sortperm(inteng, rev=true)
    origperm = invperm(perm)
    
    proc = tuple(proc[perm]...)
    rate = rate[perm, :]
    
    @assert size(rate, 1) == length(proc)
    
    (;proc, rate, maxrate, origperm)
end

"""
    Prepare a photo-emission process.
"""
function prepare_photo_emission(itm, dens)
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

    
function signature(proc)
    target = proc["target"]
    product = haskey(proc, "product") ? proc["product"] : ""
    kind = proc["kind"]
    "e + $target -> $product... ($kind)"
end
