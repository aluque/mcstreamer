# We encode an association between particle types and their populations and
# collision tables in a NamedTuple (particle index).  To this end we use
# NamedTuples pointing to struct of type
# CollisionPopulation which contain a Population and a collision table.

struct CollisionPopulation{C <: CollisionTable, P <: Population}
    colls::C
    pop::P
end

function init!(pind, particle::Particle{sym}, Δt) where sym
    @unpack pop, colls = pind[sym]
    
    @threads for i in eachindex(pop)
        @inbounds pop.active[i] = true
        @inbounds pop.s[i] = round(Int, nextcol(colls.maxrate) / Δt)
    end
end

function advance!(pind, particle::Particle{sym}, efield, Δt) where sym
    @unpack pop = pind[sym]
    n = pop.n[]
    @batch for i in 1:n
        pop.x[i], pop.v[i] = advance_free(pop.particle, pop.x[i],
                                          pop.v[i], efield, Δt)
    end
end
