"""
    A mixed population, composed of multiple particle types.
"""
struct MultiPopulation{TP <: NamedTuple}
    # index is a named tuple that connects particle types with their population
    index::TP

    function MultiPopulation(d...)
        index = NamedTuple(d)
        new{typeof(index)}(index)
    end
end


Base.get(mp::MultiPopulation, pt::Type{ParticleType{S}}) where S = getfield(mp.index, S)
Base.map(f, mp::MultiPopulation) = map(f, mp.index)
Base.pairs(mp::MultiPopulation) = pairs(mp.index)


"""
    Advance without collisions the particles in the population.
"""
function advance!(mpopl, efield, Δt)
    # WARN: Possibly type-unstable
    for (sym, popl) in pairs(mpopl)
        @batch for i in 1:popl.n[]
            l = LazyRow(popl.particles, i)
            l.active || continue
            
            state = instantiate(l)
            new_state = advance_free(state, efield, Δt)
            popl.particles[i] = new_state
        end
    end
end

"""
Check for the particles that are set to collide and then perform a
(possibly null) collision
"""
function collisions!(mpopl, Δt, tracker=VoidCollisionTracker())
    # WARN: Possibly type-unstable
    for (sym, popl) in pairs(mpopl)
        @batch for i in 1:popl.n[]
            l = LazyRow(popl.particles, i)
            l.active || continue

            l.s -= Δt * maxrate(popl.collisions)
            if l.s <= 0
                state = instantiate(l)
                do_one_collision!(mpopl, popl.collisions, state, i, tracker)
                l.s = nextcoll()
            end
        end
        #@info "$c/$m = $(c/m) collision fraction"
    end
end
