# Managing of a populations of homogeneous particles.

# Population of a given particle type
mutable struct Population{T, PS, C}
    "Number of particles"
    n::Atomic{Int}
    
    "Vector with the particles"
    particles::Vector{SuperParticleState{T, PS}}

    "Collision table"
    collisions::C
end

particle_type(popl::Population{T, PS, C}) where {T, PS, C} = particle_type(PS)
nparticles(popl::Population) = popl.n[]
maxrate(popl::Population) = maxrate(popl.collisions)

"""
Add a particle to the population `popl` with super-state super_state.
"""
function add_particle!(popl::Population, super_state)
    (;n, particles, collisions) = popl
    @assert n[] < length(particles) "Maximum number of particles reached"
    nprev = atomic_add!(n, 1)
    
    s = nextcol(maxrate(collisions))
    particles[nprev + 1] = super_state

    return nprev + 1
end


"""
Compute the total weight of a population `popl`.
"""
function weight(popl::Population{T}) where T
    (;n, particles) = popl
    tot = zero(T)
    for i in 1:n[]
        if particles[i].active
            tot += particles[i].w
        end
    end
    tot
end


"""
Compute the number of active particles in a population `pop`.
"""
actives(popl) = count(i->popl.particles[i].active, 1:popl.n[])


"""
    meanenergy(pop)

Compute the mean energy of a population `popl`.
"""
function meanenergy(popl::Population{T}) where T
    (;n, particles) = popl

    tot = zero(T)
    totw = zero(T)
    
    nparts = 0
    for i in 1:n[]
        p = particles[i]
        if p.active
            totw += p.w
            tot += p.w * energy(p.state)
        end
    end
    tot / totw
end


"""
Reorders the particles in the population `popl` to have all active particle 
at the initial positions in the list.

"""
function repack!(popl::Population)
    (;n, particles) = popl

    start = time()
    
    # new positions
    k = zeros(Int64, length(p))
    c = 0
    for i in 1:n[]
        if particles[i].active
            c += 1
            k[c] = i
        end
    end

    for i in 1:c
        particles[i] = particles[k[i]]
    end

    #@info "\u1b[0KParticles repackaged (took $(1000 * (time() - start)) ms)"
    n[] = c
end
