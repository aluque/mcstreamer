# Managing of a populations of homogeneous particles.

# Population of a given particle type
mutable struct Population{P, T}
    particle::P

    n::Atomic{Int}
    
    active::Vector{Bool}
    v::Vector{SVector{3, T}}
    x::Vector{SVector{3, T}}    
    w::Vector{T}
    
    # Time left for next collision
    s::Vector{Int}
end

function Population(particle, n::Int, active, v, x, w, s)
    Population(particle, Atomic{Int}(n), active, v, x, w, s)
end

function Population(particle, n::Int, active, v, x, s)
    w = fill(1.0, size(x))
    Population(particle, n, active, v, x, w, s)
end


Base.eachindex(p::Population) = 1:p.n[]
Base.length(p::Population) = p.n[]
setlength!(p::Population, n) = p.n[] = n

# Warning: non-thread-safe
"""
    add_particle!(pop, p, x)

Add a particle to the population `pop` with momentum `p` and position `x`.
"""
function add_particle!(pop::Population, v, x, w)
    @assert pop.n[] < length(pop.active) "Maximum number of particles reached"
    nprev = atomic_add!(pop.n, 1)
    pop.v[nprev + 1] = v
    pop.x[nprev + 1] = x
    pop.w[nprev + 1] = w

    pop.active[nprev + 1] = true

    nprev + 1
end


"""
    meanenergy(pop)

Compute the total weight of a population `pop`.
"""
function weight(pop::Population{P, T}) where {P, T}
    tot = zero(T)
    for i in eachindex(pop)
        if pop.active[i]
            tot += pop.w[i]
        end
    end
    tot
end

"""
    actives(pop)

Compute the number of active particles in a population `pop`.
"""
actives(pop) = count(i->pop.active[i], eachindex(pop))


"""
    meanenergy(pop)

Compute the mean energy of a population `pop`.
"""
function meanenergy(pop::Population{P, T}) where {P, T}
    tot = zero(T)
    nparts = 0
    for i in eachindex(pop)
        if pop.active[i]
            tot += pop.w[i] * energy(pop.particle, pop.v[i])
            nparts += 1
        end
    end
    tot / nparts
end


"""
    repack!(p)

Reorders the particles in the population `p` to have all active particle 
at the initial positions in the list.

"""
function repack!(p::Population)
    start = time()
    
    # new positions
    k = zeros(Int64, length(p))
    c = 0
    for i in eachindex(p)
        if p.active[i]
            c += 1
            k[c] = i
        end
    end

    for i in 1:c
        p.v[i] = p.v[k[i]]
        p.x[i] = p.x[k[i]]
        p.w[i] = p.w[k[i]]
        p.s[i] = p.s[k[i]]

        p.active[i] = true
    end

    #@info "\u1b[0KParticles repackaged (took $(1000 * (time() - start)) ms)"
    setlength!(p, c)
end
