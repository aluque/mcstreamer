#=
Code to handle a population of electrons without any other particle.
=#

struct ElectronPopulation{T, C}
    "Number of particles"
    n::Atomic{Int}
    
    "Vector with the particles"
    particles::Vector{SuperParticleState{T, ElectronState{T}}}

    "Collision table"
    collisions::C
end

# Move to a more generic population?
get_particle_count(p::ElectronPopulation, ::Type{Electron}) = p.n[]
set_particle_count!(p::ElectronPopulation, ::Type{Electron}, n) = p.n[] = n
maxrate(p::ElectronPopulation) = maxrate(p.collisions)

function add_particle!(p::ElectronPopulation, ::Type{Electron}, state, w)
    @assert p.n[] < length(particles) "Maximum number of particles reached"
    nprev = atomic_add!(p.n, 1)
    
    s = nextcol(maxrate(p.collisions))
    p.particles[nprev + 1] = SuperParticleState(state, true, w, s)    

    return nprev + 1
end

function get_super_particle_state(p::ElectronPopulation, ::Type{Electron}, i)
    p.particles[i]
end

function get_particle_state(p::ElectronPopulation, ::Type{Electron}, i)
    get_super_particle_state(p, Electron, i).state
end

function set_super_particle_state!(p::ElectronPopulation, ::Type{Electron}, i,
                                   super_state)
    p.particles[i].state = state
end

function set_particle_state!(p::ElectronPopulation, ::Type{Electron}, i, state)
    p.particles[i].state = state
end

function remove_particle!(p::ElectronPopulation, ::Type{Electron}, i)
    p.particles[i].active = false
end

function get_collisions(p::ElectronPopulation, ::Type{Electron})
    p.collisions
end

function init!(p::ElectronPopulation, init_state::ElectronState, nparticles)
    for i in eachindex(p.particles)
        s = nextcol(maxrate(p))
        w = 1.0
        active = true
        p.particles[i] = SuperParticleState(init_state, active, w, s)
    end
end

function advance!(p::ElectronPopulation, efield, Δt)
    advance!(particles, efield, Δt)
end
