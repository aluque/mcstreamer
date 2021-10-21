struct Particle{S}; end

const Electron = Particle{:electron}
const Positron = Particle{:positron}
const Photon = Particle{:photon}

@inline id(p::Particle{sym}) where sym = sym
@inline id(sym::Symbol) = sym

name(p::Particle) = String(id(p))



