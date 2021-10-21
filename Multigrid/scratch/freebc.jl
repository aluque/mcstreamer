using Multigrid

using OffsetArrays
using BenchmarkTools
using LinearAlgebra
using PyCall
using PyPlot
using FFTW
using SpecialFunctions

const plt = PyPlot
plt.ioff()

function solvef(xf)
    m, n = xf * 2^7, 2^10
    g = 1
    
    b = zeros(m + 2g, n + 2g)
    L = 80.0
    R = xf * 10.0
    h = L / n
    
    r = h .* ((1:m) .- 0.5)
    z = h .* ((1:n) .- 0.5)
    w = 2
    
    @. b[(begin + g):(end - g), (begin + g):(end - g)] = exp((-r^2 - (z' - L / 2)^2) / 2w^2)
    
    u = similar(b)
    r = similar(b)
    r .= 0.0
    
    bc = ((LeftBnd(), Dirichlet()),
          (RightBnd(), Dirichlet()),
          (TopBnd(), Dirichlet()),
          (BottomBnd(), Neumann()))

    getconf() = MGConfig(bc=bc, s=h^2,
                         conn=Multigrid.CylindricalConnector{1}(),
                         levels=9,
                         tolerance=1e-10,
                         smooth1=30,
                         smooth2=30,
                         verbosity=2,
                         g=1)
    conf = getconf()

    ws = Multigrid.allocate(conf, parent(u));

    u .= 0

    Multigrid.solve(conf, u, b, ws);

    grad = zeros(n)
    grad .= (u[(end - g + 1), (begin + g):(end - g)] .- u[end - g, (begin + g):(end - g)]) ./ h
    k = (π / L) .* (1:n)
    x = k .* R
    f = @. 0.5 * L * k * (besselix(1, x) / besselix(0, x) +
                          besselkx(1, x) / besselkx(0, x))
    
    bm = -(h * 0.5) .* FFTW.r2r(grad, FFTW.RODFT10)
    am = bm ./ f
    am1 = zeros(size(am))
    am1[end] = am[end]
    
    bcond = zeros(n + 2g)
    bcond[(begin + g):(end - g)] = 0.5 .* FFTW.r2r(am, FFTW.RODFT01) .+ FFTW.r2r(am1, FFTW.RODFT01)
    
    Multigrid.setinhom!(conf, b, TopBnd(), bcond)
    Multigrid.solve(conf, u, b, ws);

    
    return u
end

function main()
    #plt.plot(u[(begin + g):(end - g), 10])
    g = 1
    for xf in [2, 4, 6, 8]
        u = solvef(xf)
        
        plt.plot(u[(begin + g):(end - g), 1 + 2^9])
    end
        #m = plt.pcolormesh(u[(begin + g):(end - g), (begin + g):(end - g)])
    #plt.colorbar(m)
    plt.show()
end

main()
