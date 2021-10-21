using Multigrid

using OffsetArrays
using BenchmarkTools
using LinearAlgebra
using PyCall
using PyPlot
const plt = PyPlot
plt.ioff()

function main()
    n = 2^8
    b = zeros(0:n+1, 0:n+1)

    x = (1:n) .- 0.5
    w = n / 8
    
    @. b[1:n, 1:n] = exp((-(x)^2 - (x' - n / 2)^2) / 2w^2)
    
    u = similar(b)
    r = similar(b)
    r .= 0.0
    
    bc = ((LeftBnd(), Neumann()),
          (RightBnd(), Dirichlet()),
          (TopBnd(), Dirichlet()),
          (BottomBnd(), Neumann()))

    getconf() = MGConfig(bc=bc, s=1.0,
                         conn=Multigrid.CylindricalConnector{1}(),
                         levels=9,
                         tolerance=1e-10,
                         smooth1=30,
                         smooth2=30,
                         verbosity=2,
                         g=1)
    conf = getconf()

    ws = Multigrid.allocate(conf, parent(u));

    inhom = zeros(n + 2) .+ 15.0
    parent(u) .= 0
    parent(b) .= 0.0

    v, val = Multigrid.setinhom!(conf, parent(b), TopBnd(), inhom)
    Multigrid.solve(conf, parent(u), parent(b), ws);

    plt.plot(u[(begin + 1):(end - 1), 10])
    # m = plt.pcolormesh(u)
    # plt.colorbar(m)
    plt.show()
end


main()
