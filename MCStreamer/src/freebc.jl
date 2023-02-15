""" Solve Poisson's equation imposing free b.c. at r=R with the method 
    described by Malagón-Romero & Luque (2018).
"""
function solve_freebc!(u, b, grid, mg, ws, u1=u)
    # We allow an extra u1 array in case we want to keep separate solutions with different b.c.
    # to start with better guesses.
    (;M, N, L, R) = grid
    G = 1
    
    Multigrid.solve(mg, u1, b, ws);
    
    grad = zeros(eltype(u), N)
    grad .= (u1[end - G + 1, (begin + G):(end - G)] .-
             u1[end - G,     (begin + G):(end - G)]) ./ dr(grid)

    k = (π / L) .* (1:N)
    x = k .* R
    f = @. L * k * (besselix(1, x) / besselix(0, x) +
                    besselkx(1, x) / besselkx(0, x)) / 2
    
    bm = -(dr(grid) / 2) .* FFTW.r2r(grad, FFTW.RODFT10)
    am = bm ./ f
    am1 = zeros(eltype(u), size(am))
    am1[end] = am[end]

    bcond = zeros(eltype(u), N + 2G)
    bcond[(begin + G):(end - G)] .= (FFTW.r2r(am, FFTW.RODFT01) / 2 .+
                                     FFTW.r2r(am1, FFTW.RODFT01))

    Multigrid.setinhom!(mg, b, TopBnd(), bcond)
    Multigrid.solve(mg, u, b, ws);

    # The solve method returns u with homogeneous b.c.  We should impose here
    # the correct ones or we get high electric fields at the boundary.
    u[end - G + 1, :] .+= 2 .* dbcond
end
