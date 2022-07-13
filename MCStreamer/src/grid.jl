"""
    A structure defining a grid.
"""
struct Grid{T}
    "Length in the r axis"
    R::T

    "Length in the z axis"
    L::T
    
    "Cells in r axis"
    M::Int

    "Cells in z axis"
    N::Int

    "r coordinates of cell centers"
    rc::LinRange{T, Int}

    "r coordinates of cell faces"
    rf::LinRange{T, Int}

    "z coordinates of cell centers"
    zc::LinRange{T, Int}

    "z coordinates of cell faces"
    zf::LinRange{T, Int}

    """
    Create a grid extending up to `R` in the r-coordinate and `L` in the
    z-coordinate with `M` cells in r and `N` cells in z.
    """
    function Grid(R::T, L::T, M, N) where T
        rf = LinRange(0, R, M + 1)
        zf = LinRange(0, L, N + 1)
        
        rc = 0.5 * (rf[(begin + 1):end] + rf[begin:(end - 1)])
        zc = 0.5 * (zf[(begin + 1):end] + zf[begin:(end - 1)])

        new{T}(R, L, M, N, rc, rf, zc, zf)
    end
end

"Check if point x falls inside the grid"
@inline function inside(grid::Grid, x)
    @unpack rf, zf = grid
    r = rcyl(x)
    (rf[1] < r < rf[end]) && (zf[1] < x[3] < zf[end])
end

"""
    Allocate an array for a field evaluated at cell centers of a `grid`, 
    including space for `g` ghost cells.
"""
function calloc_centers(T, grid::Grid, g::Int=1)
    arr = zeros(T, (1 - g):(grid.M + g), (1 - g):(grid.N + g))
end

calloc_centers(grid::Grid{T}, g::Int=1) where T = calloc_centers(T, grid, g)

function calloc_centers_threads(T, grid::Grid, g::Int=1)
    arr = zeros(T, (1 - g):(grid.M + g), (1 - g):(grid.N + g),
                Threads.nthreads())
end

calloc_centers_threads(grid::Grid{T}, g::Int=1) where T =
    calloc_centers_threads(T, grid, g)

# Face-centered fields always have size (M + 1) x (N + 1), even if sometimes
# one row/column is unused.  They also do not have ghost cells.
calloc_faces(T, grid::Grid, g::Int=1) = zeros(T, (1 - g):(grid.M + g + 1),
                                              (1 - g):(grid.N + g + 1))

dr(grid::Grid) = step(grid.rc)
dz(grid::Grid) = step(grid.zc)
dV(grid::Grid, i) = 2π * grid.rc[i] * dr(grid) * dz(grid)
rcyl(x) = sqrt(x[1]^2 + x[2]^2)


"""
    Index (CartesianIndex) of location `r` inside `grid`.
"""
function cellindex(grid, r)
    ρ = sqrt(r[1]^2 + r[2]^2)

    i = Int(fld(ρ - first(grid.rf), step(grid.rf))) + 1
    j = Int(fld(r[3] - first(grid.zf), step(grid.zf))) + 1
    CartesianIndex(i, j)
end


"""
    Same as `cellindext(grid, r)` but adds an index with the current thread id
"""
function cellindext(grid, r)
    I = cellindex(grid, r)
    
    CartesianIndex(I[1], I[2], Threads.threadid())
end
