#=
  Plotting functions.
=#

function plot(fields)
    plt.matplotlib.pyplot.style.use("granada")
    M = fields.grid.M
    N = fields.grid.N
    # qfixed_t = dropdims(sum(@view(fields.qfixed[1:M, 1:N, :]), dims=3), dims=3)
    # qpart_t = dropdims(sum(@view(fields.qpart[1:M, 1:N, :]), dims=3), dims=3)

    # q = qfixed_t .- qpart_t
    @unpack rf, zf, rc, zc = fields.grid
    
    plt.figure("Charge density")
    plt.pcolormesh(zf, rf, @view(fields.q[1:M, 1:N]))
    cbar = plt.colorbar(label=L"Charge density (C/m$^3$)")
    
    plt.figure("Electric field")

    eabs = @. @views sqrt(0.25 * (fields.er[1:M, 1:N] + fields.er[2:(M + 1), 1:N])^2 +
                          0.25 * (fields.ez[1:M, 1:N] + fields.ez[1:M, 2:(N + 1)])^2)

    plt.pcolormesh(zf, rf, eabs)
    cbar = plt.colorbar(label="Electric field (V/m)")
    
    plt.show()
end

