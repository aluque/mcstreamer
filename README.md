# Monte Carlo Streamer Code
This repository contains the Monte Carlo code used in the manuscript on noise reduction by
M. Bayo, A. Malagón-Romero and A. Luque. For help on how to install and run the code contact
any of the authors.

## Code organization
The code contains several julia packages:

- `JuMC` is a generic particle Monte Carlo code that tracks particles, including ballistic advance and Monte Carlo collisions.
- `MCStreamer` integrates `JuMC` with a Poisson solver in order to simulate streamers. It also includes calls to a denoiser scheme based on tensorflow.
- `Multigrid` is a geometric multigrid solver for the Poisson equation in an uniform mesh. It works both for CPU and GPUs.
- `Swarm` uses `JuMC` to compute reaction rates and transport coefficients from cross-sections.
- `Constants` contains useful physical constants.

## Running streamer simulations
The streamer code `MCStreamer` reads input parameters from a `.toml` file. An example is provided in the
file `MCStreamer/samples/input.toml`. An example to start a simulation with 32 parallel threads is:

```
julia -t 32 --project=/path/to/jumc/MCStreamer/ /path/to/jumc/MCStreamer/scripts/run.jl input.toml
```
