# Monte Carlo Streamer Code

This code is very preliminar and experimental: use with care.

## Install
* Julia version >1.6 is required.
* Clone the git repo and move to the newly created folder `mcstreamer`
* Start julia and type `]` to activate the package mode.
* Type `activate .` to activate the project in the current folder. 
* To install all dependencies run `instantiate` still in package mode (this may take a while).
* Exit package mode with a backspace.

## Run a simulation
* As above, activate pkg mode with `]` and `activate .`; exit pkg mode with backspace.
* Type `using Revise` (this allows julia to reload files as you modify them).
* Load the streamer code with 
```julia
includet("streamer.jl")
```
* If no error was issued run the code with
```julia
Streamer.main()
```
* Currently all configuration is defined in the `main` function in `streamer.jl`, which is a bit messy.


