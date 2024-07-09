# Constants

This is a simple julia package to use physical constants.  Use it as e.g.

```julia
julia> using Constants: co
julia> 1 / sqrt(co.epsilon_0 * co.mu_0) == co.c   # => true
```

