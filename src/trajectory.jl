# The whole-trajectory latent block and its iFFBS kernel.
#
# In every EpidemicTrajectories fit, the hidden trajectory `X` is stored as ONE
# whole-matrix latent (`X ~ TrajectoryLatent(...)`), routed by PracticalBayes to
# the value store so it is AD-constant during the NUTS gradients and resampled once
# per Gibbs sweep by the iFFBS kernel. Both pieces are byte-identical across models
# except for how the kernel gathers the parameters out of the Gibbs state — so both
# are provided here, the kernel taking that one varying piece as a closure.

"""
    TrajectoryLatent(n_time, n_ind)

A placeholder discrete matrix distribution for the whole hidden trajectory `X`.
Its `logpdf` is a constant `0.0` and its `rand` is an all-first-state matrix: the
trajectory's real conditional is supplied by the iFFBS likelihood term and its
real draw by [`iffbs_kernel`](@ref). Declaring it *Discrete* is what makes
PracticalBayes route `X` to the value store (owned by the latent kernel, never
touched by NUTS) rather than to the continuous sampler.
"""
struct TrajectoryLatent <: Distributions.DiscreteMatrixDistribution
    n_time::Int
    n_ind::Int
end
Base.size(d::TrajectoryLatent) = (d.n_time, d.n_ind)
Distributions.logpdf(::TrajectoryLatent, X::AbstractMatrix) = 0.0
Distributions.rand(rng::AbstractRNG, d::TrajectoryLatent) = fill(1, d.n_time, d.n_ind)

"""
    iffbs_kernel(latent!; params)

Build the PracticalBayes latent kernel that resamples the whole trajectory `X`
once per Gibbs sweep with EpidemicTrajectories' iFFBS sampler `latent!` (the
function returned by `epidemic_latent_sampler(data)`).

`params` is a closure mapping the Gibbs state to the parameter NamedTuple the rate
functions expect: `params(values) -> (; α=values.α, ...)`, where `values` is the
`ModelConditional`'s `.values` (every current parameter draw). This is the one
piece that varies between models — the reparameterisations and field names live
here, out of the otherwise-identical kernel body.

```julia
latent! = epidemic_latent_sampler(data)
kX = iffbs_kernel(latent!; params = v -> (; α=v.α, β=v.β, m=v.m̃ + 1.0,
                                            ν=v.ν, θʳ=v.θʳ, θᶠ=v.θᶠ))
# ... spl = Gibbs(:X => kX, ...)
```
"""
struct iFFBSKernel{F,P} <: AbstractLatentKernel
    latent!::F
    params::P
end
iffbs_kernel(latent!; params) = iFFBSKernel(latent!, params)

function PracticalBayes.latent_step(rng, k::iFFBSKernel, block_names, c::ModelConditional)
    block_names == (:X,) || error("iffbs_kernel only handles the :X block (got $(block_names))")
    X = copy(c.values.X)
    k.latent!(rng, k.params(c.values), X)
    (; X=X)
end
