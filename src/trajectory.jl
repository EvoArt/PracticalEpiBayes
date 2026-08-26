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

`params` maps the Gibbs state to the parameter NamedTuple the rate functions
expect. Give it either form:

  * **a `Tuple` of names** — the common case, where the rate functions want some
    of the model's own parameters under their own names. The kernel selects them,
    so nothing is transcribed:

    ```julia
    kX = iffbs_kernel(latent!; params = (:α, :β, :m, :ν, :θʳ, :θᶠ))
    ```

    A name the Gibbs state does not carry is an error naming it, raised on the
    first sweep rather than surfacing as a `type has no field` from inside a rate
    function.

  * **a closure** `params(values) -> NamedTuple`, where `values` is the
    `ModelConditional`'s `.values` (every current parameter draw). Needed when a
    rate function wants something the model does not store under that name — a
    reparameterisation, or a rename:

    ```julia
    kX = iffbs_kernel(latent!; params = v -> (; α=v.α, β=v.β, m=v.m̃ + 1.0,
                                                ν=v.ν, θʳ=v.θʳ, θᶠ=v.θᶠ))
    ```

The `Tuple` form exists because the closure was, in the common case, pure
transcription of the model's own parameter list into a THIRD place — after the
`~` declarations and the `pars = (; ...)` tuple in the model body. Adding a
parameter then meant updating three, and forgetting this one showed up only at
run time.

```julia
latent! = epidemic_latent_sampler(data)
kX = iffbs_kernel(latent!; params = (:α, :β, :m, :ν, :θʳ, :θᶠ))
# ... spl = Gibbs(:X => kX, ...)
```
"""
struct iFFBSKernel{F,P} <: AbstractLatentKernel
    latent!::F
    params::P
end

iffbs_kernel(latent!; params) = iFFBSKernel(latent!, _params_selector(params))

# A closure is used as given; a collection of names becomes a selector.
_params_selector(f) = f
_params_selector(names::Tuple{Vararg{Symbol}}) = ParamSelector(names)
_params_selector(names::AbstractVector{Symbol}) = ParamSelector(Tuple(names))

"""
    ParamSelector(names)

Selects `names` out of the Gibbs state into a NamedTuple — what
[`iffbs_kernel`](@ref)'s `params = (:a, :b, ...)` form builds.

A callable struct rather than a closure so `names` lives in the TYPE and the
`NamedTuple` construction is fully inferred. This runs once per sweep, but its
result is read for every individual at every timepoint, so an abstractly-typed
parameter tuple would be expensive in exactly the way EpidemicTrajectories'
CLAUDE.md warns about.
"""
struct ParamSelector{N}
    ParamSelector(names::Tuple{Vararg{Symbol}}) = new{names}()
end

@inline function (::ParamSelector{N})(values) where {N}
    _check_param_names(values, Val(N))
    return NamedTuple{N}(map(n -> getproperty(values, n), N))
end

# The check folds away when the names are present (it is decided by types), so it
# costs nothing in the common case and gives a precise message in the other one.
@inline _check_param_names(values, ::Val{()}) = nothing
@inline function _check_param_names(values, ::Val{N}) where {N}
    n = first(N)
    hasproperty(values, n) || error(
        "iffbs_kernel: `params` names `:$n`, which is not a variable in this " *
        "model. Available: " * join(keys(values), ", ") * ". Pass a closure if " *
        "the rate functions need a value the model does not store under that " *
        "name (a reparameterisation or a rename).")
    return _check_param_names(values, Val(Base.tail(N)))
end

function PracticalBayes.latent_step(rng, k::iFFBSKernel, block_names, c::ModelConditional)
    block_names == (:X,) || error("iffbs_kernel only handles the :X block (got $(block_names))")
    X = copy(c.values.X)
    k.latent!(rng, k.params(c.values), X)
    (; X=X)
end
