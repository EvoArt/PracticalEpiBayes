# The latent block of an EpidemicTrajectories LatentHMMData model, and the
# kernel that updates it once per Gibbs sweep.

"""
    LatentBlock(data)

Placeholder distribution for a [`LatentHMMData`](@ref EpidemicTrajectories.LatentHMMData)
model's latents, held as one `LatentState`. Declared discrete so PracticalBayes
keeps it in the value store, constant to the HMC blocks; its `logpdf` is zero
because the latent priors are scored by the likelihood and the sampler.

```julia
@model function m(data, mll)
    p ~ Beta(1, 1)
    L ~ LatentBlock(data)
    @addlogprob! mll((; p), data, L)
end
```
"""
struct LatentBlock{D} <: Distributions.Distribution{Distributions.Multivariate,Distributions.Discrete}
    data::D
end
Distributions.logpdf(::LatentBlock, ::EpidemicTrajectories.LatentState) = 0.0
Base.rand(::AbstractRNG, b::LatentBlock) = copy(EpidemicTrajectories.current_state(b.data))
Base.rand(b::LatentBlock) = copy(EpidemicTrajectories.current_state(b.data))

"""
    latent_kernel(sampler; params)

The Gibbs kernel for a [`LatentBlock`](@ref): calls `sampler` (from
`epidemic_latent_sampler(data)`) on a copy of the block's state once per sweep.
`params` is a tuple of parameter names or a closure, as for
[`iffbs_kernel`](@ref).

```julia
spl = Gibbs(:L => latent_kernel(epidemic_latent_sampler(data); params = (:p,)), :p => NUTS())
```
"""
struct LatentKernel{F,P} <: AbstractGibbsKernel
    sampler::F
    params::P
end

latent_kernel(sampler; params) = LatentKernel(sampler, _params_selector(params))

function PracticalBayes.gibbs_step(rng, k::LatentKernel, block_names, c::ModelConditional)
    length(block_names) == 1 || error("latent_kernel handles one LatentBlock (got $(block_names))")
    name = block_names[1]
    state = copy(getproperty(c.values, name))
    k.sampler(rng, k.params(c.values), state)
    NamedTuple{(name,)}((state,))
end
