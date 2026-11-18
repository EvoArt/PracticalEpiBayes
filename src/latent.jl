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
[`iffbs_kernel`](@ref); by default every value but the block itself.

```julia
spl = Gibbs(:L => latent_kernel(epidemic_latent_sampler(data); params = (:p,)), :p => NUTS())
```
"""
struct LatentKernel{F,P} <: AbstractGibbsKernel
    sampler::F
    params::P
end

latent_kernel(sampler; params = nothing) = LatentKernel(sampler, params === nothing ? nothing : _params_selector(params))

function PracticalBayes.gibbs_step(rng, k::LatentKernel, block_names, c::ModelConditional)
    length(block_names) == 1 || error("latent_kernel handles one LatentBlock (got $(block_names))")
    name = block_names[1]
    state = copy(getproperty(c.values, name))
    pars = k.params === nothing ? Base.structdiff(c.values, NamedTuple{(name,)}) : k.params(c.values)
    k.sampler(rng, pars, state)
    NamedTuple{(name,)}((state,))
end

"""
    latent_loglik(pars, data, latents)

The marginal log likelihood of a `LatentHMMData` model: `pars` a NamedTuple of
parameters, `latents` the block's `LatentState`.
"""
latent_loglik(pars, data, latents) = EpidemicTrajectories.epidemic_marginal_loglik(data)(pars, data, latents)

"""
    @latent_model function name(data, args...)
        a ~ Beta(1, 1)
        b ~ filldist(Normal(0, 1), 3)
        ...
    end

A PracticalBayes model for the `LatentHMMData` model `data` (the first
argument). Write the priors; the macro adds `latents ~ LatentBlock(data)` and
the marginal log likelihood given every parameter declared with `~` at the top
level of the body. Sample it with [`latent_gibbs`](@ref).
"""
macro latent_model(fdef)
    (fdef isa Expr && fdef.head in (:function, :(=))) || error("@latent_model wraps a function definition")
    sig, body = fdef.args
    data = sig.args[2]
    data isa Expr && (data = data.args[1])        # data::T
    names = Symbol[]
    for line in body.args
        line isa Expr && line.head == :call && line.args[1] == :~ && line.args[2] isa Symbol &&
            push!(names, line.args[2])
    end
    :latents in names && error("@latent_model: `latents` is the latent block's name")
    lnn = LineNumberNode(@__LINE__, @__FILE__)
    pars = Expr(:tuple, Expr(:parameters, names...))
    newbody = Expr(:block, body.args...,
        :(latents ~ $(GlobalRef(@__MODULE__, :LatentBlock))($data)),
        Expr(:macrocall, Symbol("@addlogprob!"), lnn,
             :($(GlobalRef(@__MODULE__, :latent_loglik))($pars, $data, latents))))
    esc(Expr(:macrocall, GlobalRef(PracticalBayes, Symbol("@model")), __source__,
             Expr(fdef.head, sig, newbody)))
end

"""
    latent_gibbs(data, blocks...; n_adapt = 0, seed = 1)

The Gibbs sampler for an [`@latent_model`](@ref): the latents updated by
`epidemic_latent_sampler(data)` (whose proposal scales adapt over the first
`n_adapt` sweeps), then each block as given, e.g.
`(:a, :b) => AdaptiveHMC(0.8; n_leapfrog = 10)`.
"""
latent_gibbs(data, blocks::Pair...; n_adapt = 0, seed = 1) =
    PracticalBayes.Gibbs(:latents => latent_kernel(EpidemicTrajectories.epidemic_latent_sampler(data; n_adapt, seed)), blocks...)
