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
    Interweave(params, step; field, elements = nothing, radius = Inf, scale = 0.05, centred = 0.05, n = 3)

Joint moves of some parameters and a `Field`'s paths, for a parameter that
sets the scale of the steps between an individual's consecutive elements:
e.g. a movement scale and the activity centres it links. One at a time,
each mixes slowly, because a narrow step scale pins the paths and the paths
pin the scale. Interweaving the centred and non-centred forms (Yu and Meng
2011) moves them together.

Each sweep, after the latents, `n` rounds of:
- a centred update: a random walk (sd `centred`) on each of `params`, the
  paths held;
- a non-centred update: `u ~ N(0, scale^2)`, the parameters moved by
  `step(values, u)`, and every individual's steps between its elements in use
  multiplied by `exp(u)` (each path anchored at its first element in use).
  `elements(i)`, optional, gives the elements of individual `i` to scale
  instead of every element in use, e.g. only the ones the likelihood reads.
  With `radius = c`, each step of length `r` is stretched along its
  direction by the flow of `dr/du = r / (1 + (r/c)^2)` for time `u` instead:
  steps much shorter than `c` by about `exp(u)`, steps much longer than `c`
  hardly at all. Use it when the parameter scales the short steps only, e.g.
  a resident movement scale when some steps are long dispersals.

`step(values, u)` returns `(new, logjac)`: a NamedTuple of the new values of
`params`, and the log Jacobian of that map in the parameters' sampled
coordinates. Return `nothing` to reject. Both updates are Metropolis steps on
the full log joint plus the paths' own priors, so the model is unchanged. The
sds adapt during warm-up.

The parameters named here belong to the latent block, so leave them out of
the HMC blocks.
"""
struct Interweave{F,E}
    params::Tuple{Vararg{Symbol}}
    step::F
    field::Symbol
    elements::E
    radius::Float64
    scale::Base.RefValue{Float64}
    centred::Base.RefValue{Float64}
    n::Int
    tally::Vector{Float64}          # centred accepted, tried; non-centred accepted, tried: last sweep, then all sweeps
end
function Interweave(params, step; field, elements = nothing, radius = Inf, scale = 0.05, centred = 0.05, n = 3)
    radius > 0 || throw(ArgumentError("Interweave: radius must be positive"))
    Interweave(Tuple(Symbol.(params)), step, Symbol(field), elements, Float64(radius), Ref(Float64(scale)),
               Ref(Float64(centred)), Int(n), zeros(8))
end

"""
    interweave_rates(iw::Interweave; total = false) -> (centred, noncentred)

The acceptance rates of an [`Interweave`](@ref)'s two updates: in the last sweep,
or with `total = true` over every sweep before it.
"""
interweave_rates(iw::Interweave; total = false) = (o = total ? 4 : 0;
    (iw.tally[o+1] / max(iw.tally[o+2], 1), iw.tally[o+3] / max(iw.tally[o+4], 1)))

struct InterweaveKernel{F,I<:Tuple,P} <: AbstractGibbsKernel
    sampler::F
    iws::I
    n_adapt::Int
    sweep::Base.RefValue{Int}
    params::P
end

"""
    interweave_kernel(sampler, iws::Interweave...; n_adapt = 0, params = nothing)

The Gibbs kernel for a [`LatentBlock`](@ref) together with one or more
[`Interweave`](@ref)s' parameters: `sampler` (from
`epidemic_latent_sampler(data)`) updates the latents, then each `Interweave`
in turn moves its parameters and the paths. List the block first, then every
parameter once: `(:L, iw.params...) => interweave_kernel(smp, iw; n_adapt)`.
`params` is as for
[`latent_kernel`](@ref); `n_adapt` the number of sweeps over which the
interweaving step sizes adapt.
"""
interweave_kernel(sampler, iws::Interweave...; n_adapt = 0, params = nothing) =
    InterweaveKernel(sampler, iws, Int(n_adapt), Ref(0), params === nothing ? nothing : _params_selector(params))

# The length a step of length r has after time u of dr/du = r / (1 + (r/c)^2),
# i.e. log r' + r'^2 / 2c^2 = log r + r^2 / 2c^2 + u (Newton in log r').
function _radial_flow(r, u, c)
    isinf(c) && return r * exp(u)
    r > 0 || return r
    K = log(r) + r^2 / (2c^2) + u
    x = log(r) + u / (1 + r^2 / c^2)
    for _ in 1:100
        e = exp(2x) / c^2
        dx = (x + e / 2 - K) / (1 + e)
        x -= dx
        abs(dx) < 1e-13 * max(1, abs(x)) && break
    end
    exp(x)
end

# Every individual's run of elements in use (or `elements(i)`), each step
# stretched by the radial flow (by exp(u) when the radius is infinite), the
# path rebuilt from its first element; returns the log Jacobian.
function _scale_paths!(S, k, data, u, elements, c = Inf)
    jac = 0.0
    step = zeros(k.dim)
    for i in 1:data.n_individuals
        lo, hi = data.sampling_period[i]
        E = elements !== nothing ? collect(elements(i)) :
            [e for e in 1:k.n if (w = k.touches(e); max(first(w), lo) <= min(last(w), hi))]
        length(E) > 1 || continue
        prev_old = S[:, E[1], i]
        for j in 2:length(E)
            e = E[j]
            for d in 1:k.dim
                step[d] = S[d, e, i] - prev_old[d]
                prev_old[d] = S[d, e, i]
            end
            r = sqrt(sum(abs2, step))
            r2 = _radial_flow(r, u, c)
            f = r > 0 ? r2 / r : exp(u)
            for d in 1:k.dim
                S[d, e, i] = S[d, E[j-1], i] + f * step[d]
            end
            # radial map in dim dimensions: (r'/r)^dim * (1 + r^2/c^2) / (1 + r'^2/c^2)
            jac += isinf(c) ? k.dim * u : k.dim * log(f) + log1p((r / c)^2) - log1p((r2 / c)^2)
        end
    end
    jac
end
_field_logprior(S, k, data) = sum(EpidemicTrajectories._vec_logprior(k.prior, view(S, :, e, i))
                                  for i in 1:data.n_individuals, e in 1:k.n)

function PracticalBayes.gibbs_step(rng, kern::InterweaveKernel, block_names, c::ModelConditional)
    data = kern.sampler.data
    kern.sweep[] += 1
    adapt = kern.sweep[] <= kern.n_adapt
    name = block_names[1]
    state = copy(getproperty(c.values, name))
    kern.sampler(rng, kern.params === nothing ? Base.structdiff(c.values, NamedTuple{(name,)}) : kern.params(c.values),
                 state)
    vals = merge(c.values, NamedTuple{(name,)}((state,)))
    lj = PracticalBayes.logjoint(ModelConditional(c.model, vals))
    rw(x, sd) = x isa AbstractArray ? x .+ sd .* randn(rng, size(x)) : x + sd * randn(rng)
    for iw in kern.iws
        k = data.latent_decls[iw.field]
        iw.tally[5:8] .+= view(iw.tally, 1:4); fill!(view(iw.tally, 1:4), 0.0)
        for _ in 1:iw.n
            # centred: the parameters alone
            prop = merge(vals, NamedTuple{iw.params}(map(p -> rw(vals[p], iw.centred[]), iw.params)))
            ljp = PracticalBayes.logjoint(ModelConditional(c.model, prop))
            a = isfinite(ljp) && log(rand(rng)) < ljp - lj
            a && (vals = prop; lj = ljp)
            iw.tally[1] += a; iw.tally[2] += 1
            adapt && (iw.centred[] = clamp(iw.centred[] * exp(kern.sweep[]^-0.6 * (a - 0.35)), 1e-6, 1e3))
            # non-centred: the parameters and every path's steps together
            u = iw.scale[] * randn(rng)
            r = iw.step(vals, u)
            a = false
            if r !== nothing
                new, logjac = r
                st = copy(vals[name])
                S = st.values[iw.field]
                lp0 = _field_logprior(S, k, data)
                jac = _scale_paths!(S, k, data, u, iw.elements, iw.radius)
                EpidemicTrajectories.restamp!(st)
                prop = merge(vals, new, NamedTuple{(name,)}((st,)))
                ljp = PracticalBayes.logjoint(ModelConditional(c.model, prop))
                if isfinite(ljp) && log(rand(rng)) < ljp - lj + _field_logprior(S, k, data) - lp0 + logjac + jac
                    a = true; vals = prop; lj = ljp
                end
            end
            iw.tally[3] += a; iw.tally[4] += 1
            adapt && (iw.scale[] = clamp(iw.scale[] * exp(kern.sweep[]^-0.6 * (a - 0.35)), 1e-6, 1e3))
        end
    end
    data.current[] = vals[name]
    NamedTuple{block_names}(Tuple(vals[n] for n in block_names))
end

"""
    latent_gibbs(data, blocks...; n_adapt = 0, seed = 1, interweave = nothing)

The Gibbs sampler for an [`@latent_model`](@ref): the latents updated by
`epidemic_latent_sampler(data)` (whose proposal scales adapt over the first
`n_adapt` sweeps), then each block as given, e.g.
`(:a, :b) => AdaptiveHMC(0.8; n_leapfrog = 10)`. With an [`Interweave`](@ref) (or a
tuple of them), the latent block also owns their parameters and interweaves
them with the paths.
"""
function latent_gibbs(data, blocks::Pair...; n_adapt = 0, seed = 1, interweave = nothing)
    smp = EpidemicTrajectories.epidemic_latent_sampler(data; n_adapt, seed)
    interweave === nothing && return PracticalBayes.Gibbs(:latents => latent_kernel(smp), blocks...)
    iws = interweave isa Interweave ? (interweave,) : Tuple(interweave)
    pars = Tuple(unique(Iterators.flatten(iw.params for iw in iws)))
    PracticalBayes.Gibbs((:latents, pars...) => interweave_kernel(smp, iws...; n_adapt),
                         blocks...)
end
