# Conjugate Gibbs kernels for standard epidemic parameters.
#
# Fitting an EpidemicTrajectories model with PracticalBayes means writing, next to
# the iFFBS trajectory kernel, a handful of small closed-form Gibbs updates: a
# test's sensitivity, a capture probability, the initial-state mixing. Each has the
# same shape — sweep the current trajectory `X` (and some data) counting
# "successes" and "trials", then draw from a conjugate Beta or Dirichlet.
#
# The GENERIC engine for that ("given counts, draw the conjugate posterior") is not
# epidemic-specific and lives in PracticalBayes as `ConjugateGibbs`. This file is
# the thin epidemic layer over it: a macro (`@conjugate`) that lets the user write
# the counting loop in terms of `X` and `data`, and named prebuilts that write the
# loop for the exact recurring cases. All of them are just `ConjugateGibbs` with an
# epidemic-flavoured `count` closure that reaches `X`/`data` out of the
# `ModelConditional` for you.
#
# Nothing here constrains the general case: a user who wants a bespoke closed form
# writes a plain `ConjugateGibbs` (or a hand-rolled `AbstractLatentKernel`) exactly
# as they would without this package.

# The trajectory `X` and the `EpidemicData` a count closure reads, pulled out of
# the ModelConditional. `X` is a Gibbs block, so it is in `c.values`. `data` is not
# duplicated into the values; PracticalBayes exposes the model's call arguments on
# `c.model.args`, and the loglik term is `loglik_fn(pars, data, X)`, so the data is
# the model argument named `data` (falling back to the first positional argument).
_traj(c::ModelConditional) = c.values.X
function _data(c::ModelConditional)
    args = c.model.args
    hasproperty(args, :data) && return args.data
    return first(args)
end

# ---------------------------------------------------------------------------
# The @conjugate macro — write the count loop in terms of X and data.
# ---------------------------------------------------------------------------

"""
    @conjugate name ~ Beta(a, b) count_body
    @conjugate name ~ Dirichlet(αs) count_body
    @conjugate name[1:n] ~ Beta(a, b) count_body

Build a conjugate Gibbs kernel for the block `name` in one expression, without the
struct-plus-`latent_step` boilerplate. Expands to a PracticalBayes
[`ConjugateGibbs`](@ref) kernel you drop straight into `Gibbs`.

The `~` line names the block and its conjugate prior. `count_body` is ordinary
Julia with `X` and `data` in scope — the current trajectory (a `Matrix{Int}`
indexed `X[t, i]`) and the `EpidemicData` — that must **return the counts** the
family expects:

  * `Beta`      → `(successes, failures)`
  * `Dirichlet` → a vector of category counts (same length as `αs`)

The vector form `name[1:n]` runs the body once per index `k ∈ 1:n` (with `k` in
scope) and collects the `n` draws into a vector.

# Example — a test's sensitivity (cattle `θʳ`)

```julia
θʳ = @conjugate θʳ ~ Beta(1, 1) begin
    pos = neg = 0
    for t in axes(data.rams, 1), i in axes(data.rams, 2)
        y = data.rams[t, i]
        (y < 0 || X[t, i] != 2) && continue   # untested, or not infected
        y == 1 ? (pos += 1) : (neg += 1)
    end
    (pos, neg)
end
# ... spl = Gibbs(:θʳ => θʳ, ...)
```

# Example — capture probability per season (badger `etas`, vector-valued)

```julia
etas = @conjugate etas[1:NS] ~ Beta(1, 1) begin
    caught = available = 0
    for t in axes(X, 1)
        data.season[t] == k || continue        # `k` is the season index
        for i in axes(X, 2)
            g = data.social_group[i, t]
            (g > 0 && data.capt_effort[g, t] == 1 && X[t, i] != 4) || continue
            available += 1
            caught += data.capture[t, i] == 1
        end
    end
    (caught, available - caught)
end
```
"""
macro conjugate(sig, body)
    Meta.isexpr(sig, :call) && sig.args[1] === :~ ||
        error("@conjugate: expected `name ~ Family(...)` (got $(sig))")
    lhs, rhs = sig.args[2], sig.args[3]

    # Left side: bare `name`, or indexed `name[1:n]` / `name[n]`.
    if lhs isa Symbol
        name, n_expr = lhs, nothing
    elseif Meta.isexpr(lhs, :ref)
        name = lhs.args[1]
        name isa Symbol || error("@conjugate: bad name in `$(lhs)`")
        rng = lhs.args[2]
        n_expr = (Meta.isexpr(rng, :call) && rng.args[1] === :(:)) ? rng.args[3] : rng
    else
        error("@conjugate: bad left-hand side `$(lhs)` (want `name` or `name[1:n]`)")
    end

    # Right side: the conjugate family and its prior.
    Meta.isexpr(rhs, :call) ||
        error("@conjugate: expected a distribution call on the right of `~`")
    fam = rhs.args[1]
    if fam === :Beta
        length(rhs.args) == 3 || error("@conjugate: Beta needs two hyperparameters")
        family = QuoteNode(:beta)
        prior = Expr(:tuple, esc(rhs.args[2]), esc(rhs.args[3]))
    elseif fam === :Dirichlet
        length(rhs.args) == 2 || error("@conjugate: Dirichlet needs one concentration vector")
        family = QuoteNode(:dirichlet)
        prior = esc(rhs.args[2])
    else
        error("@conjugate: unsupported family `$(fam)` (want Beta or Dirichlet)")
    end

    # The count closure ConjugateGibbs wants takes the ModelConditional (and, for
    # the vector form, the index k). We bind `X`/`data` from it so the user's body
    # can be written in those plain terms.
    # `X`/`data` are pulled from the ModelConditional via this module's helpers,
    # module-qualified so they resolve wherever the macro is expanded.
    if n_expr === nothing
        count_fn = :(__c__ -> let X = $PracticalEpiBayes._traj(__c__),
                                  data = $PracticalEpiBayes._data(__c__)
                        $(esc(body))
                    end)
        n_val = :(nothing)
    else
        count_fn = :((__c__, k) -> let X = $PracticalEpiBayes._traj(__c__),
                                       data = $PracticalEpiBayes._data(__c__)
                        $(esc(body))
                    end)
        n_val = esc(n_expr)
    end

    :($PracticalBayes.ConjugateGibbs($(QuoteNode(name)), $family, $prior, $count_fn; n=$n_val))
end

# ---------------------------------------------------------------------------
# Prebuilt kernels — named helpers for the exact recurring epidemic cases.
# Each is a ConjugateGibbs whose count closure writes the loop for you.
# ---------------------------------------------------------------------------

"""
    test_sensitivity_kernel(name; Y, infected_state, prior=(1, 1))

Conjugate Beta kernel for a diagnostic test's sensitivity — the probability a
truly-infected individual tests positive. `Y[t, i]` is the result matrix (`1`
positive, `0` negative, negative = not tested); `infected_state` is the state index
counted as infected (`2` for S/I, an SEID `I`, etc.); `prior` is the `Beta(a, b)`.

Assumes perfect specificity (susceptibles never test positive), so only infected
tested cells contribute — the closed form behind the cattle `TestSensKernel`. For a
test with imperfect specificity, resample sensitivity and specificity with separate
kernels over the appropriate cells.
"""
function test_sensitivity_kernel(name::Symbol; Y, infected_state::Integer, prior=(1, 1))
    count = function (c)
        X = _traj(c)
        pos = 0
        neg = 0
        for t in axes(Y, 1), i in axes(Y, 2)
            y = Y[t, i]
            (y < 0 || X[t, i] != infected_state) && continue
            y == 1 ? (pos += 1) : (neg += 1)
        end
        (pos, neg)
    end
    PracticalBayes.ConjugateGibbs(name, :beta, prior, count)
end

"""
    capture_prob_kernel(name; caught, effort, group, index, dead_state, n,
                        prior=(1, 1), available=nothing)

Conjugate Beta kernel for a per-index capture/detection probability (the badger
`etas`): the chance an available individual is actually caught. One Beta is drawn
per index `k ∈ 1:n` and the draws are collected into a length-`n` vector.

An individual is *available* at `(t, i)` when it is being trapped
(`effort[group[i, t], t] == 1`, group `> 0`) and not dead (`X[t, i] != dead_state`),
and *caught* when additionally `caught[t, i] == 1`. `index[t]` selects which of the
`n` probabilities applies at time `t` (e.g. `data.season`).

`available`, if given, is a predicate `(X, data, i, t) -> Bool` replacing the
default "not dead" test, for models whose availability is more than one state.
"""
function capture_prob_kernel(name::Symbol; caught, effort, group, index,
                             dead_state::Integer, n::Integer, prior=(1, 1),
                             available=nothing)
    is_avail = available === nothing ?
        ((X, data, i, t) -> X[t, i] != dead_state) : available
    count = function (c, k)
        X = _traj(c)
        data = _data(c)
        got = 0
        avail = 0
        for t in axes(X, 1)
            index[t] == k || continue
            for i in axes(X, 2)
                g = group[i, t]
                (g > 0 && effort[g, t] == 1) || continue
                is_avail(X, data, i, t) || continue
                avail += 1
                got += caught[t, i] == 1
            end
        end
        (got, avail - got)
    end
    PracticalBayes.ConjugateGibbs(name, :beta, prior, count; n=n)
end

"""
    initial_state_kernel(name; at, eligible, states, prior, n=1, collect=nothing)

Conjugate Dirichlet kernel for the initial-state mixing of newly-entering
individuals (the cattle `ν`, the badger `nu`). For each of `n` entry times, it
counts how many eligible individuals start in each of `states` and draws the mixing
proportions from `Dirichlet(prior)`.

- `at` — a single time index (`n=1`) or, for `n > 1`, a vector of `n` entry times.
- `eligible` — predicate `(X, data, i, t) -> Bool` selecting the contributing
  individuals at cohort time `t`.
- `states` — state indices whose counts form the Dirichlet categories, ordered to
  match `prior`.
- `collect` — assembles the `n` probability vectors into the block value. The
  default covers both real cases with no configuration:
    * `n == 1`, two states → the scalar `P(states[2])` (the cattle `ν` = `P(I)`);
    * `n == 1`, more states → the single probability vector;
    * `n > 1` → an `n × (length(states) − 1)` matrix of the non-reference
      components (the badger `(nuE, nuI)` layout: the first state is the implied
      `1 − Σrest`).
  Pass your own `collect` to override.
"""
function initial_state_kernel(name::Symbol; at, eligible, states, prior,
                              n::Integer=1, collect=nothing)
    ns = length(states)
    if collect === nothing
        collect = if n == 1 && ns == 2
            ps -> ps[1][2]                                            # scalar P(second state)
        elseif n == 1
            ps -> ps[1]                                              # the one probability vector
        else
            ps -> reduce(vcat, (reshape(p[2:end], 1, :) for p in ps))  # n × (ns-1) matrix
        end
    end
    times = (n == 1 && !(at isa AbstractVector)) ? [at] : at
    count = function (c, k)
        X = _traj(c)
        data = _data(c)
        nt = times[k]
        counts = zeros(Int, ns)
        for i in axes(X, 2)
            eligible(X, data, i, nt) || continue
            s = X[nt, i]
            for (j, st) in enumerate(states)
                if s == st
                    counts[j] += 1
                    break
                end
            end
        end
        counts
    end
    PracticalBayes.ConjugateGibbs(name, :dirichlet, prior, count; n=n, collect=collect)
end
