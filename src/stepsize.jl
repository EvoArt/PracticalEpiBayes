# Building an HMC block's diagonal metric from per-parameter step sizes.
#
# `HMC(L; metric=DiagEuclideanMetric(eps .^ 2))` wants ONE flat vector whose
# ordering matches the block's parameters as the layout lays them out. Written by
# hand that is a `vcat` of scalars and `fill`s whose correspondence to the block's
# name list is maintained only by a comment:
#
#     # one leading `tau`, `n_groups` alphas, then the scalar epidemic params, ...
#     vcat(0.002, fill(0.2, n_groups), 0.01, 0.05, ...)
#
# Nothing checks that. Adding a parameter, reordering the block, or changing a
# vector parameter's length silently misaligns every step size after the change —
# the sampler still runs, with the wrong metric, and the only symptom is worse
# mixing. `hmc_step_sizes` takes the names and lengths explicitly so the mistake
# becomes an error.

"""
    hmc_step_sizes(pairs...; lengths=NamedTuple())

Build the flat step-size vector for an HMC block's diagonal metric, from
`:name => step` pairs given in the block's own order.

A vector-valued parameter needs its length, either as `:name => (step, n)` or via
`lengths = (; name = n)`. Passing a per-element vector (`:name => [s1, s2, s3]`)
works too and fixes the length from it.

The result is the concatenation in the order the pairs are given, which must be
the order of the block's names in `Gibbs` — that ordering is the thing this
function exists to make explicit rather than implicit.

```julia
eps = hmc_step_sizes(
    :tau => 0.002, :alpha => (0.2, n_groups), :lambda => 0.01, :beta => 0.05,
    :q => 0.05, :c1 => 0.02, :a2 => 0.001, :b2 => 0.001,
    :thetas => (0.005, n_tests), :rhos => (0.005, n_tests),
    :phis => (0.005, n_tests))

block = (:tau, :alpha, :lambda, :beta, :q, :c1, :a2, :b2, :thetas, :rhos, :phis)
spl = Gibbs(block => hmc_block(eps, 15), ...)
```

Use [`check_step_sizes`](@ref) to assert the result lines up with the block's
names before sampling.
"""
function hmc_step_sizes(pairs::Pair{Symbol}...; lengths=NamedTuple())
    out = Float64[]
    names = Symbol[]
    for (name, spec) in pairs
        vals = _expand_step(name, spec, lengths)
        append!(out, vals)
        push!(names, name)
    end
    return StepSizes(out, Tuple(names), Tuple(_expand_len(n, s, lengths)
                                              for (n, s) in pairs))
end

"""
    StepSizes

The flat step-size vector built by [`hmc_step_sizes`](@ref), carrying the names
and per-name lengths it was built from so the layout can be checked. Behaves as
the vector wherever one is wanted.
"""
struct StepSizes <: AbstractVector{Float64}
    values::Vector{Float64}
    names::Tuple{Vararg{Symbol}}
    lens::Tuple{Vararg{Int}}
end

Base.size(s::StepSizes) = size(s.values)
Base.getindex(s::StepSizes, i::Int) = s.values[i]
Base.IndexStyle(::Type{StepSizes}) = IndexLinear()

function _expand_step(name::Symbol, spec, lengths)
    if spec isa Tuple{<:Real,<:Integer}
        step, n = spec
        n >= 0 || error("hmc_step_sizes: `:$name` has negative length $n.")
        return fill(Float64(step), n)
    elseif spec isa AbstractVector{<:Real}
        return Float64.(collect(spec))
    elseif spec isa Real
        n = hasproperty(lengths, name) ? getproperty(lengths, name) : 1
        return fill(Float64(spec), n)
    end
    error("hmc_step_sizes: `:$name` needs a number, a `(step, n)` tuple, or a " *
          "vector of per-element steps; got $(typeof(spec)).")
end

function _expand_len(name::Symbol, spec, lengths)
    spec isa Tuple{<:Real,<:Integer} && return Int(spec[2])
    spec isa AbstractVector{<:Real} && return length(spec)
    return hasproperty(lengths, name) ? Int(getproperty(lengths, name)) : 1
end

"""
    check_step_sizes(eps::StepSizes, block_names; lengths=NamedTuple())

Assert that `eps` was built for exactly `block_names`, in that order. Throws a
message naming the discrepancy otherwise.

The metric is a flat vector, so a mismatch between it and the block's name order
is not a type error — it silently assigns one parameter's step size to another.
This is the check that turns that into a failure.
"""
function check_step_sizes(eps::StepSizes, block_names)
    bn = Tuple(block_names)
    if eps.names != bn
        extra = setdiff(eps.names, bn)
        missing_ = setdiff(bn, eps.names)
        msg = "check_step_sizes: step sizes do not match the block.\n" *
              "  block: $(bn)\n  steps: $(eps.names)"
        isempty(missing_) || (msg *= "\n  no step size for: " * join(missing_, ", "))
        isempty(extra) || (msg *= "\n  step size for non-member: " * join(extra, ", "))
        if isempty(missing_) && isempty(extra)
            msg *= "\n  same names, different ORDER - the metric is a flat " *
                   "vector, so order is what aligns it with the block."
        end
        error(msg)
    end
    return eps
end

"""
    hmc_block(eps, n_steps)

The fixed-trajectory HMC kernel used with a hand-tuned diagonal metric:
`HMC(n_steps; integrator=Leapfrog(1.0), metric=DiagEuclideanMetric(eps .^ 2))`.

`n_steps` should match the EXPECTED trajectory length you are targeting, not a
nominal maximum — the C++ badger reference draws `L` uniformly on `1..30` (mean
15.5), so a fixed `L = 30` does 1.8x the work for no added fidelity.
"""
function hmc_block(eps, n_steps::Integer)
    v = eps isa StepSizes ? eps.values : collect(Float64, eps)
    return PracticalBayes.HMC(n_steps;
        integrator=AdvancedHMC.Leapfrog(1.0),
        metric=AdvancedHMC.DiagEuclideanMetric(v .^ 2))
end
