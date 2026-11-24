using PracticalEpiBayes
using PracticalBayes
using EpidemicTrajectories
using Distributions
using Random
using StableRNGs: StableRNG
using Statistics: mean, std
using Test

# A ModelConditional needs a real PracticalBayes Model to read `c.model.args` and
# `c.values`. We build the smallest model that carries a `data` argument and the
# blocks the kernels resample, then call `gibbs_step` directly with a chosen X.
@model function _tiny(data)
    θ ~ Beta(1, 1)
    etas ~ PracticalBayes.filldist(Beta(1, 1), 3)
    ν ~ Beta(1, 1)
    nu ~ PracticalBayes.filldist(Beta(1, 1), 2)
    X ~ TrajectoryLatent(size(data.Y, 1), size(data.Y, 2))
    @addlogprob! 0.0
end

# A model whose data argument is not called `data`
@model function _other(obs)
    θ ~ Beta(1, 1)
end

# Drive a kernel's gibbs_step with an explicit X and values, as Gibbs would.
function _step(kernel, name, data, X; extra...)
    model = _tiny(data)
    vals = merge((; θ=0.5, etas=[0.5, 0.5, 0.5], ν=0.5, nu=[0.3, 0.3], X=X), NamedTuple(extra))
    c = PracticalBayes.ModelConditional(model, vals)
    PracticalBayes.gibbs_step(StableRNG(1), kernel, (name,), c)
end

@testset "PracticalEpiBayes" begin

    @testset "test_sensitivity_kernel: known-answer counts" begin
        # Tested-and-infected cells: (t1,i2)+, (t1,i3)0, (t1,i4)+, (t2,i1)+,
        # (t2,i3)+ → 4 positive, 1 negative → Beta(1+4, 1+1).
        Y = [-1  1  0  1;
              1 -1  1 -1]                      # (t, i) result matrix
        X = [1  2  2  2;
             2  1  2  1]                       # state (2 = infected)
        data = (; Y=Y)
        k = test_sensitivity_kernel(:θ; Y=Y, infected_state=2, prior=(1, 1))
        # Recover the counts the kernel would form.
        pos = neg = 0
        for t in axes(Y, 1), i in axes(Y, 2)
            y = Y[t, i]; (y < 0 || X[t, i] != 2) && continue
            y == 1 ? (pos += 1) : (neg += 1)
        end
        @test (pos, neg) == (4, 1)
        # Draw is a valid Beta sample and lands in (0,1).
        out = _step(k, :θ, data, X)
        @test keys(out) == (:θ,)
        @test 0 < out.θ < 1
        # Mean over many draws matches Beta(1+4, 1+1) mean = 5/7.
        draws = [rand(Beta(1 + pos, 1 + neg)) for _ in 1:20_000]
        @test isapprox(mean(draws), 5 / 7; atol=0.02)
    end

    @testset "@conjugate macro matches the prebuilt" begin
        Y = [-1 1 0 1; 1 -1 1 -1]
        X = [1 2 2 2; 2 1 2 1]
        data = (; Y=Y)
        k_macro = @conjugate θ ~ Beta(1, 1) begin
            pos = neg = 0
            for t in axes(data.Y, 1), i in axes(data.Y, 2)
                y = data.Y[t, i]
                (y < 0 || X[t, i] != 2) && continue
                y == 1 ? (pos += 1) : (neg += 1)
            end
            (pos, neg)
        end
        # Same RNG, same counts ⇒ identical draw as the prebuilt.
        k_pre = test_sensitivity_kernel(:θ; Y=Y, infected_state=2)
        @test _step(k_macro, :θ, data, X).θ == _step(k_pre, :θ, data, X).θ
    end

    # REGRESSION. The macro must bind `X`/`data`/`k` from the ModelConditional, not
    # from whatever happens to be in scope where it is expanded. The test above
    # cannot detect that: it defines local `X`/`data` right before the macro call,
    # so a body that silently captured those globals would still pass.
    #
    # These build the kernel INSIDE a function, where no `X`/`data` binding exists
    # in the expansion scope. Before the `esc` fix in the macro, both raised
    # `UndefVarError: X not defined` when the kernel ran.
    @testset "@conjugate binds X/data/k from the conditional, not the call site" begin
        # No `X` or `data` local exists here — only inside `_step`'s conditional.
        make_scalar_kernel() = @conjugate θ ~ Beta(1, 1) begin
            pos = neg = 0
            for t in axes(data.Y, 1), i in axes(data.Y, 2)
                y = data.Y[t, i]
                (y < 0 || X[t, i] != 2) && continue
                y == 1 ? (pos += 1) : (neg += 1)
            end
            (pos, neg)
        end

        Y = [-1  1  0  1;
              1 -1  1 -1]
        Xt = [1  2  2  2;
              2  1  2  1]
        dat = (; Y=Y)
        k_macro = make_scalar_kernel()
        k_pre = test_sensitivity_kernel(:θ; Y=Y, infected_state=2)
        @test _step(k_macro, :θ, dat, Xt).θ == _step(k_pre, :θ, dat, Xt).θ

        # The indexed form additionally binds `k` (the index) for the body.
        # Counts per index differ, so a mis-bound `k` changes the answer.
        make_vec_kernel() = @conjugate etas[1:3] ~ Beta(1, 1) begin
            # index k selects the state counted as a "success" at row 1
            hit = miss = 0
            for i in axes(X, 2)
                X[1, i] == k ? (hit += 1) : (miss += 1)
            end
            (hit, miss)
        end
        kv = make_vec_kernel()
        out = _step(kv, :etas, dat, Xt)
        @test length(out.etas) == 3
        @test all(0 .< out.etas .< 1)
    end

    @testset "capture_prob_kernel: vector-valued, per index" begin
        # 2 timepoints, 2 individuals; season index picks which eta applies.
        Y = zeros(Int, 2, 2)                          # only used to size the model
        X = [1 4; 1 1]                                 # (t,i); state 4 = dead
        season   = [1, 2]                              # index[t]
        group    = [1 1; 1 1]                          # group[i,t]
        effort   = [1 1]                               # effort[g,t] (one group)
        captured = [1 0; 0 1]                          # caught[t,i]
        data = (; Y=Y)
        k = capture_prob_kernel(:etas; caught=captured, effort=effort, group=group,
                                index=season, dead_state=4, n=2, prior=(1, 1))
        out = _step(k, :etas, data, X)
        @test keys(out) == (:etas,)
        @test length(out.etas) == 2
        @test all(0 .< out.etas .< 1)
    end

    # A refit on data truncated at a cutoff keeps the full observation arrays and
    # clamps each sampling period instead. Counting outside the period would let
    # results after the cutoff inform the parameter. Each kernel, given a clamped
    # period, must draw exactly what it draws when those cells are masked out.
    @testset "conjugate kernels count only each individual's sampling period" begin
        Y = [-1  1  0  1;
              1 -1  1 -1]
        X = [1  2  2  2;
             2  1  2  1]
        sp = [(1, 2), (1, 2), (1, 1), (1, 2)]      # individual 3 stops at t = 1
        Ymasked = copy(Y); Ymasked[2, 3] = -1      # ...so its t = 2 positive goes
        k = test_sensitivity_kernel(:θ; Y=Y, infected_state=2)
        k_masked = test_sensitivity_kernel(:θ; Y=Ymasked, infected_state=2)
        clamped = _step(k, :θ, (; Y=Y, sampling_period=sp), X).θ
        @test clamped == _step(k_masked, :θ, (; Y=Y), X).θ
        @test clamped != _step(k, :θ, (; Y=Y), X).θ

        Xc = [1 4; 1 1]
        season = [1, 2]; effort = [1 1]; captured = [1 0; 0 1]
        group = [1 1; 1 1]
        group_masked = [1 1; 1 0]                  # individual 2 unavailable at t = 2
        spc = [(1, 2), (1, 1)]
        kc = capture_prob_kernel(:etas; caught=captured, effort=effort, group=group,
                                 index=season, dead_state=4, n=2)
        kc_masked = capture_prob_kernel(:etas; caught=captured, effort=effort,
                                        group=group_masked, index=season,
                                        dead_state=4, n=2)
        d = (; Y=zeros(Int, 2, 2))
        clamped = _step(kc, :etas, merge(d, (; sampling_period=spc)), Xc).etas
        @test clamped == _step(kc_masked, :etas, d, Xc).etas
        @test clamped != _step(kc, :etas, d, Xc).etas
    end

    @testset "initial_state_kernel: Dirichlet mixing" begin
        Y = zeros(Int, 3, 4)
        X = [2 1 1 2;                                  # t=1 states: two S(1), two I(2)
             1 1 1 1;
             1 1 1 1]
        data = (; Y=Y)
        eligible = (X, data, i, t) -> true
        # Zero-config default for the n=1, two-state case returns the scalar P(I).
        k = initial_state_kernel(:ν; at=1, eligible=eligible, states=[1, 2],
                                 prior=[1.0, 1.0])
        out = _step(k, :ν, data, X)
        @test keys(out) == (:ν,)
        @test out.ν isa Real
        @test 0 < out.ν < 1
        # counts were S=2, I=2 → Dirichlet([3,3]); mean of the I component ≈ 0.5
        draws = [rand(Dirichlet([3.0, 3.0]))[2] for _ in 1:20_000]
        @test isapprox(mean(draws), 0.5; atol=0.02)
    end

    @testset "initial_state_kernel: n>1 matrix default (badger nu layout)" begin
        Y = zeros(Int, 2, 3)
        X = [1 2 3;                                   # t=1: one each of S,E,I
             1 1 1]
        data = (; Y=Y)
        eligible = (X, data, i, t) -> true
        # Two cohorts, three states → n×2 matrix of the (E, I) components.
        k = initial_state_kernel(:nu; at=[1, 1], eligible=eligible, states=[1, 2, 3],
                                 prior=[1.0, 1.0, 1.0], n=2)
        out = _step(k, :nu, data, X)
        @test keys(out) == (:nu,)
        @test size(out.nu) == (2, 2)                  # n × (states-1)
        @test all(0 .< out.nu .< 1)
    end

    @testset "generic ConjugateGibbs (PracticalBayes engine)" begin
        Y = [-1 1 0 1; 1 -1 1 -1]
        X = [1 2 2 2; 2 1 2 1]
        data = (; Y=Y)
        # The name-first constructor and the `do`-block (count-first) form build the
        # same kernel; with the same RNG they must draw identically.
        cnt(c) = (count(==(2), c.values.X), count(==(1), c.values.X))
        k1 = ConjugateGibbs(:θ, :beta, (1, 1), cnt)
        k2 = ConjugateGibbs(:θ, :beta, (1, 1)) do c
            (count(==(2), c.values.X), count(==(1), c.values.X))
        end
        d1 = _step(k1, :θ, data, X).θ
        d2 = _step(k2, :θ, data, X).θ
        @test d1 == d2
        @test 0 < d1 < 1
    end

    # -----------------------------------------------------------------------
    # iffbs_kernel's `params` name-tuple form
    # -----------------------------------------------------------------------

    @testset "params as a name tuple selects from the Gibbs state" begin
        data = (; Y = fill(-1, 4, 3))
        seen = Ref{Any}(nothing)
        latent!(rng, pars, X) = (seen[] = pars; X)
        k = iffbs_kernel(latent!; params = (:θ, :ν))
        X = fill(1, 4, 3)
        out = _step(k, :X, data, X)
        @test out.X == X
        @test seen[] == (; θ=0.5, ν=0.5)
        # The names live in the TYPE, so the selection is inferable.
        @test seen[] isa NamedTuple{(:θ, :ν)}
    end

    @testset "params tuple order is the tuple's, not the model's" begin
        data = (; Y = fill(-1, 4, 3))
        seen = Ref{Any}(nothing)
        latent!(rng, pars, X) = (seen[] = pars; X)
        _step(iffbs_kernel(latent!; params = (:ν, :θ)), :X, data, fill(1, 4, 3))
        @test keys(seen[]) == (:ν, :θ)
    end

    @testset "a params name the model lacks errors, naming it" begin
        # The whole reason the tuple form exists is to catch this at the first
        # sweep with a clear message, rather than as a `no field` raised from
        # deep inside a rate function.
        data = (; Y = fill(-1, 4, 3))
        latent!(rng, pars, X) = X
        k = iffbs_kernel(latent!; params = (:θ, :not_a_param))
        err = try
            _step(k, :X, data, fill(1, 4, 3)); nothing
        catch e
            e
        end
        @test err !== nothing
        @test occursin("not_a_param", sprint(showerror, err))
    end

    @testset "a params closure still works unchanged" begin
        data = (; Y = fill(-1, 4, 3))
        seen = Ref{Any}(nothing)
        latent!(rng, pars, X) = (seen[] = pars; X)
        k = iffbs_kernel(latent!; params = v -> (; a = v.θ + 1.0))
        _step(k, :X, data, fill(1, 4, 3))
        @test seen[] == (; a = 1.5)
    end

    @testset "a vector of names is accepted too" begin
        data = (; Y = fill(-1, 4, 3))
        seen = Ref{Any}(nothing)
        latent!(rng, pars, X) = (seen[] = pars; X)
        _step(iffbs_kernel(latent!; params = [:θ, :ν]), :X, data, fill(1, 4, 3))
        @test keys(seen[]) == (:θ, :ν)
    end

    # -----------------------------------------------------------------------
    # HMC step sizes
    # -----------------------------------------------------------------------

    @testset "hmc_step_sizes concatenates in the order given" begin
        eps = hmc_step_sizes(:tau => 0.002, :alpha => (0.2, 3), :q => 0.05)
        @test collect(eps) == [0.002, 0.2, 0.2, 0.2, 0.05]
        @test eps.names == (:tau, :alpha, :q)
        @test eps.lens == (1, 3, 1)
        @test length(eps) == 5
    end

    @testset "vector lengths from `lengths=` or a per-element vector" begin
        e1 = hmc_step_sizes(:a => 0.1, :v => 0.2; lengths = (; v = 3))
        @test collect(e1) == [0.1, 0.2, 0.2, 0.2]
        e2 = hmc_step_sizes(:a => 0.1, :v => [0.2, 0.3, 0.4])
        @test collect(e2) == [0.1, 0.2, 0.3, 0.4]
        @test e2.lens == (1, 3)
    end

    @testset "a zero-length vector parameter contributes nothing" begin
        eps = hmc_step_sizes(:a => 0.1, :v => (0.2, 0))
        @test collect(eps) == [0.1]
    end

    @testset "check_step_sizes accepts a matching block" begin
        eps = hmc_step_sizes(:tau => 0.002, :alpha => (0.2, 3), :q => 0.05)
        @test check_step_sizes(eps, (:tau, :alpha, :q)) === eps
    end

    @testset "check_step_sizes CATCHES a reordered block" begin
        # The metric is a flat vector, so same-names-wrong-order silently gives
        # one parameter another's step size. That is the failure this exists for,
        # and it must be an error rather than a warning.
        eps = hmc_step_sizes(:tau => 0.002, :alpha => (0.2, 3), :q => 0.05)
        err = try
            check_step_sizes(eps, (:alpha, :tau, :q)); nothing
        catch e
            e
        end
        @test err !== nothing
        @test occursin("ORDER", sprint(showerror, err))
    end

    @testset "check_step_sizes CATCHES a missing or extra name" begin
        eps = hmc_step_sizes(:tau => 0.002, :q => 0.05)
        m = sprint(showerror, try check_step_sizes(eps, (:tau, :q, :beta)) catch e; e end)
        @test occursin("no step size for", m) && occursin("beta", m)
        m2 = sprint(showerror, try check_step_sizes(eps, (:tau,)) catch e; e end)
        @test occursin("non-member", m2) && occursin("q", m2)
    end

    @testset "hmc_block builds an HMC kernel with the squared metric" begin
        eps = hmc_step_sizes(:a => 0.1, :v => (0.2, 2))
        @test hmc_block(eps, 15) isa PracticalBayes.HMC
        # A plain vector works too, for a caller not using hmc_step_sizes.
        @test hmc_block([0.1, 0.2, 0.2], 15) isa PracticalBayes.HMC
    end

    @testset "a bad step-size spec is refused" begin
        m = sprint(showerror, try hmc_step_sizes(:a => "oops") catch e; e end)
        @test occursin("needs a number", m)
        m2 = sprint(showerror, try hmc_step_sizes(:a => (0.1, -2)) catch e; e end)
        @test occursin("negative length", m2)
    end

    @testset "@latent_model and latent_gibbs" begin
        # S -> I with a per-step rate r, observed with 10% error; u is a
        # per-individual effect on the rate
        expit(x) = 1 / (1 + exp(-x))
        N, T = 150, 10
        rng = StableRNG(11)
        u0 = randn(rng, N)
        y = zeros(Int, T, N)
        for i in 1:N
            s = 1
            for t in 1:T
                t > 1 && s == 1 && rand(rng) < expit(log(0.2 / 0.8) + 0.5u0[i]) && (s = 2)
                y[t, i] = rand(rng) < 0.9 ? s : 3 - s
            end
        end
        rate(m, d, i, t) = t == 1 ? 0.0 : expit(log(m.r / (1 - m.r)) + 0.5 * d.latents.u[i])
        spec = @transitions [:S, :I] :weights begin
            S -> S = 1 - rate
            S -> I = rate
            I -> I = 1
        end
        data = latent_hmm_data(; n_individuals = N, n_timepoints = T, trans_mat = spec,
                               starting_state = (m, d, i) -> (1.0, 0.0),
                               observation_weight = (m, d, i, t, s, sh) -> s == d.y[t, i] ? 0.9 : 0.1,
                               latents = @latents(begin u ~ PerIndividual(Normal(0, 1)) end), y = y)
        @latent_model function tiny_latent(data)
            r ~ Beta(1, 1)
        end
        m = tiny_latent(data)
        spl = latent_gibbs(data, :r => AdaptiveHMC(0.8; n_leapfrog = 4); n_adapt = 200)
        chn = sample(StableRNG(3), m, spl, 600; n_adapts = 200, discard_initial = 200, init = (; r = 0.5),
                     progress = false)
        @test abs(mean(chn[:r]) - 0.2) < 0.05
        # the kernel passes every parameter by default
        k = latent_kernel(epidemic_latent_sampler(data))
        @test k.params === nothing
        # the placeholder block draws the data's current latent state, scored as 0
        b = LatentBlock(data)
        L = rand(b)
        @test logpdf(b, L) == 0.0
        @test typeof(rand(StableRNG(1), b)) == typeof(L)
    end

    @testset "placeholder trajectory distribution" begin
        d = TrajectoryLatent(3, 4)
        @test size(d) == (3, 4)
        @test rand(StableRNG(1), d) == fill(1, 3, 4)
        @test logpdf(d, fill(2, 3, 4)) == 0.0
    end

    @testset "@conjugate: Dirichlet, data fallback, malformed calls" begin
        X = [1 2 3 3; 1 1 1 1]
        dat = (; Y=zeros(Int, 2, 4))
        kd = @conjugate etas ~ Dirichlet([1.0, 1.0, 1.0]) begin
            [count(==(s), X[1, :]) for s in 1:3]
        end
        out = _step(kd, :etas, dat, X)
        @test length(out.etas) == 3 && sum(out.etas) ≈ 1
        # n = 1 with three states: the default collect returns the probability vector
        ki = initial_state_kernel(:etas; at=1, eligible=(X, d, i, t) -> true, states=[1, 2, 3],
                                  prior=[1.0, 1.0, 1.0])
        pi3 = _step(ki, :etas, dat, X).etas
        @test length(pi3) == 3 && sum(pi3) ≈ 1
        # a model whose data argument is not called `data`: the first argument is used
        c = PracticalBayes.ModelConditional(_other(dat), (; θ=0.5))
        @test PracticalEpiBayes._data(c) === dat
        bad(ex) = try
            macroexpand(@__MODULE__, ex); nothing
        catch e
            e
        end
        @test bad(:(@conjugate θ + Beta(1, 1) begin (1, 1) end)) !== nothing
        @test bad(:(@conjugate (a, b) ~ Beta(1, 1) begin (1, 1) end)) !== nothing
        @test bad(:(@conjugate a.b[1:2] ~ Beta(1, 1) begin (1, 1) end)) !== nothing
        @test bad(:(@conjugate θ ~ 3 begin (1, 1) end)) !== nothing
        @test bad(:(@conjugate θ ~ Beta(1) begin (1, 1) end)) !== nothing
        @test bad(:(@conjugate θ ~ Dirichlet(1, 2) begin [1] end)) !== nothing
        @test bad(:(@conjugate θ ~ Gamma(1, 1) begin (1, 1) end)) !== nothing
        @test bad(:(@conjugate θ ~ Beta(1, 1) begin (1, 1) end)) === nothing
    end

end

include("interweave.jl")
