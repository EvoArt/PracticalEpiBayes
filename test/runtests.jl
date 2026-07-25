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
# blocks the kernels resample, then call `latent_step` directly with a chosen X.
@model function _tiny(data)
    θ ~ Beta(1, 1)
    etas ~ PracticalBayes.filldist(Beta(1, 1), 3)
    ν ~ Beta(1, 1)
    nu ~ PracticalBayes.filldist(Beta(1, 1), 2)
    X ~ TrajectoryLatent(size(data.Y, 1), size(data.Y, 2))
    @addlogprob! 0.0
end

# Drive a kernel's latent_step with an explicit X and values, as Gibbs would.
function _step(kernel, name, data, X; extra...)
    model = _tiny(data)
    vals = merge((; θ=0.5, etas=[0.5, 0.5, 0.5], ν=0.5, nu=[0.3, 0.3], X=X), NamedTuple(extra))
    c = PracticalBayes.ModelConditional(model, vals)
    PracticalBayes.latent_step(StableRNG(1), kernel, (name,), c)
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

end
