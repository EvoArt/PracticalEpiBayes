# Interweave: the step scale of a random walk observed with noise, sampled
# jointly with the walk. The scale's posterior is known exactly, by
# integrating the Gaussian walk out.
using Test, PracticalBayes, PracticalEpiBayes, EpidemicTrajectories, Distributions, LinearAlgebra,
      StableRNGs, Statistics

@testset "Interweave against the exact posterior of a step scale" begin
    N, T, τ, s0 = 4, 10, 0.3, 10.0
    rng = StableRNG(11)
    y = cumsum(0.4 .* randn(rng, T, N); dims = 1) .+ τ .* randn(rng, T, N)
    stepw(model, data, i, t) = t == 1 ? 1.0 :
        pdf(Normal(data.latents.S[1, t-1, i], exp(model.ls)), data.latents.S[1, t, i])
    spec = @transitions [:a] :weights begin
        a -> a = stepw
    end
    lat = @latents begin
        S ~ Field(1, T; prior = Normal(0, s0), index = identity)
    end
    obs(model, data, i, t, sh) = (pdf(Normal(data.latents.S[1, t, i], τ), data.y[t, i]),)
    d = latent_hmm_data(; n_individuals = N, n_timepoints = T, trans_mat = spec,
                        starting_state = (m, dd, i) -> (1.0,), observation_weights = obs, latents = lat,
                        init = (; S = reshape(copy(y), 1, T, N)), y = y)
    @latent_model function toy(data)
        ls ~ Normal(log(0.5), 1)
    end
    iw = Interweave((:ls,), (v, u) -> ((; ls = v.ls + u), 0.0); field = :S)
    spl = latent_gibbs(d; n_adapt = 500, interweave = iw)
    chain = sample(StableRNG(3), toy(d), spl, 8000; n_adapts = 500, discard_initial = 500,
                   init = (; ls = log(0.5)), progress = false)
    draws = vec(chain[:ls])

    # log p(y | sigma): the walk's prior, steps and noise are all Gaussian
    R = SymTridiagonal([1.0; fill(2.0, T - 2); 1.0], fill(-1.0, T - 1))
    function loglik(ls)
        σ = exp(ls); Q = Matrix(I / s0^2 + R / σ^2 + I / τ^2)
        sum(1:N) do i
            b = y[:, i] ./ τ^2
            T / 2 * log(2π) - logdet(Q) / 2 + dot(b, Q \ b) / 2 - sum(abs2, y[:, i]) / (2τ^2) -
                T * log(sqrt(2π) * s0) - (T - 1) * log(sqrt(2π) * σ) - T * log(sqrt(2π) * τ)
        end
    end
    grid = range(log(0.02), log(5), length = 4000)
    lp = [logpdf(Normal(log(0.5), 1), g) + loglik(g) for g in grid]
    w = exp.(lp .- maximum(lp)); w ./= sum(w)
    m = sum(w .* grid); sd = sqrt(sum(w .* (grid .- m) .^ 2))
    @test abs(mean(draws) - m) < 0.25sd
    @test 0.8 < std(draws) / sd < 1.25
    c, nc = interweave_rates(iw; total = true)
    @test 0 < c < 1 && 0 < nc < 1
end
