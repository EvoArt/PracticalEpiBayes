module PracticalEpiBayes

# The glue between EpidemicTrajectories.jl (PPL-agnostic epidemic simulators /
# likelihoods / iFFBS latent samplers) and PracticalBayes.jl (the latent-kernel
# Gibbs seam). EpidemicTrajectories deliberately knows nothing about any PPL; this
# package supplies the small, repetitive PracticalBayes-shaped pieces you would
# otherwise hand-write per model to fit one of those models:
#
#   * the whole-trajectory latent block + its iFFBS kernel (trajectory.jl), and
#   * conjugate Gibbs kernels for the standard epidemic parameters — a test's
#     sensitivity, a capture probability, the initial-state mixing (conjugate.jl).
#
# The design mirrors EpidemicTrajectories' own rule: nothing here constrains the
# general case. The conjugate layer offers a compact macro (`@conjugate`) for
# arbitrary closed-form updates AND named prebuilts for the exact recurring cases,
# and a user who wants a bespoke kernel still writes one by hand exactly as they
# would without this package.

using Random: AbstractRNG
using Distributions: Distributions
using PracticalBayes: PracticalBayes, AbstractLatentKernel, latent_step, ModelConditional
using EpidemicTrajectories: EpidemicTrajectories

include("trajectory.jl")
include("conjugate.jl")

# The trajectory block + iFFBS kernel
export TrajectoryLatent, iffbs_kernel

# The conjugate-kernel convenience layer. The generic engine (`ConjugateGibbs`)
# lives in PracticalBayes; here we re-export it for convenience and add the
# epidemic surface on top of it.
using PracticalBayes: ConjugateGibbs
export ConjugateGibbs, @conjugate
export test_sensitivity_kernel, capture_prob_kernel, initial_state_kernel

end # module PracticalEpiBayes
