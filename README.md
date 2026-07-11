# PracticalEpiBayes.jl

**Placeholder — not yet implemented.**

The glue package that will automate and simplify integrating
[EpidemicTrajectories.jl](https://github.com/EvoArt/EpidemicTrajectories) (epidemic
simulators / likelihoods / iFFBS latent samplers) with
[PracticalBayes.jl](https://github.com/EvoArt/PracticalBayes) (the fast
Turing-like PPL with a first-class latent-kernel Gibbs seam).

Planned: reusable `TrajectoryLatent` distribution + `iFFBSKernel <:
AbstractLatentKernel` + conjugate-Gibbs kernels, so wiring an EpidemicTrajectories
model into a PracticalBayes `@model` is a couple of lines rather than the
hand-written boilerplate currently shown in
`EpidemicTrajectories/examples/cattle_ecoli_iffbs.jl`.
