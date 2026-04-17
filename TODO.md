# To Do
A list of things to do.

## Docs
- [ ] Add figures to readme
- [ ] Add non-sea ice data in data section of readme

## Code
- [ ] Clean up `icepinn.network.ForcedAdvectionDiffusion._sobolev_regularization_terms`
- [ ] Clean up `icepinn.network.ModelV2.K`, `_K_impl`, and `_K_core`

## Theory/Implementation
- [ ] Add synthetic data test cases; roughly ordered by difficulty,
    - [ ] Homogeneous advection
    - [ ] Homogeneous diffusion
    - [ ] Inhomogeneous advection
    - [ ] Inhomogeneous diffusion
    - [ ] Homogeneous advection-diffusion
    - [ ] Imhomogenous advection-diffusion
- [ ] Add bigger context for model in (future) `ModelV3`
    - Deeper history context (i.e., three frames instead of one frame)
    - Surface wind data
    - Surface temperature data
- [ ] Consider regularization suitable for stochastic/highly non-convex loss
