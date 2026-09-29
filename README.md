# essrs

[![CI](https://github.com/jonnyclarke/essrs/actions/workflows/coverage_test_unit.yml/badge.svg)]
[![CI](https://github.com/jonnyclarke/essrs/actions/workflows/integration_testing.yml/badge.svg)]
[![CI](https://github.com/jonnyclarke/essrs/actions/workflows/clippy.yml/badge.svg)]
[![CI](https://github.com/jonnyclarke/essrs/actions/workflows/rustfmt.yml/badge.svg)]
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]

A lightweight implementation of Markov Chain Monte Carlo Ensemble Sampling.

## Summary

Vanilla metropolis-hastings MCMC methods suffer from the following issues:
- slow exploration in correlated spaces
- inefficient exploration due to hyper-parameter tuning requirements.

Ensemble sampling mitigates these problems by running many walkers simultaneously [the ensemble] to leverage the snapshot of the posterior probability distribution.
- Moves account for the structure of the posterior thereby enabling much more rapid exploration of highly correlated spaces.
- Since move sampling is based on the current distribution the system auto scales step lengths without any heuristics required on top.

## Moves [expandable]

### Differential Move

Differential move samples a direction vector by randomly sampling two other walkers and using the direction vector between them to define the expansion direction.
The move then occurs in two phases.
We `step out` by expanding the search area in both directions until the log-likelihood has fallen below the sampling threshold.
This is followed by a `step in` where a point within the region is sampled until a valid point is found.

This naturally handles high correlation since the shape of the ensemble will eventually mirror that of the posterior.

Convergence from a poor initial guess can be slow due to walkers being left behind in low posterior regions and taking small steps to reach the equilibrium region.

NOTE: this move guarantees that all walkers move in each iteration.

### Gaussian Move

This move type should be considered a `global` move in that there is no limitation on where the jump can be proposed throughout parameter space. [This is different to differential move which is limited to a finite set of vectors defined by the ensemble.]
For this we take the current distribution of walker positions and compute the covariance matrix. We then sample from that covariance matrix (thereby accounting for highly correlated regions and automatic step length refinement) to propose the jump.
The proposal is then sampled to satisfy detailed balance.

This method naturally improves with number of walkers since then there is a better representation of the posterior. Too few and one might face quite high rejection rates and slow convergence.

NOTE: in this scheme moves can be rejected as we only propose once for each walker and accept or reject. We do not then resample on rejection.

### Complimentary methods

Empirically it works well to mix these moves since gaussian move can mitigate the effect of walkers getting stuck in low posterior regions and differential move is designed to always update parameters.

## Caveats

- The differential move scheme contains implicit requirements on the number of walkers that must be used. This limit is given as
```
MAX(3, 2 * n_dimensions)
```
where `n_dimensions` is the number of parameters being optimised.
The minimum number of 3 is required because, given the i'th walker, the algorithm always requires at least two more positions to define the direction vector along which we jump. Thus 3 is the minimum number of walkers that can be used.
The factor `2 * n_dimensions` is required to ensure that the walkers do not get stuck in a dimensional sub-space which would leave them unable to fully explore the full parameter space.

## Usage

This package is currently only available as a github repo.

To add to a project you can use:
```
[dependencies]
essrs = { git = "https://github.com/jonnyclarke/essrs" }
```

## Examples

Please see examples in dedicated directory.

Current example:
- Script to fit the mean and standard deviation of a set of points taking into account the errors on those points.

```bash
cargo run --example gaussian_1d_fit
cargo run --example gaussian_1d_fit --release
```

```

## Limitations / Status
- This project is currently only available as a github repo and has no official release. It is not currently suitable for production grade implementations.
- Current implementation is locked to f64. Generalising to f32 may require extensive refactoring and is not something I am currently looking to do.

## Future enhancements

The following is a non-exhaustive list of future enhancements I would like to make to the repository.
- Publishing on crates.io for better accesibility.
- Python bindings for pre-defined log-likelihood functions for easy use.
- Extended library of python bindings including: two-dimensional normal distribution fit with errors, GMM of two-dimensional normal distribution with errors.
- Implementation of auto-correlation computation with auto-stop capabilities.


## License

This project is licensed under the MIT License.