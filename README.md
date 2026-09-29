# essrs [Ensemble Slice Sampling (in Rust)]

[![CI](https://github.com/jonnyclarke/essrs/actions/workflows/coverage_test_unit.yml/badge.svg)]
[![CI](https://github.com/jonnyclarke/essrs/actions/workflows/integration_testing.yml/badge.svg)]
[![CI](https://github.com/jonnyclarke/essrs/actions/workflows/clippy.yml/badge.svg)]
[![CI](https://github.com/jonnyclarke/essrs/actions/workflows/rustfmt.yml/badge.svg)]
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]

A lightweight implementation of Markov Chain Monte Carlo Ensemble Slice Sampling.



## Summary (and a little history)

There are by now multiple approaches for MCMC sampling:
- metropolis hastings
- gibbs sampling
- no-U-turn sampling
- ensemble slice sampling
each of which has pros and cons which I will not go into exhaustively here..

The simplest, metropolis-hastings, can be as simple as a single chain where, at state i, a new point, state j, is sampled from `epsilon * N(0, 1)` in n-dimensional space. This point is accepted based upon the detailed balance criterion (here I refer you to google) and the chain continues. While incredibly simple to implement there is an inherent trade off.
Larger values of epsilon mean the jumps are larger and the space can be explored more efficiently however, in highly correlated spaces, the proposal will often fall on a low likelihood state causing a high rejection rate. While the value of `epsilon` can be tuned on the fly the issue of the tradeoff remains.

Ensemble slice sampling mitigates these problems with the caveat that it requires many chains running together (the ensemble) to work effectively. The slice sampler terminology comes from the concept that, for each walker, a direction is chosen and the algorithm explores a slice of parameter space. This is characterised by a step-out / step-in approach. First, centred on state i the algorithm steps out in direction `d` until the likelihood of the point falls below a threshold. We then repeat and step out in the negative direction vector `-d`.
At the end we have an interval (the slice) given by `[x_i - l * d, x_i + r * d]` where `l` and `r` are the number of steps in the so-called left and right directions respectively.
The next stage is the step in process where, within the interval, we select a random point. If that point is still below the threshold we contract the interval and sample again. If the likelihood is above the threshold the point is accepted. This algorithm has the notable benefit that the jump to a new state is always accepted.
A key element of this algorithm is thus computing a direction vector. This could be sampled randomly however that approach would also suffer from not respecting the underlying covariance structure of the posterior and would result in inefficient exploration.
We have currently implemented two moves:
- differential move
- gaussian move
which are explained below.
Both of these methods use the current ensemble state to derive direction vectors which automatically:
- respects the covariance structure once converged
- removes the need for fine-tuning parameters since all relevant scale lengths are already baked into the estimate of the posterior distribution from which we are sampling.
This naturally allows for efficient exploration of highly correlated parameter spaces at the cost of requiring many walkers in parallel. A further immediate benefit is that, since we are expanding and then contracting an interval, the state always changes from one iteration to the next.

## Moves

### Differential Move

Differential move samples a direction vector by randomly sampling two other walkers and using the direction vector between them to define the expansion direction.
The move then occurs in two phases.
We `step out` by expanding the search area in both directions until the log-likelihood has fallen below the sampling threshold.
This is followed by a `step in` where a point within the region is sampled until a valid point is found.

This naturally handles high correlation since the shape of the ensemble will eventually mirror that of the posterior.

Convergence from a poor initial guess can be slow due to walkers being left behind in low posterior regions and taking small steps to reach the equilibrium region.

### Gaussian Move

The Gaussian move takes state i and estimates covariance of the walker distribution.
We then sample from this n-dimensional covariance, centering on the origin, which provides the direction vector `d`.
This approach thus does not limit the possible directions in the way differential move does.
Additionally, since we use a global covariance, walkers which are stranded in low likelihood regions have better chance to converge into the main ensemble as the fact of their isolation will influence the covariance matrix from which a direction is sampled.

### Complimentary methods

Empirically it works well to mix these moves since gaussian move can mitigate the effect of walkers getting stuck in low posterior regions and differential move is designed to always update parameters.

### Caveats

- The differential move scheme contains implicit requirements on the number of walkers that must be used. This limit is given as
```
MAX(3, 2 * n_dimensions)
```
where `n_dimensions` is the number of parameters being optimised.
The minimum number of 3 is required because, given the i'th walker, the algorithm always requires at least two more positions to define the direction vector along which we jump. Thus 3 is the minimum number of walkers that can be used.
The factor `2 * n_dimensions` is required to ensure that the walkers do not get stuck in a dimensional sub-space which would leave them unable to fully explore the full parameter space.

## Package Usage

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