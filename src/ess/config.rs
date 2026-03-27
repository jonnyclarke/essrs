use crate::ess::{EnsembleSliceSamplerConfig, EnsembleSliceSamplerConfigTrait};

impl<const MAX_N_STEPS: usize, const N_WALKERS: usize, const N_PARAMETERS: usize>
    EnsembleSliceSamplerConfig<MAX_N_STEPS, N_WALKERS, N_PARAMETERS>
{
    const _RUNTIME_TIME_ASSERTION: () = assert!(
        N_WALKERS >= (2 * N_PARAMETERS) && N_WALKERS >= 3,
        "Number of walkers must be >= max(3, [2 * n_parameters])"
    );
}

impl<const MAX_N_STEPS: usize, const N_WALKERS: usize, const N_PARAMETERS: usize>
    EnsembleSliceSamplerConfigTrait
    for EnsembleSliceSamplerConfig<MAX_N_STEPS, N_WALKERS, N_PARAMETERS>
{
    const MAX_N_STEPS: usize = MAX_N_STEPS;
    const N_WALKERS: usize = N_WALKERS;
    const N_PARAMETERS: usize = N_PARAMETERS;
}
