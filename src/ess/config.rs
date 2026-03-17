use crate::ess::{EnsembleSliceSamplerConfig, EnsembleSliceSamplerConfigTrait};

impl<
    const MAX_N_STEPS: usize,
    const N_WALKERS: usize,
    const N_PARAMETERS: usize,
    const N_BURN_IN: usize,
    const N_THIN_STRIDE: usize,
    const ENABLE_EARLY_STOPPING: bool,
>
    EnsembleSliceSamplerConfig<
        MAX_N_STEPS,
        N_WALKERS,
        N_PARAMETERS,
        N_BURN_IN,
        N_THIN_STRIDE,
        ENABLE_EARLY_STOPPING,
    >
{
    const _RUNTIME_TIME_ASSERTION: () = assert!(
        N_WALKERS >= (2 * N_PARAMETERS) && N_WALKERS >= 3,
        "Number of walkers must be >= max(3, [2 * n_parameters])"
    );
}

impl<
    const MAX_N_STEPS: usize,
    const N_WALKERS: usize,
    const N_PARAMETERS: usize,
    const N_BURN_IN: usize,
    const N_THIN_STRIDE: usize,
    const ENABLE_EARLY_STOPPING: bool,
> EnsembleSliceSamplerConfigTrait
    for EnsembleSliceSamplerConfig<
        MAX_N_STEPS,
        N_WALKERS,
        N_PARAMETERS,
        N_BURN_IN,
        N_THIN_STRIDE,
        ENABLE_EARLY_STOPPING,
    >
{
    const MAX_N_STEPS: usize = MAX_N_STEPS;
    const N_WALKERS: usize = N_WALKERS;
    const N_PARAMETERS: usize = N_PARAMETERS;
    const N_BURN_IN: usize = N_BURN_IN;
    const N_THIN_STRIDE: usize = N_THIN_STRIDE;
    const ENABLE_EARLY_STOPPING: bool = ENABLE_EARLY_STOPPING;
}
