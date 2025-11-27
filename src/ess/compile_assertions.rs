use crate::{
    FloatExt,
    chains::ChainBuffer,
    ess::{EnsemblSliceSampler, EnsembleSliceSamplerConfigTrait},
    log_likelihood::LogLikelihoodModel,
};

impl<
    T: FloatExt,
    CHAINS: ChainBuffer<T>,
    MODEL: LogLikelihoodModel<T>,
    CONFIG: EnsembleSliceSamplerConfigTrait,
> EnsemblSliceSampler<T, CHAINS, MODEL, CONFIG>
{
    const _RUNTIME_TIME_ASSERTION_PARAMETERS: () = assert!(
        MODEL::N_PARAMETERS == CONFIG::N_PARAMETERS,
        "CONFIG::N_PARAMETERS does not match MODEL::N_PARAMETERS"
    );

    const _RUNTIME_TIME_ASSERTION_DATA_DIMENS: () = assert!(
        MODEL::N_DATA_DIMENS == CONFIG::N_DATA_DIMENS,
        "CONFIG::N_DATA_DIMENS does not match MODEL::N_DATA_DIMENS"
    );
}
