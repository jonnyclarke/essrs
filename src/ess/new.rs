use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::{
    FloatExt,
    chains::ChainBuffer,
    data::DataBuffer,
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
    pub fn new(data: DataBuffer<T, CONFIG>, chains: CHAINS, model: MODEL) -> Self {
        Self {
            data,
            chains,
            model,

            step: Array2::<T>::zeros((CONFIG::N_WALKERS, CONFIG::N_PARAMETERS)),
            next: Array2::<T>::zeros((CONFIG::N_WALKERS, CONFIG::N_PARAMETERS)),

            step_ll: Array1::<T>::zeros(CONFIG::N_WALKERS),
            next_ll: Array1::<T>::zeros(CONFIG::N_WALKERS),

            rng: rand::rng(),

            _config: PhantomData,
            _type: PhantomData,
        }
    }
}
