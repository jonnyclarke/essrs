use std::marker::PhantomData;

use crate::{
    chains::ChainBuffer,
    ess::{EnsembleSliceSampler, EnsembleSliceSamplerConfigTrait},
    log_likelihood::LogLikelihoodModel,
    state::WalkerState,
};

impl<CHAINS: ChainBuffer, MODEL: LogLikelihoodModel, CONFIG: EnsembleSliceSamplerConfigTrait>
    EnsembleSliceSampler<CHAINS, MODEL, CONFIG>
{
    pub fn new(chains: CHAINS, model: MODEL) -> Self {
        let nw = CONFIG::N_WALKERS;
        let np = CONFIG::N_PARAMETERS;
        let required_walkers = (np * 2).max(3);
        assert!(nw >= required_walkers);

        Self {
            chains,
            model,

            state_i: WalkerState::new(CONFIG::N_WALKERS, CONFIG::N_PARAMETERS),
            state_j: WalkerState::new(CONFIG::N_WALKERS, CONFIG::N_PARAMETERS),

            rng: rand::rng(),

            _config: PhantomData,
        }
    }
}
