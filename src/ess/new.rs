use crate::{
    chains::ChainBuffer, ess::EnsembleSliceSampler, log_likelihood::LogLikelihoodModel,
    state::WalkerState,
};

impl<C: ChainBuffer, L: LogLikelihoodModel> EnsembleSliceSampler<C, L> {
    pub fn new(
        max_n_steps: usize,
        n_walkers: usize,
        n_parameters: usize,
        chains: C,
        model: L,
    ) -> Self {
        let required_walkers = (n_parameters * 2).max(3);
        assert!(n_walkers >= required_walkers);

        Self {
            max_n_steps,
            n_walkers,
            n_parameters,

            chains,
            model,

            state_i: WalkerState::new(n_walkers, n_parameters),
            state_j: WalkerState::new(n_walkers, n_parameters),

            rng: rand::rng(),
        }
    }
}
