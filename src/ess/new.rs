use crate::{
    chains::ChainBuffer, ess::EnsembleSliceSampler, log_likelihood::WrappedLogLikelihoodModel,
    moves::EnsembleMoveHandler, state::WalkerState,
};

impl<C: ChainBuffer, M: EnsembleMoveHandler, L: WrappedLogLikelihoodModel>
    EnsembleSliceSampler<C, M, L>
{
    pub fn new(
        max_n_steps: usize,
        n_walkers: usize,
        n_parameters: usize,
        chains: C,
        move_handler: M,
        model: L,
    ) -> Self {
        let required_walkers = (n_parameters * 2).max(3);
        assert!(n_walkers >= required_walkers);

        Self {
            max_n_steps,
            n_walkers,
            n_parameters,

            chains,
            move_handler,
            model,

            state_i: WalkerState::new(n_walkers, n_parameters),
            state_j: WalkerState::new(n_walkers, n_parameters),
        }
    }
}
