//! TESTING

use crate::{
    chains::ChainBuffer, ess::EnsembleSliceSampler, log_likelihood::LogLikelihoodModel,
    moves::EnsembleMove,
};

// struct-types used to control which values have been set
pub struct Missing;
pub struct Set;

/// Builder for EnsembleSliceSampler.
///
/// C, M, L are the concrete component types.
/// CS, MS, LS track whether each component has been supplied.
pub struct EnsembleSliceSamplerBuilder<
    C = (),
    M = (),
    L = (),
    CS = Missing,
    MS = Missing,
    LS = Missing,
> {
    max_n_steps: usize,
    n_walkers: usize,
    n_parameters: usize,

    chains: Option<C>,
    move_handler: Option<M>,
    model: Option<L>,

    _states: std::marker::PhantomData<(CS, MS, LS)>,
}

/// BOWSER
impl EnsembleSliceSamplerBuilder<(), (), (), Missing, Missing, Missing> {
    /// TESTING
    pub fn new(n_steps: usize, n_walkers: usize, n_parameters: usize) -> Self {
        Self {
            max_n_steps: n_steps,
            n_walkers,
            n_parameters,

            chains: None,
            move_handler: None,
            model: None,

            _states: std::marker::PhantomData,
        }
    }
}

impl<M, L, MS, LS> EnsembleSliceSamplerBuilder<(), M, L, Missing, MS, LS> {
    pub fn chains<C: ChainBuffer>(
        self,
        chains: C,
    ) -> EnsembleSliceSamplerBuilder<C, M, L, Set, MS, LS> {
        assert_eq!(self.max_n_steps, chains.get_max_n_steps());
        assert_eq!(self.n_walkers, chains.get_n_walkers());
        assert_eq!(self.n_parameters, chains.get_n_parameters());

        EnsembleSliceSamplerBuilder {
            max_n_steps: self.max_n_steps,
            n_walkers: self.n_walkers,
            n_parameters: self.n_parameters,

            chains: Some(chains),
            move_handler: self.move_handler,
            model: self.model,

            _states: std::marker::PhantomData,
        }
    }
}

impl<C, L, CS, LS> EnsembleSliceSamplerBuilder<C, (), L, CS, Missing, LS> {
    pub fn move_handler<M: EnsembleMove>(
        self,
        move_handler: M,
    ) -> EnsembleSliceSamplerBuilder<C, M, L, CS, Set, LS> {
        EnsembleSliceSamplerBuilder {
            max_n_steps: self.max_n_steps,
            n_walkers: self.n_walkers,
            n_parameters: self.n_parameters,

            chains: self.chains,
            move_handler: Some(move_handler),
            model: self.model,

            _states: std::marker::PhantomData,
        }
    }
}

impl<C, M, CS, MS> EnsembleSliceSamplerBuilder<C, M, (), CS, MS, Missing> {
    pub fn model<L: LogLikelihoodModel>(
        self,
        model: L,
    ) -> EnsembleSliceSamplerBuilder<C, M, L, CS, MS, Set> {
        assert_eq!(self.n_parameters, model.get_n_parameters());

        EnsembleSliceSamplerBuilder {
            max_n_steps: self.max_n_steps,
            n_walkers: self.n_walkers,
            n_parameters: self.n_parameters,

            chains: self.chains,
            move_handler: self.move_handler,
            model: Some(model),

            _states: std::marker::PhantomData,
        }
    }
}

impl<C, M, L> EnsembleSliceSamplerBuilder<C, M, L, Set, Set, Set>
where
    C: ChainBuffer,
    M: EnsembleMove,
    L: LogLikelihoodModel,
{
    pub fn build(
        self,
        // state: WalkerState,
    ) -> EnsembleSliceSampler<C, M, L> {
        EnsembleSliceSampler::new(
            self.max_n_steps,
            self.n_walkers,
            self.n_parameters,
            self.chains.expect("builder state invariant"),
            self.move_handler.expect("builder state invariant"),
            self.model.expect("builder state invariant"),
        )
    }
}

#[cfg(test)]
mod tests_ess_builder {

    use super::*;
    use crate::{
        chains::static_buffer::StaticBuffer,
        log_likelihood::toy::rosenbrock::AnalyticRosenbrockToyLogL,
        moves::differential_move::DifferentialMove,
    };

    #[test]
    fn test_build() -> anyhow::Result<()> {
        let ess_builder = EnsembleSliceSamplerBuilder::new(10, 4, 2);

        let model = AnalyticRosenbrockToyLogL::new(1.0, 3.0);

        let chains = StaticBuffer::new(10, 4, 2);

        let move_handler = DifferentialMove::new();

        let _ess = ess_builder
            .model(model)
            .chains(chains)
            .move_handler(move_handler)
            .build();

        Ok(())
    }
}
