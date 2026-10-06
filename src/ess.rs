mod initialise;
mod new;

use std::path::Path;

use crate::{
    chains::ChainBuffer, log_likelihood::WrappedLogLikelihoodModel, moves::EnsembleMoveHandler,
    state::WalkerState,
};

pub struct EnsembleSliceSampler<
    C: ChainBuffer,
    M: EnsembleMoveHandler,
    L: WrappedLogLikelihoodModel,
> {
    max_n_steps: usize,
    n_walkers: usize,
    n_parameters: usize,

    pub chains: C,
    pub move_handler: M,
    pub model: L,

    state_i: WalkerState,
    state_j: WalkerState,
}

impl<C: ChainBuffer, M: EnsembleMoveHandler, L: WrappedLogLikelihoodModel>
    EnsembleSliceSampler<C, M, L>
{
    pub fn accept_proposed_state(&mut self) {
        // let mut state0 = self.state_i.get_mut_state_matrix();
        self.state_i
            .get_mut_state_matrix()
            .assign(&self.state_j.get_state_matrix());

        // let mut ll0 = self.state_i.get_mut_ll_vector();
        self.state_i
            .get_mut_ll_vector()
            .assign(&self.state_j.get_mut_ll_vector());
    }

    pub fn run_sampler(&mut self, n_burn_in: usize) -> anyhow::Result<()> {
        // we allow burn in to eliminate effect on chains of the starting point
        for iteration in 1..=n_burn_in {
            self.move_handler.distribute_jump(
                iteration,
                &self.model,
                &self.state_i,
                &mut self.state_j,
            )?;
            self.accept_proposed_state(); // we accept new state before storing to avoid storing the initial conditions state...
        }

        for iteration in 1..=self.max_n_steps {
            self.move_handler.distribute_jump(
                iteration,
                &self.model,
                &self.state_i,
                &mut self.state_j,
            )?;
            self.accept_proposed_state(); // we accept new state before storing to avoid storing the initial conditions state...

            let internal = self.state_i.get_state_matrix();
            let mut physical = internal.to_owned();

            self.model
                .columnar_transform_internal_to_physical(internal, physical.view_mut())?;
            self.chains.record_state(&physical);
        }

        Ok(())
    }

    pub fn display_parameter_summaries(&self) {
        self.chains
            .display_parameter_summaries(self.n_parameters as i32);
    }

    pub fn save_log_likelihood<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        self.state_i.dump_final_log_likelihood(path)
    }

    // this function should be removed from implementation and offered as extension
    pub fn print_quantile_summary(&self) -> anyhow::Result<()> {
        // Make a sorted copy
        let mut values = self.state_i.get_ll_vector().to_vec();
        values.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("sorting of log-likelihood vector has failed")
        });

        let n = values.len();

        let p25 = n / 4;
        let p50 = n / 2;
        let p75 = 3 * n / 4;

        for (i, v) in values.iter().enumerate() {
            if i == 0 {
                println!("-- MIN: {}", v)
            }
            if i == p25 {
                println!("-- p25: {}", v)
            }
            if i == p50 {
                println!("-- p50: {}", v)
            }
            if i == p75 {
                println!("-- p75: {}", v)
            }
            if i == n - 1 {
                println!("-- MAX: {}\n", v)
            }
        }
        Ok(())
    }
}
