mod config;
mod initialise;
mod new;

use std::{marker::PhantomData, path::Path, time::Instant};

use rand::{self};
use tracing::info;

use crate::{
    chains::ChainBuffer, log_likelihood::LogLikelihoodModel, moves::MoveHandler, state::WalkerState,
};

pub trait EnsembleSliceSamplerConfigTrait {
    const MAX_N_STEPS: usize;
    const N_WALKERS: usize;
    const N_PARAMETERS: usize;
}

pub struct EnsembleSliceSamplerConfig<
    const MAX_N_STEPS: usize,
    const N_WALKERS: usize,
    const N_PARAMETERS: usize,
> {}

pub struct EnsemblSliceSampler<
    CHAINS: ChainBuffer,
    MODEL: LogLikelihoodModel,
    CONFIG: EnsembleSliceSamplerConfigTrait,
> {
    pub chains: CHAINS,
    pub model: MODEL,

    state_i: WalkerState,
    state_j: WalkerState,

    rng: rand::rngs::ThreadRng,

    _config: PhantomData<CONFIG>,
}

impl<CHAINS: ChainBuffer, MODEL: LogLikelihoodModel, CONFIG: EnsembleSliceSamplerConfigTrait>
    EnsemblSliceSampler<CHAINS, MODEL, CONFIG>
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

    pub fn run_sampler(&mut self, move_handler: MoveHandler) {
        let start_time = Instant::now();

        for i in 1..=CONFIG::MAX_N_STEPS {
            info!(iteration = i);
            move_handler.distribute_jump(
                &mut self.rng,
                &self.model,
                &self.state_i,
                &mut self.state_j,
            );
            self.accept_proposed_state(); // we accept new state before storing to avoid storing the initial conditions state...

            let sampler_state = &self.state_i.get_state_matrix().to_owned();
            self.chains.record_state(
                &self
                    .model
                    .columnar_transform_internal_to_physical(sampler_state),
            );

            let duration = start_time.elapsed();

            let projected = duration / (i as u32) * (CONFIG::MAX_N_STEPS as u32);

            println!(
                "Iteration {i} / {} :: TIME ELAPSED -- {} SEC [{} TOTAL]",
                CONFIG::MAX_N_STEPS,
                duration.as_secs(),
                projected.as_secs()
            );
            if i % 10 == 0 {
                self.print_quantile_summary();
            }
        }
    }

    pub fn display_parameter_summaries(&self) {
        self.chains.display_parameter_summaries();
    }

    pub fn save_log_likelihood<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        self.state_i.dump_final_log_likelihood(path)
    }

    pub fn print_quantile_summary(&self) {
        // Make a sorted copy
        let mut values = self.state_i.get_ll_vector().to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

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
    }
}
