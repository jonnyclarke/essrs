mod compile_assertions;
mod config;
mod initialise;
mod new;
mod random;

use std::marker::PhantomData;

use ndarray::{Array1, Array2}; // ← THIS imports all arithmetic traits
use ndarray::{ArrayBase, prelude::*};
use rand::{self, Rng};
use tracing::info;

use crate::{FloatExt, chains::ChainBuffer, data::DataBuffer, log_likelihood::LogLikelihoodModel};

pub trait EnsembleSliceSamplerConfigTrait {
    const MAX_N_STEPS: usize;
    const N_WALKERS: usize;
    const N_PARAMETERS: usize;
    const N_DATA_DIMENS: usize;
    const N_BURN_IN: usize;
    const N_THIN_STRIDE: usize;
    const ENABLE_EARLY_STOPPING: bool;
}

pub struct EnsembleSliceSamplerConfig<
    const MAX_N_STEPS: usize,
    const N_WALKERS: usize,
    const N_PARAMETERS: usize,
    const N_DATA_DIMENS: usize,
    const N_BURN_IN: usize,
    const N_THIN_STRIDE: usize,
    const ENABLE_EARLY_STOPPING: bool,
> {}

pub struct EnsemblSliceSampler<
    T: FloatExt,
    CHAINS: ChainBuffer<T>,
    MODEL: LogLikelihoodModel<T>,
    CONFIG: EnsembleSliceSamplerConfigTrait,
> {
    data: DataBuffer<T, CONFIG>,
    chains: CHAINS,
    model: MODEL,

    step: Array2<T>,
    next: Array2<T>,

    step_ll: Array1<T>,
    next_ll: Array1<T>,

    rng: rand::rngs::ThreadRng,

    _config: PhantomData<CONFIG>,
    _type: PhantomData<T>,
}

impl<
    T: FloatExt,
    CHAINS: ChainBuffer<T>,
    MODEL: LogLikelihoodModel<T>,
    CONFIG: EnsembleSliceSamplerConfigTrait,
> EnsemblSliceSampler<T, CHAINS, MODEL, CONFIG>
{
    fn get_likelihood_floor(&mut self, i: usize) -> T {
        self.step_ll[i]
            + self
                .rng
                .random_range((T::my_min_positive())..T::from(1.0).unwrap())
                .ln()
    }

    fn get_vector(&self, l: usize, r: usize) -> Array1<T> {
        &self.step.row(l) - &self.step.row(r)
    }

    fn step_out(&self, anchor_vec: &Array1<T>, direction_vec: &Array1<T>, ll_floor: T) -> (T, T) {
        let mut nl: T = T::get_neg_one();
        let mut nr: T = T::get_one();
        let one: T = T::get_one();

        // Left
        let params_l = anchor_vec + &(direction_vec * nl);
        let mut log_l_l = self.model.log_likelihood(&self.data, &params_l);

        while log_l_l > ll_floor {
            nl -= one;
            let params_l = anchor_vec + &(direction_vec * nl);
            log_l_l = self.model.log_likelihood(&self.data, &params_l);
        }

        // Right
        let params_r = anchor_vec + &(direction_vec * nr);
        let mut log_l_r = self.model.log_likelihood(&self.data, &params_r);

        while log_l_r > ll_floor {
            nr += one;
            let params_r = anchor_vec + &(direction_vec * nr);
            log_l_r = self.model.log_likelihood(&self.data, &params_r);
        }

        (nl, nr)
    }

    fn step_in(
        &mut self,
        anchor_vec: &Array1<T>,
        direction_vec: &Array1<T>,
        mut nl: T,
        mut nr: T,
        ll_floor: T,
    ) -> (T, Array1<T>) {
        let zero: T = T::get_zero();
        let mut shift = self.rng.random_range(nl..nr);
        let mut params = anchor_vec + &(direction_vec * shift);
        let mut ll = self.model.log_likelihood(&self.data, &params);

        while ll < ll_floor {
            if shift < zero {
                nl = shift
            } else {
                nr = shift
            }

            shift = self.rng.random_range(nl..nr);
            params = anchor_vec + &(direction_vec * shift);
            ll = self.model.log_likelihood(&self.data, &params);
        }

        (ll, params)
    }

    pub fn jump(&mut self, _iteration: usize) {
        for i in 0..CONFIG::N_WALKERS {
            let current: &ArrayBase<ndarray::OwnedRepr<T>, Dim<[usize; 1]>, T> =
                &self.step.row(i).to_owned();

            let l: usize = self.get_rn_not(i);
            let r: usize = self.get_rn_not_or(i, l);

            let ll_floor: T = self.get_likelihood_floor(i);

            let direction: Array1<T> = self.get_vector(l, r);

            let (nl, nr) = self.step_out(current, &direction, ll_floor);

            let (ll, accepted) = self.step_in(current, &direction, nl, nr, ll_floor);

            let mut nxt = self.next.row_mut(i);
            nxt.assign(&accepted);

            self.next_ll[i] = ll;

            // if iteration % 10 == 0 {
            //     println!("WALKER: {} :: Log-likelihood = {} :: Parameters = [{}, {}]", i+1, ll, accepted[0], accepted[1].exp());
            // }
        }
    }

    pub fn accept_proposals(&mut self) {
        self.step.assign(&self.next);
        self.step_ll.assign(&self.next_ll);
    }

    pub fn run_sampler(&mut self) {
        // initial jump to avoid storing chain initialisation
        // println!("INITIALISING");
        self.jump(0);
        self.accept_proposals();

        for i in 1..=CONFIG::MAX_N_STEPS {
            // if i % 10 == 0 {println!("\n\nIteration {i} / {N_STEPS}\n");}
            info!(iteration = i);
            self.jump(i);
            self.chains.record_state(&self.step);
            self.accept_proposals();
            // if i==5 {panic!("FORCE PANIC");}
        }
    }
}
