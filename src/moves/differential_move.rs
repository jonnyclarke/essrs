use ndarray::{Array1, ArrayView1};
use rand::{Rng, RngCore};

use crate::{
    log_likelihood::LogLikelihoodModel,
    moves::EnsembleMove,
    random::{get_rn_not, get_rn_not_or},
    state::WalkerState,
};

struct LikelihoodBounds {
    nl: f64,
    nr: f64,
    _ll_l: f64,
    _ll_r: f64,
}

pub struct DifferentialMove {}
impl DifferentialMove {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for DifferentialMove {
    fn default() -> Self {
        Self::new()
    }
}

impl DifferentialMove {
    fn get_vector(&self, state: &WalkerState, l: usize, r: usize) -> Array1<f64> {
        &state.get_ith_state_vector(l) - &state.get_ith_state_vector(r)
    }

    fn step_out(
        &self,
        model: &dyn LogLikelihoodModel,
        anchor_vec: ArrayView1<f64>,
        direction_vec: ArrayView1<f64>,
        ll_floor: f64,
    ) -> LikelihoodBounds {
        let mut nl = -1.0;
        let mut nr = 1.0;

        // Left
        let mut params = &anchor_vec + &direction_vec * nl;
        let mut ll_l = model.log_likelihood(params.view());

        while ll_l > ll_floor {
            nl *= 2.0; // multiple by 2 to expand faster as the fall-in is also binary-tree reduction
            params.assign(&(&anchor_vec + &direction_vec * nl));
            ll_l = model.log_likelihood(params.view());
        }

        // Right
        params.assign(&(&anchor_vec + &direction_vec * nr));
        let mut ll_r = model.log_likelihood(params.view());

        while ll_r > ll_floor {
            nr *= 2.0;
            params.assign(&(&anchor_vec + &direction_vec * nr));
            ll_r = model.log_likelihood(params.view());
        }

        LikelihoodBounds {
            nl,
            nr,
            _ll_l: ll_l,
            _ll_r: ll_r,
        }
    }

    fn step_in(
        &self,
        rng: &mut dyn RngCore,
        model: &dyn LogLikelihoodModel,
        anchor_vec: ArrayView1<f64>,
        direction_vec: ArrayView1<f64>,
        search_bounds: LikelihoodBounds,
        ll_floor: f64,
    ) -> (f64, Array1<f64>) {
        let mut nl = search_bounds.nl;
        let mut nr = search_bounds.nr;

        let mut shift = rng.random_range(nl..nr);

        let mut params = Array1::zeros(anchor_vec.len());

        loop {
            params.assign(&(&anchor_vec + &direction_vec * shift));
            let ll = model.log_likelihood(params.view());

            if ll >= ll_floor {
                return (ll, params);
            }

            if shift < 0.0 {
                nl = shift
            } else {
                nr = shift
            }

            shift = rng.random_range(nl..nr);
        }
    }
}

impl EnsembleMove for DifferentialMove {
    fn jump(
        &self,
        rng: &mut dyn RngCore,
        log_likelihood_model: &dyn LogLikelihoodModel,
        state_i: &WalkerState,
        state_j: &mut WalkerState,
    ) {
        let n_walkers = state_i.n_walkers();

        for i in 0..n_walkers {
            let l = get_rn_not(rng, n_walkers, i);
            let r = get_rn_not_or(rng, n_walkers, i, l);
            let ll_floor = state_i.get_ith_ll(i) + self.get_likelihood_floor(rng);

            let current = state_i.get_ith_state_vector(i).to_owned();
            let direction = self.get_vector(state_i, l, r);

            let search_bounds = self.step_out(
                log_likelihood_model,
                current.view(),
                direction.view(),
                ll_floor,
            );

            let (ll, accepted) = self.step_in(
                rng,
                log_likelihood_model,
                current.view(),
                direction.view(),
                search_bounds,
                ll_floor,
            );

            state_j.get_mut_ith_state_vector(i).assign(&accepted);
            *state_j.get_mut_ith_ll(i) = ll;
        }
    }
}
