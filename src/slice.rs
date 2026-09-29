//! This script contains the step in and step out functions used for ensemble slice sampling
use ndarray::{Array1, ArrayView1};
use rand::{Rng, RngCore};

use crate::{
    log_likelihood::LogLikelihoodModel
};

/// Structure storing information on the likelihood boundaries following the step-out part of the algorithm.
/// This will be adjusted during step-in until a valid point is found.
pub struct LikelihoodBounds {
    nl: f64,
    nr: f64,
    _ll_l: f64,
    _ll_r: f64,
}

pub fn step_out(
    model: &dyn LogLikelihoodModel,
    anchor_vec: ArrayView1<f64>,
    direction_vec: ArrayView1<f64>,
    ll_floor: f64,
) -> LikelihoodBounds {
    let mut nl = -1.0;
    let mut nr = 1.0;

    // Left
    let mut params = &anchor_vec + &direction_vec * nl;
    let mut ll_l = model.internal_log_likelihood(params.view());

    while ll_l > ll_floor {
        nl *= 2.0; // multiple by 2 to expand faster as the fall-in is also binary-tree reduction
        params.assign(&(&anchor_vec + &direction_vec * nl));
        ll_l = model.internal_log_likelihood(params.view());
    }

    // Right
    params.assign(&(&anchor_vec + &direction_vec * nr));
    let mut ll_r = model.internal_log_likelihood(params.view());

    while ll_r > ll_floor {
        nr *= 2.0;
        params.assign(&(&anchor_vec + &direction_vec * nr));
        ll_r = model.internal_log_likelihood(params.view());
    }

    LikelihoodBounds {
        nl,
        nr,
        _ll_l: ll_l,
        _ll_r: ll_r,
    }
}

pub fn step_in(
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
        let ll = model.internal_log_likelihood(params.view());

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