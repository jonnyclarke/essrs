//! This script contains the step in and step out functions used for ensemble slice sampling
use ndarray::{ArrayView1, ArrayViewMut1, Zip};
use rand::{Rng, RngCore};
use thiserror::Error;

use crate::log_likelihood::{LogLikelihoodError, LogLikelihoodModel, WrappedLogLikelihoodModel};

#[derive(Debug, Error)]
pub enum SliceSamplerError {
    #[error("failed stepping out left: {source}")]
    StepOutLeftError {
        #[source]
        source: LogLikelihoodError,
    },

    #[error("failed stepping out right: {source}")]
    StepOutRightError {
        #[source]
        source: LogLikelihoodError,
    },

    #[error("failed stepping in: {source}")]
    StepInError {
        #[source]
        source: LogLikelihoodError,
    },
}
/// Structure storing information on the likelihood boundaries following the step-out part of the algorithm.
/// This will be adjusted during step-in until a valid point is found.
pub struct LikelihoodBounds {
    nl: f64,
    nr: f64,
    ll_floor: f64,
    _ll_l: f64,
    _ll_r: f64,
}

const N_MAX_STEP_OUT_ITERATIONS: usize = 20;

pub fn step_out(
    model: &dyn LogLikelihoodModel,
    anchor_vec: ArrayView1<f64>,
    direction_vec: ArrayView1<f64>,
    internal: &mut ArrayViewMut1<f64>, // specifically to avoid many array allocations
    physical: &mut ArrayViewMut1<f64>,
    ll_floor: f64,
) -> Result<LikelihoodBounds, SliceSamplerError> {
    let mut nl = -1.0;
    let mut nr = 1.0;

    // Left
    // let mut params = &anchor_vec + &direction_vec * nl;
    Zip::from(&mut *internal)
        .and(&anchor_vec)
        .and(&direction_vec)
        .for_each(|out, &anchor, &direction| {
            *out = anchor + direction * nl;
        });

    let mut ll_l = model
        .wrapped_log_likelihood(internal.view(), physical)
        .map_err(|source| SliceSamplerError::StepOutLeftError { source })?;

    let mut il: usize = 1;

    while ll_l > ll_floor && il <= N_MAX_STEP_OUT_ITERATIONS {
        nl *= 2.0; // multiple by 2 to expand faster as the fall-in is also binary-tree reduction

        Zip::from(&mut *internal)
            .and(&anchor_vec)
            .and(&direction_vec)
            .for_each(|out, &anchor, &direction| {
                *out = anchor + direction * nl;
            });

        ll_l = model
            .wrapped_log_likelihood(internal.view(), physical)
            .map_err(|source| SliceSamplerError::StepOutLeftError { source })?;
        il += 1;
    }

    // Right
    Zip::from(&mut *internal)
        .and(&anchor_vec)
        .and(&direction_vec)
        .for_each(|out, &anchor, &direction| {
            *out = anchor + direction * nl;
        });

    let mut ll_r = model
        .wrapped_log_likelihood(internal.view(), physical)
        .map_err(|source| SliceSamplerError::StepOutRightError { source })?;

    let mut ir: usize = 1;

    while ll_r > ll_floor && ir <= N_MAX_STEP_OUT_ITERATIONS {
        nr *= 2.0;

        Zip::from(&mut *internal)
            .and(&anchor_vec)
            .and(&direction_vec)
            .for_each(|out, &anchor, &direction| {
                *out = anchor + direction * nl;
            });

        ll_r = model
            .wrapped_log_likelihood(internal.view(), physical)
            .map_err(|source| SliceSamplerError::StepOutRightError { source })?;
        ir += 1;
    }

    Ok(LikelihoodBounds {
        nl,
        nr,
        ll_floor,
        _ll_l: ll_l,
        _ll_r: ll_r,
    })
}

pub fn step_in(
    rng: &mut dyn RngCore,
    model: &dyn LogLikelihoodModel,
    anchor_vec: ArrayView1<f64>,
    direction_vec: ArrayView1<f64>,
    search_bounds: LikelihoodBounds,
    internal: &mut ArrayViewMut1<f64>, // specifically to avoid many array allocations
    physical: &mut ArrayViewMut1<f64>,
) -> Result<f64, SliceSamplerError> {
    let mut nl = search_bounds.nl;
    let mut nr = search_bounds.nr;

    let mut shift = rng.random_range(nl..nr);

    loop {
        Zip::from(&mut *internal)
            .and(&anchor_vec)
            .and(&direction_vec)
            .for_each(|out, &anchor, &direction| {
                *out = anchor + direction * shift;
            });

        let ll = model
            .wrapped_log_likelihood(internal.view(), physical)
            .map_err(|source| SliceSamplerError::StepInError { source })?;

        if ll >= search_bounds.ll_floor {
            return Ok(ll);
        }

        if shift < 0.0 {
            nl = shift
        } else {
            nr = shift
        }

        shift = rng.random_range(nl..nr);
    }
}
