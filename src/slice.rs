//! This script contains the step in and step out functions used for ensemble slice sampling
use ndarray::{ArrayView1, ArrayViewMut1, Zip};
use rand::{Rng, RngCore};
use thiserror::Error;

use crate::log_likelihood::{LogLikelihoodError, LogLikelihoodModel, WrappedLogLikelihoodModel};

#[derive(Debug, Error)]
pub(crate) enum SliceSamplerError {
    #[error("failed stepping out left: {source}")]
    LeftStep {
        #[source]
        source: LogLikelihoodError,
    },

    #[error("failed stepping out right: {source}")]
    RightStep {
        #[source]
        source: LogLikelihoodError,
    },

    #[error("failed stepping in: {source}")]
    StepIn {
        #[source]
        source: LogLikelihoodError,
    },
}
/// Structure storing information on the likelihood boundaries following the step-out part of the algorithm.
/// This will be adjusted during step-in until a valid point is found.
pub(crate) struct LikelihoodBounds {
    nl: f64,
    nr: f64,
    ll_floor: f64,
    _ll_l: f64,
    _ll_r: f64,
}

const N_MAX_STEP_OUT_ITERATIONS: usize = 20;

fn parameter_proposal(
    internal: &mut ArrayViewMut1<f64>,
    anchor: ArrayView1<f64>,
    direction: ArrayView1<f64>,
    n: f64,
) {
    Zip::from(internal)
        .and(&anchor)
        .and(&direction)
        .for_each(|out, &anchor, &direction| {
            *out = anchor + direction * n;
        });
}

pub(crate) fn step_out<L: LogLikelihoodModel + ?Sized>(
    model: &L,
    anchor_vec: ArrayView1<f64>,
    direction_vec: ArrayView1<f64>,
    internal: &mut ArrayViewMut1<f64>, // specifically to avoid many array allocations
    physical: &mut ArrayViewMut1<f64>,
    ll_floor: f64,
) -> Result<LikelihoodBounds, SliceSamplerError> {
    let mut nl = -1.0;
    let mut nr = 1.0;

    // Left
    parameter_proposal(internal, anchor_vec, direction_vec, nl);

    let mut ll_l = model
        .wrapped_log_likelihood(internal.view(), physical)
        .map_err(|source| SliceSamplerError::LeftStep { source })?;

    let mut il: usize = 1;

    while ll_l > ll_floor && il <= N_MAX_STEP_OUT_ITERATIONS {
        nl *= 2.0; // multiple by 2 to expand faster as the fall-in is also binary-tree reduction

        parameter_proposal(internal, anchor_vec, direction_vec, nl);

        ll_l = model
            .wrapped_log_likelihood(internal.view(), physical)
            .map_err(|source| SliceSamplerError::LeftStep { source })?;
        il += 1;
    }

    // Right
    parameter_proposal(internal, anchor_vec, direction_vec, nr);

    let mut ll_r = model
        .wrapped_log_likelihood(internal.view(), physical)
        .map_err(|source| SliceSamplerError::RightStep { source })?;

    let mut ir: usize = 1;

    while ll_r > ll_floor && ir <= N_MAX_STEP_OUT_ITERATIONS {
        nr *= 2.0;

        parameter_proposal(internal, anchor_vec, direction_vec, nr);

        ll_r = model
            .wrapped_log_likelihood(internal.view(), physical)
            .map_err(|source| SliceSamplerError::RightStep { source })?;
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

pub(crate) fn step_in<R: RngCore, L: LogLikelihoodModel + ?Sized>(
    rng: &mut R,
    model: &L,
    anchor_vec: ArrayView1<f64>,
    direction_vec: ArrayView1<f64>,
    search_bounds: LikelihoodBounds,
    internal: &mut ArrayViewMut1<f64>, // specifically to avoid many array allocations
    physical: &mut ArrayViewMut1<f64>,
) -> Result<f64, SliceSamplerError> {
    let mut nl = search_bounds.nl;
    let mut nr = search_bounds.nr;

    let mut shift: f64 = rng.random_range(nl..=nr);

    loop {
        parameter_proposal(internal, anchor_vec, direction_vec, shift);

        let ll = model
            .wrapped_log_likelihood(internal.view(), physical)
            .map_err(|source| SliceSamplerError::StepIn { source })?;

        if ll >= search_bounds.ll_floor {
            return Ok(ll);
        }

        if shift < 0.0 {
            nl = shift
        } else {
            nr = shift
        }

        shift = rng.random_range(nl..=nr);
    }
}

#[cfg(test)]
mod tests {

    use ndarray::array;

    use super::*;
    use crate::{
        functions::likelihood::log_normalised_gaussian,
        log_likelihood::f_toy::AnalyticNormalToyLogL,
    };

    fn test_ll() -> AnalyticNormalToyLogL {
        AnalyticNormalToyLogL::new(0.0, 1.0)
    }

    #[test]
    fn test_step_out() -> anyhow::Result<()> {
        let model = test_ll();

        let anchor = array![0.0];
        let direction = array![0.3];

        let mut internal = array![0.0];
        let mut physical = array![0.0];

        let ll_floor: f64 = log_normalised_gaussian(1.0, 0.0, 1.0);

        let bounds = step_out(
            &model,
            anchor.view(),
            direction.view(),
            &mut internal.view_mut(),
            &mut physical.view_mut(),
            ll_floor,
        )?;

        assert_eq!(bounds.nl, -4.0);
        assert_eq!(bounds.nr, 4.0);
        assert_eq!(bounds.ll_floor, ll_floor);
        assert_eq!(bounds._ll_l, log_normalised_gaussian(-1.2, 0.0, 1.0));
        assert_eq!(bounds._ll_r, log_normalised_gaussian(1.2, 0.0, 1.0));

        Ok(())
    }
}
