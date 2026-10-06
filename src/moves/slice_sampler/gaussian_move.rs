//! Implementation of Gaussian Move
//! The algorithm is as follows:
//! - the covariance of the walker distribution is computed
//! - this is decomposed via cholesky
//! - the direction vector is then randomly sampled from the cholesky decomposition
//!
//! This move algorithm permits all directions in parameter space whilst also respecting the geometry of the posterior given by the current walker distribution.

use anyhow;
use nalgebra::{Cholesky, DMatrix, DVector};
use ndarray::{Array1, Array2};
use rand_distr::{Distribution, StandardNormal};
use thiserror::Error;

use crate::{
    log_likelihood::LogLikelihoodModel,
    moves::EnsembleMove,
    slice::{step_in, step_out},
    state::WalkerState,
};

#[derive(Debug, Error)]
pub enum GaussianMoveError {
    #[error("failure converting state matrix to a slice")]
    StateMatrixToSliceError,

    #[error("failure converting covariance matrix into a Cholesky object")]
    CovarianceToCholeskyError,
}

fn state_to_covariance(state_matrix: Array2<f64>) -> anyhow::Result<DMatrix<f64>> {
    let mat = DMatrix::from_row_slice(
        state_matrix.nrows(),
        state_matrix.ncols(),
        state_matrix
            .as_slice()
            .ok_or(GaussianMoveError::StateMatrixToSliceError)?,
    );

    let mean = DVector::from_iterator(mat.ncols(), (0..mat.ncols()).map(|j| mat.column(j).mean()));

    let dim = mat.ncols();
    let n = mat.nrows() as f64;
    let mut cov = DMatrix::zeros(dim, dim);

    for i in 0..dim {
        for j in 0..dim {
            let cov_ij = mat
                .column(i)
                .iter()
                .zip(mat.column(j).iter())
                .map(|(x, y)| (x - mean[i]) * (y - mean[j]))
                .sum::<f64>()
                / (n - 1.0);

            cov[(i, j)] = cov_ij;
        }
    }

    Ok(cov)
}

pub struct GaussianMove {}

impl GaussianMove {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for GaussianMove {
    fn default() -> Self {
        Self::new()
    }
}

impl EnsembleMove for GaussianMove {
    fn jump<L: LogLikelihoodModel>(
        &self,
        log_likelihood_model: &L,
        state_i: &WalkerState,
        state_j: &mut WalkerState,
    ) -> anyhow::Result<()> {
        let mut rng = rand::rng();
        let state_matrix = state_i.get_state_matrix().to_owned();

        let cov = state_to_covariance(state_matrix)?;
        let dim = cov.ncols();

        // Obtain Cholesky matrix
        let chol = Cholesky::new(cov).ok_or(GaussianMoveError::CovarianceToCholeskyError)?;
        let l = chol.l();

        // we generate multiple scratch arrays to avoid constantly re-allocating
        let mut internal = state_i.get_ith_state_vector(0).to_owned();
        let mut physical = state_i.get_ith_state_vector(0).to_owned();

        for i in 0..state_i.get_ll_vector().len() {
            // get the threshold for acceptance
            let ll_floor = state_i.get_ith_ll(i) + self.get_likelihood_floor(&mut rng);

            let current = state_i.get_ith_state_vector(i).to_owned();

            let z = DVector::from_iterator(dim, (0..dim).map(|_| StandardNormal.sample(&mut rng)));
            let direction = Array1::from((&l * &z).iter().cloned().collect::<Vec<f64>>());

            let search_bounds = step_out(
                log_likelihood_model,
                current.view(),
                direction.view(),
                &mut internal.view_mut(),
                &mut physical.view_mut(),
                ll_floor,
            )?;

            let log_likelihood = step_in(
                &mut rng,
                log_likelihood_model,
                current.view(),
                direction.view(),
                search_bounds,
                &mut internal.view_mut(),
                &mut physical.view_mut(),
            )?;

            state_j.get_mut_ith_state_vector(i).assign(&internal);
            *state_j.get_mut_ith_ll(i) = log_likelihood;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::dmatrix;
    use ndarray::array;

    use super::*;

    // unit tests derived from running data through python numpy routine
    #[test]
    fn test_state_to_covariance() -> anyhow::Result<()> {
        let state = array![[1.0, 1.2], [2.3, 1.8], [1.8, 2.2], [0.9, 3.2]];

        let cov = state_to_covariance(state)?;
        let true_cov: DMatrix<f64> = dmatrix![
            0.44666667, -0.14;
            -0.14, 0.70666667;
        ];
        assert_relative_eq!(cov, true_cov, epsilon = 1e-8);

        Ok(())
    }
}
