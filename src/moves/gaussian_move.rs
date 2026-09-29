use nalgebra::{Cholesky, DMatrix, DVector};
use ndarray::Array1;
// This struct takes the current state and computes the covariance matrix
// It then samples the covariance matrix around a random point and if the point is an improvement
// then it accepts the jump.
use rand::RngCore;
use rand_distr::{Distribution, StandardNormal};

use crate::{
    log_likelihood::LogLikelihoodModel,
    moves::EnsembleMove,
    slice::{step_in, step_out},
    state::WalkerState,
};

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
    fn jump(
        &self,
        rng: &mut dyn RngCore,
        log_likelihood_model: &dyn LogLikelihoodModel,
        state_i: &WalkerState,
        state_j: &mut WalkerState,
    ) {
        let state_matrix = state_i.get_state_matrix().to_owned();

        let mat = DMatrix::from_row_slice(
            state_matrix.nrows(),
            state_matrix.ncols(),
            state_matrix.as_slice().unwrap(),
        );

        let mean =
            DVector::from_iterator(mat.ncols(), (0..mat.ncols()).map(|j| mat.column(j).mean()));

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

                if i == j {
                    cov[(i, j)] *= 1.0 + 1e-8; // to avoid numerical problems
                }
            }
        }
        // Obtain Cholesky matrix
        let chol = Cholesky::new(cov).unwrap();
        let l = chol.l();

        for i in 0..state_i.get_ll_vector().len() {
            // get the threshold for acceptance
            let ll_floor = state_i.get_ith_ll(i) + self.get_likelihood_floor(rng);

            let current = state_i.get_ith_state_vector(i).to_owned();

            let z = DVector::from_iterator(dim, (0..dim).map(|_| StandardNormal.sample(rng)));
            let direction = Array1::from((&l * &z).iter().cloned().collect::<Vec<f64>>());

            let search_bounds = step_out(
                log_likelihood_model,
                current.view(),
                direction.view(),
                ll_floor,
            );

            let (ll, accepted) = step_in(
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
