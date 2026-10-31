//! Functions for computation of normalised N-dimensional normal distribution log-likelihoods.

use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};

/// Compute the numerical f(n_dimensions) constant 
pub fn ndim_to_norm_const(n: usize) -> f64 {
    n as f64 * (2.0 * PI).ln()
}

/// Compute a covariance matrix to cholesky
pub fn covariance_to_cholesky_l(covariance: &DMatrix<f64>) -> DMatrix<f64> {
    // Cholesky decomposition: covariance = L * L^T
    let chol = covariance
        .clone()
        .cholesky()
        .expect("Covariance must be positive definite");

    chol.l()
}

/// Compute log determinent from lower triangular cholesky matrix
pub fn cholesky_l_to_log_det(cholesky_l: &DMatrix<f64>) -> f64 {
    (0..cholesky_l.nrows())
        .map(|i| cholesky_l[(i, i)].ln())
        .sum::<f64>()
        * 2.0
}

/// Compute mahalanobis distance of (x - mu) from cholesky matrix
pub fn mahalanobis_from_cholesky(
    x: &DVector<f64>,
    mean: &DVector<f64>,
    cholesky_l: &DMatrix<f64>,
) -> f64 {
    let diff = x - mean;

    let y = cholesky_l
        .solve_lower_triangular(&diff)
        .expect("Triangle solve failed");

    y.dot(&y)
}

/// Helper function to combine three separate numerical components
pub fn normal_nd_ll(n_pi_ln_const: f64, log_det: f64, mahalanobis: f64) -> f64 {
    -0.5 * (n_pi_ln_const + log_det + mahalanobis)
}

/// Compute log likelihood of an n-dimensional normal distribution
pub fn multivariate_gaussian_log_likelihood(
    x: &DVector<f64>,
    mean: &DVector<f64>,
    covariance: &DMatrix<f64>,
) -> f64 {
    let n = x.len();

    assert_eq!(mean.len(), n);
    assert_eq!(covariance.shape(), (n, n));

    let cholesky_l = covariance_to_cholesky_l(covariance);

    let log_det = cholesky_l_to_log_det(&cholesky_l);

    let mahalanobis = mahalanobis_from_cholesky(x, mean, &cholesky_l);

    -0.5 * normal_nd_ll(ndim_to_norm_const(n), log_det, mahalanobis)
}
