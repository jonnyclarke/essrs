use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, ArrayView1};

/// Convert ndarray Array1 into nalgebra DVector.
///
/// This is useful when transforming parameters into format required for N-dimensional normal distribution computations.
pub fn array1_to_dvector(array: &Array1<f64>) -> DVector<f64> {
    DVector::from_iterator(array.len(), array.iter().copied())
}

/// Convert ndarray ArrayView1 into nalgebra DVector.
///
/// This is useful when transforming parameters into format required for N-dimensional normal distribution computations.
pub fn arrayview1_to_dvector(array: &ArrayView1<f64>) -> DVector<f64> {
    DVector::from_iterator(array.len(), array.iter().copied())
}

/// Construct a symmetric covariance matrix from standard deviations and correlations.
///
/// Performs transform from Array1 type into nalgebra type needed for N-dimensional normal distribution computations.
///
/// # Arguments:
/// * sigma: Standard deviations (diagonal = sigma[i]^2)
/// * correlations: Lower triangular off-diagonal elements, ordered row-wise: [p10, p20, p21, p30, p31, p32, ...]
///
/// # Returns:
/// * covariance matrix
pub fn covariance_to_dmatrix(
    sigma: &Array1<f64>,
    covariances: &Array1<f64>,
) -> Result<DMatrix<f64>, String> {
    let n = sigma.len();

    let expected = n * (n - 1) / 2;

    if covariances.len() != expected {
        return Err(format!(
            "Expected {} covariance values for {} dimensions, got {}",
            expected,
            n,
            covariances.len()
        ));
    }

    let mut covariance = DMatrix::<f64>::zeros(n, n);

    // Populate diagonal using variance = sigma^2
    for i in 0..n {
        covariance[(i, i)] = sigma[i] * sigma[i];
    }

    // Populate lower triangle and mirror into upper triangle
    let mut k = 0;

    for i in 1..n {
        for j in 0..i {
            let value = covariances[k];

            covariance[(i, j)] = value;
            covariance[(j, i)] = value;

            k += 1;
        }
    }

    Ok(covariance)
}
