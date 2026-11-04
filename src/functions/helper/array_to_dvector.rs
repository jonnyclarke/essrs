use anyhow::bail;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, ArrayView1};

/// Convert ndarray Array1 into nalgebra `DVector`.
///
/// This is useful when transforming parameters into format required for N-dimensional normal distribution computations.
pub fn array1_to_dvector(array: &Array1<f64>) -> DVector<f64> {
    DVector::from_iterator(array.len(), array.iter().copied())
}

/// Convert ndarray `ArrayView1` into nalgebra `DVector`.
///
/// This is useful when transforming parameters into format required for N-dimensional normal distribution computations.
pub fn arrayview1_to_dvector(array: &ArrayView1<f64>) -> DVector<f64> {
    DVector::from_iterator(array.len(), array.iter().copied())
}

/// Construct a symmetric covariance matrix from standard deviations and correlations.
///
/// Performs transform from Array1 type into nalgebra type needed for N-dimensional normal distribution computations.
///
/// # Arguments
/// * `sigma`: Standard deviations (diagonal = sigma`\[`i`\]`^2)
/// * `correlations`: Lower triangular off-diagonal elements, ordered row-wise: [p10, p20, p21, p30, p31, p32, ...]
///
/// # Returns
/// * covariance matrix
///
/// # Errors
/// * correlations is the wrong size relative to sigma
pub fn covariance_to_dmatrix(
    sigma: &Array1<f64>,
    correlations: &Array1<f64>,
) -> anyhow::Result<DMatrix<f64>> {
    let n = sigma.len();

    let expected = n * (n - 1) / 2;

    if correlations.len() != expected {
        bail!(
            "Expected {} correlations values for {} dimensions, got {}",
            expected,
            n,
            correlations.len()
        );
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
            let value = correlations[k] * sigma[i] * sigma[j];

            covariance[(i, j)] = value;
            covariance[(j, i)] = value;

            k += 1;
        }
    }

    Ok(covariance)
}

#[cfg(test)]
mod tests {

    use nalgebra::{DMatrix, dmatrix};
    use ndarray::array;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(
            array![0.8, 0.8],
            array![0.5],
            dmatrix![
                0.8_f64.powi(2), 0.5*0.8*0.8;
                0.5*0.8*0.8, 0.8_f64.powi(2)
            ],
        )
    ]
    fn test_covariance_to_dmatrix(
        #[case] sigma: Array1<f64>,
        #[case] correlation: Array1<f64>,
        #[case] truth: DMatrix<f64>,
    ) -> anyhow::Result<()> {
        let matrix = covariance_to_dmatrix(&sigma, &correlation)?;

        assert_eq!(matrix, truth);

        Ok(())
    }
}
