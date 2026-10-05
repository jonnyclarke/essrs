//! Helper functions to transform parameters from unconstrained spaces to constrained space
//! and apply the relevant Jacobian correction to the log-likelihood to account for the transform.

pub mod ordseq;
pub mod softplus;
pub mod tanh;

use ndarray::{Array1, ArrayView1, s};
use thiserror::Error;

use crate::functions::numerical::NumericalError;

#[derive(Debug, Error)]
pub enum TransformError {
    #[error("invalid value passed to softplus_inverse: {value} <= 0")]
    InvalidSoftplusInput { value: f64 },

    #[error("invalid value passed to tanh_inverse: -1 !<= {value} !<= 1")]
    InvalidAtanhInput { value: f64 },

    #[error("'{computation}' computation failed: {source}")]
    Transform {
        computation: &'static str,

        #[source]
        source: NumericalError,
    },
}

#[derive(Debug, Error)]
pub enum TransformHelperError {
    #[error("failure applying '{computation}' transform: {source}")]
    Transform {
        computation: &'static str,

        #[source]
        source: TransformError,
    },
}

pub fn apply_transform_array1<F>(
    f: F,
    xs: ArrayView1<f64>,
    ll: &mut f64,
) -> Result<Array1<f64>, TransformHelperError>
where
    F: Fn(f64, &mut f64) -> Result<f64, TransformError>,
{
    let mut out = Array1::<f64>::zeros(xs.len());

    for (o, &x) in out.iter_mut().zip(xs.iter()) {
        *o = f(x, ll).map_err(|source| TransformHelperError::Transform {
            computation: "array1",
            source,
        })?;
    }

    Ok(out)
}

use ndarray::Array2;

pub fn apply_transform_column<F>(
    f: F,
    array: &mut Array2<f64>,
    col: usize,
) -> Result<(), TransformHelperError>
where
    F: Fn(f64) -> Result<f64, TransformError>,
{
    for x in array.column_mut(col).iter_mut() {
        *x = f(*x).map_err(|source| TransformHelperError::Transform {
            computation: "column",
            source,
        })?;
    }

    Ok(())
}

/// Function to apply an inplace column transform reliant on two columns.
/// An example of usage is to provide ordered softplus transforms.
pub fn apply_inplace_multi_column_transform<F>(
    f: F,
    array: &mut Array2<f64>,
    col1: usize,
    col2: usize,
) -> Result<(), TransformHelperError>
where
    F: Fn(f64, &mut f64) -> Result<(), TransformError>,
{
    let (column1, mut column2) = array.multi_slice_mut((s![.., col1], s![.., col2]));

    for (x1, x2) in column1.iter().zip(column2.iter_mut()) {
        f(*x1, x2).map_err(|source| TransformHelperError::Transform {
            computation: "columns",
            source,
        })?;
    }

    Ok(())
}

// Function to apply the exponential change of variable
pub fn lj_exp(x: f64, ll: &mut f64) -> f64 {
    *ll += x;
    x.exp()
}

/// Function to apply logistic transform.
/// Maps values to interval (0, 1) asymptotically [values never reach 0 or 1].
/// 1 / (1 + exp(-x))
///
/// $$
/// y = \frac{1}{1 + e^{-x}}
/// $$
///
/// $$
/// \frac{dy}{dx} = \frac{e^{-x}}{\left( 1 + e^{-x} \right)^2}
/// $$
pub fn lj_logistic(x: f64, ll: &mut f64) -> f64 {
    let y = 1.0 / (1.0 + (-x).exp());

    *ll += y.ln() + (1.0 - y).ln();

    y
}

#[cfg(test)]
mod tests {

    use anyhow;
    use ndarray::array;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(
        0.0,
        0.0_f64.exp(),
    )]
    #[case(
        2.0,
        2.0_f64.exp(),
    )]
    #[case(
        1.2345,
        1.2345_f64.exp(),
    )]
    fn test_lj_exp(#[case] x: f64, #[case] target_y: f64) {
        let mut ll: f64 = 0.0;
        let y = lj_exp(x, &mut ll);

        assert_eq!(y, target_y);
        assert_eq!(ll, x);
    }

    #[rstest]
    #[case(0.0, 0.5, 0.25_f64.ln())]
    #[case(
        1.2345,
        1.0 / (1.0 + (-1.2345_f64).exp()),
        ((-1.2345_f64).exp() / (1.0 + (-1.2345_f64).exp()).powi(2)).ln()
    )]
    fn test_lj_logistic(#[case] x: f64, #[case] target_y: f64, #[case] target_ll: f64) {
        let mut ll: f64 = 0.0;
        let y = lj_logistic(x, &mut ll);

        assert_eq!(y, target_y);
        assert_eq!(ll, target_ll);
    }

    #[test]
    fn test_identity_transform() -> anyhow::Result<()> {
        let xs = array![1.0, 2.0, 3.0];
        let mut ll = 0.0;

        let result = apply_transform_array1(
            |x, _ll| Ok(x), // identity
            xs.view(),
            &mut ll,
        )?;

        assert_eq!(result, xs);
        assert_eq!(ll, 0.0);

        Ok(())
    }

    #[test]
    fn test_simple_scaling_transform() -> anyhow::Result<()> {
        let xs = array![1.0, 2.0, 3.0];
        let mut ll = 0.0;

        let result = apply_transform_array1(|x, _ll| Ok(2.0 * x), xs.view(), &mut ll)?;

        let expected = array![2.0, 4.0, 6.0];
        assert_eq!(result, expected);
        assert_eq!(ll, 0.0);

        Ok(())
    }

    #[test]
    fn test_log_likelihood_accumulation() -> anyhow::Result<()> {
        let xs = array![1.0, 2.0, 3.0];
        let mut ll = 0.0;

        let result = apply_transform_array1(
            |x, ll| {
                *ll += x;
                Ok(x + 1.0)
            },
            xs.view(),
            &mut ll,
        )?;

        let expected = array![2.0, 3.0, 4.0];

        assert_eq!(result, expected);
        assert_eq!(ll, 6.0); // 1 + 2 + 3

        Ok(())
    }

    #[test]
    fn test_apply_inplace_multi_column_transform() -> anyhow::Result<()> {
        let mut arr = array![[1.0, 2.0], [3.0, 4.0]];

        apply_inplace_multi_column_transform(
            |x0, x| {
                *x += x0;
                Ok(())
            },
            &mut arr,
            0,
            1,
        )?;

        assert_eq!(arr, array![[1.0, 3.0], [3.0, 7.0]]);

        Ok(())
    }
}
