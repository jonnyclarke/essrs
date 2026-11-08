//! Helper functions to transform parameters from unconstrained spaces to constrained space
//! and apply the relevant Jacobian correction to the log-likelihood to account for the transform.

pub mod exponential;
pub mod logistic;
pub mod ordseq;
pub mod softplus;
pub mod stick_break;
pub mod tanh;

use anyhow::{Context, ensure};
use ndarray::{Array1, ArrayView1, ArrayViewMut2, s};

/// Compute a transform over elements in array
///
/// # Arguments:
/// * f: function to be applied
/// * xs: Array to be iterated over
/// * ll: Mutable float which is permuted within function
///
/// # Return
/// * Array containing transformed values
///
/// # Errors
/// * propagate supplied `f` error
pub fn apply_transform_array1<F>(
    f: F,
    xs: ArrayView1<f64>,
    ll: &mut f64,
) -> anyhow::Result<Array1<f64>>
where
    F: Fn(f64, &mut f64) -> anyhow::Result<f64>,
{
    let mut out = Array1::<f64>::zeros(xs.len());

    for (o, &x) in out.iter_mut().zip(xs.iter()) {
        *o = f(x, ll).context("elementwise function application failed")?
    }

    Ok(out)
}

/// Apply a transform into a mutable Array2 column
///
/// # Arguments
/// * f: function which will be applied
/// * array: array within which a mutable column will be updated
/// * col: index of the column to be changed
///
/// # Errors
/// * column index outside of array shape
/// * propagate column wise transform function failure
pub fn apply_transform_column<F>(
    f: F,
    array: &mut ArrayViewMut2<f64>,
    col: usize,
) -> anyhow::Result<()>
where
    F: Fn(f64) -> anyhow::Result<f64>,
{
    ensure!(
        col.ge(&0_usize) && col.lt(&(array.ncols())),
        "column index, {col}, outside bounds of array shape, {:#?}",
        array.shape()
    );
    for x in array.column_mut(col).iter_mut() {
        *x = f(*x).context("columnwise transform failed")?
    }

    Ok(())
}

/// Function to apply an inplace column transform reliant on two columns.
/// An example of usage is to provide ordered softplus transforms.
///
/// # Errors
/// * column indices are beyond the column range of the given array
/// * propagates error in event the user defined function fails
pub fn apply_inplace_multi_column_transform<F>(
    f: F,
    array: &mut ArrayViewMut2<f64>,
    col1: usize,
    col2: usize,
) -> anyhow::Result<()>
where
    F: Fn(f64, &mut f64) -> anyhow::Result<()>,
{
    ensure!(
        col1.ge(&0_usize) && col1.lt(&(array.ncols())),
        "column index, {col1}, outside bounds of array shape, {:#?}",
        array.shape()
    );
    ensure!(
        col2.ge(&0_usize) && col2.lt(&(array.ncols())),
        "column index, {col2}, outside bounds of array shape, {:#?}",
        array.shape()
    );
    let (column1, mut column2) = array.multi_slice_mut((s![.., col1], s![.., col2]));

    for (x1, x2) in column1.iter().zip(column2.iter_mut()) {
        f(*x1, x2).context("columnwise [a, mut b] has failed")?
    }

    Ok(())
}

#[cfg(test)]
mod tests {

    use anyhow;
    use ndarray::array;
    use rstest::rstest;

    use super::*;
    use crate::functions::transforms::exponential::lj_exp;

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
            &mut arr.view_mut(),
            0,
            1,
        )?;

        assert_eq!(arr, array![[1.0, 3.0], [3.0, 7.0]]);

        Ok(())
    }
}
