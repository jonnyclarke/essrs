//! Module containing numerical helper functions.
//! These functions are designed to help perform logarithmic operations while avoiding numerical under- or overflow.
//! Such functions arise often in mixture models where single-point probability is the sum of multiple components.
use ndarray::{Array1, ArrayView1};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NumericalError {
    #[error("non-finite argument passed to {computation}: {a}")]
    InvalidArgumentError { computation: &'static str, a: f64 },

    #[error(
        "invalid values passed to `log_diff_exp`: {a} <= {b} :: results in ln(-0) which is undefined"
    )]
    InvalidLogDiffExpInputs { a: f64, b: f64 },
}

pub fn apply_numerical_f_array1<F>(
    f: F,
    a_arr: ArrayView1<f64>,
    b_arr: ArrayView1<f64>,
) -> Result<Array1<f64>, NumericalError>
where
    F: Fn(f64, f64) -> Result<f64, NumericalError>,
{
    let mut out = Array1::<f64>::zeros(a_arr.len());

    for ((o, &a), &b) in out.iter_mut().zip(a_arr.iter()).zip(b_arr.iter()) {
        *o = f(a, b)?; // propagates None
    }

    Ok(out)
}

pub fn log_diff_exp(a: f64, b: f64) -> Result<f64, NumericalError> {
    if a <= b {
        // Difference would be zero or negative → log undefined
        return Err(NumericalError::InvalidLogDiffExpInputs { a, b });
    }

    Ok(a + (1.0 - (b - a).exp()).ln())
}

#[cfg_attr(all(doc, feature = "doc-math"), katexit::katexit)]
/// Function to sum two exponential values.
///
/// $$
/// y = \ln\left( e^a + e^b \right)
/// $$
///
/// Applies `log_sum_exp` algorithm to safely combine the two values.
///
/// $$
/// y = \max(a,b) + \ln\left(e^{a-\max(a,b)} + e^{b-\max(a,b)}\right)
/// $$
///
/// # Arguments
/// * `a` - the logarithm of the first value to be added.
/// * `b` - the logarithm of the second value to be added
///
/// # Returns
/// `Option<f64>` - Logarithm of sum of exponentiated values.
pub fn log_sum_exp(a: f64, b: f64) -> Result<f64, NumericalError> {
    if !a.is_finite() {
        return Err(NumericalError::InvalidArgumentError {
            computation: "log_sum_exp",
            a,
        });
    }
    if !b.is_finite() {
        return Err(NumericalError::InvalidArgumentError {
            computation: "log_sum_exp",
            a: b,
        });
    }

    let v_max = a.max(b);

    Ok(v_max + ((a - v_max).exp() + (b - v_max).exp()).ln())
}

#[cfg(test)]
mod tests {
    use anyhow;
    use approx::assert_relative_eq;
    use ndarray::array;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(0.0, 0.0, 2.0_f64.ln())]
    #[case(4.0, 5.0, ((4.0_f64).exp() + (5.0_f64).exp()).ln())]
    #[case(4.0, -2.0, ((4.0_f64).exp() + (-2.0_f64).exp()).ln())]
    fn test_log_sum_exp(
        #[case] a: f64,
        #[case] b: f64,
        #[case] expected: f64,
    ) -> anyhow::Result<()> {
        assert_relative_eq!(log_sum_exp(a, b)?, expected, epsilon = 1e-8);

        Ok(())
    }

    #[rstest]
    #[case(12.2, -21.098, ((12.2_f64).exp() + (-21.098_f64).exp()).ln())]
    #[case(12.2, 21.098, ((-12.2_f64).exp() + (21.098_f64).exp()).ln())]
    fn test_log_sum_exp_large(
        #[case] a: f64,
        #[case] b: f64,
        #[case] expected: f64,
    ) -> anyhow::Result<()> {
        assert_relative_eq!(log_sum_exp(a, b)?, expected, epsilon = 1e-2);

        Ok(())
    }

    #[rstest]
    #[case(1.0, 0.0, ((1.0_f64).exp() - (0.0_f64).exp()).ln())]
    #[case(4.0, 3.0, ((4.0_f64).exp() - (3.0_f64).exp()).ln())]
    #[case(4.0, -3.0, ((4.0_f64).exp() - (-3.0_f64).exp()).ln())]
    fn test_log_diff_exp(
        #[case] a: f64,
        #[case] b: f64,
        #[case] expected: f64,
    ) -> anyhow::Result<()> {
        assert_relative_eq!(log_diff_exp(a, b)?, expected, epsilon = 1e-8);

        Ok(())
    }

    #[rstest]
    #[case(1.0, 1.0)]
    #[case(0.0, 1.0)]
    #[case(-2.0, -1.0)]
    fn test_log_diff_exp_none(#[case] a: f64, #[case] b: f64) {
        assert!(matches!(
            log_diff_exp(a, b),
            Err(NumericalError::InvalidLogDiffExpInputs { a: _, b: _ })
        ));
    }

    #[test]
    fn test_apply_numerical_f_array1_all_some() -> anyhow::Result<()> {
        // simple f that always returns Some
        let f = |a: f64, b: f64| Ok(a + b);

        let a_arr = array![1.0, 2.0, 3.0];
        let b_arr = array![4.0, 5.0, 6.0];

        let result = apply_numerical_f_array1(f, a_arr.view(), b_arr.view())?;

        let expected = array![5.0, 7.0, 9.0];
        assert!(
            result
                .iter()
                .zip(expected.iter())
                .all(|(r, e)| (r - e).abs() < 1e-8)
        );

        Ok(())
    }

    // #[test]
    // fn test_apply_numerical_f_array1_some_and_none() {
    //     // f returns None if a <= b
    //     let f = |a: f64, b: f64| if a > b { Some(a - b) } else { None };

    //     let a_arr = array![3.0, 1.0, 5.0];
    //     let b_arr = array![2.0, 2.0, 1.0];

    //     // second element triggers None
    //     let result = apply_numerical_f_array1(f, a_arr.view(), b_arr.view());

    //     assert!(result.is_none());
    // }

    #[test]
    fn test_apply_numerical_f_array1_empty_arrays() -> anyhow::Result<()> {
        let f = |a: f64, b: f64| Ok(a * b);

        let a_arr = array![];
        let b_arr = array![];

        let result = apply_numerical_f_array1(f, a_arr.view(), b_arr.view())?;
        assert_eq!(result.len(), 0);

        Ok(())
    }
}
