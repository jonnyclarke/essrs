//! Transforms for the so-called `stick-break` transform.
//!
//! This transform is where logistic units are chained together in order to generate a sequence of parameters whose weight sums to one.

// use anyhow::ensure;
use ndarray::{ArrayViewMut1, ArrayViewMut2};

// /// Stick-break transformation
// ///
// /// # Errors
// /// -
// pub fn stick_break(v1: f64, v2: f64) -> anyhow::Result<f64> {
//     Ok((1_f64 - v1) * v2)
// }

// /// Apply stick break inplace
// pub fn stick_break_inplace(v1: f64, v2: &mut f64) -> anyhow::Result<()> {
//     *v2 *= 1_f64 - v1;

//     Ok(())
// }

// // Stick break log jacobian
// /// # Errors
// /// * propagates `stick_break` errors
// pub fn stick_break_log_jacobian(v1: f64, v2: f64) -> anyhow::Result<(f64, f64)> {
//     let y = stick_break(v1, v2)?;

//     let lj = 1_f64 - v1;

//     Ok((y, lj))
// }

// /// Stick-break log jacobian -- inplace
// ///
// /// # Errors
// /// * propagates `stick_break_log_jacobian` errors
// pub fn stick_break_log_jacobian_inplace(v1: f64, v2: &mut f64, lj: &mut f64) -> anyhow::Result<()> {
//     let (y, lj_i) = stick_break_log_jacobian(v1, *v2)?;

//     *lj += lj_i;
//     *v2 = y;

//     Ok(())
// }

// /// Inverse stick break
// ///
// /// # Errors
// /// * the return value should be between `[0, 1]` for a correctly defined stick-break computation
// pub fn stick_break_inverse(v1: f64, v2: f64) -> anyhow::Result<f64> {
//     let y = v2 / (1_f64 - v1);
//     ensure!(
//         y.gt(&0_f64) && y.lt(&1_f64),
//         "Stick break inverse failed due to incompatible inputs, {v2} / (1 - {v1}) = {y}, which is out of bounds for logistic function"
//     );

//     Ok(y)
// }

/// Apply the `stick break` transformation with log-jacobian warp
///
/// # Arguments
/// * `data`: the parameter values being changed
/// * `weight_cols`: the columns through which transform should propagate
/// * `log_jacobian`: mutable reference where log jacobian warp is being accumulated
#[inline(always)]
pub fn stick_break_1d(
    data: &mut ArrayViewMut1<f64>,
    weight_cols: &[usize],
    log_jacobian: &mut f64,
) {
    let mut remaining = 1.0_f64;

    for &col in weight_cols {
        let v = data[col];

        *log_jacobian += remaining.ln();

        data[col] = v * remaining;

        remaining *= 1.0 - v;
    }
}

/// Apply the `stick break` transformation to a grid of `internal` parameters
///
/// # Arguments
/// * `data`: the grid of parameter values being changed
/// * `weight_cols`: the columns through which transform should propagate
/// * `log_jacobian`: mutable reference where log jacobian warp is being accumulated
#[inline(always)]
pub fn stick_break_array2_inplace(data: &mut ArrayViewMut2<f64>, weight_cols: &[usize]) {
    for mut row in data.rows_mut() {
        let mut remaining = 1.0;

        for &col in weight_cols {
            let v = row[col];

            row[col] = v * remaining;

            remaining *= 1.0 - v;
        }
    }
}

/// Apply the inverse `stick break` transformation to a grid of `physical` parameters
///
/// # Arguments
/// * `data`: the grid of parameter values being changed
/// * `weight_cols`: the columns through which transform should propagate
/// * `log_jacobian`: mutable reference where log jacobian warp is being accumulated
#[inline(always)]
pub fn stick_break_inverse_array2_inplace(data: &mut ArrayViewMut2<f64>, weight_cols: &[usize]) {
    for mut row in data.rows_mut() {
        let mut remaining = 1.0;

        for &col in weight_cols {
            let weight = row[col];

            row[col] = weight / remaining;

            remaining -= weight;
        }
    }
}

#[cfg(test)]
mod tests {

    use anyhow;
    use approx::assert_relative_eq;
    use ndarray::{Array2, array};
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(array![[0.8, 0.1]], array![[0.8, 0.5]], vec![0, 1])]
    #[case(
        array![[0.5, 0.25, 0.1], [0.6, 0.2, 0.1]],
        array![[0.5, 0.25 / 0.5, 0.1 / 0.25], [0.6, 0.2 / 0.4, 0.1/0.2]],
        vec![0, 1, 2]
    )]
    #[case(
        array![[0.5, 0.2, 0.25, 0.1], [0.6, 0.2, 0.2, 0.1]],
        array![[0.5, 0.2,  0.25 / 0.5, 0.1 / 0.25], [0.6, 0.2, 0.2 / 0.4, 0.1 / 0.2]],
        vec![0, 2, 3]
    )]
    fn test_stick_break_inverse_array2_inplace(
        #[case] mut a: Array2<f64>,
        #[case] b: Array2<f64>,
        #[case] cols: Vec<usize>,
    ) -> anyhow::Result<()> {
        stick_break_inverse_array2_inplace(&mut a.view_mut(), &cols);

        assert_relative_eq!(a, b, epsilon = 1e-8);

        Ok(())
    }
}
