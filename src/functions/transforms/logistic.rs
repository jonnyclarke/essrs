use crate::functions::{
    // numerical::{log_diff_exp, log_sum_exp},
    transforms::TransformError,
};

#[cfg_attr(all(doc, feature = "doc-math"), katexit::katexit)]
/// Function to apply logistic transform.
/// Maps values to interval (0, 1) asymptotically [values never reach 0 or 1].
/// 1 / (1 + exp(-x))
///
/// $$
/// y = \frac{1}{1 + e^{-x}}
/// $$
pub fn logistic(x: f64) -> Result<f64, TransformError> {
    let y = 1.0 / (1.0 + (-x).exp());
    Ok(y)
}

#[cfg_attr(all(doc, feature = "doc-math"), katexit::katexit)]
/// Function to compute logistic transform and log-jacobian warp.
///
/// $$
/// \frac{dy}{dx} = \frac{e^{-x}}{\left( 1 + e^{-x} \right)^2}
/// $$
pub fn logistic_log_jacobian(x: f64) -> Result<(f64, f64), TransformError> {
    let y = logistic(x)?;

    let ll = y.ln() + (1.0 - y).ln();

    Ok((y, ll))
}

pub fn logistic_log_jacobian_inplace(x: &mut f64, ll: &mut f64) -> Result<(), TransformError> {
    let (y, lj) = logistic_log_jacobian(*x)?;

    *ll += lj;
    *x = y;

    Ok(())
}

#[cfg_attr(all(doc, feature = "doc-math"), katexit::katexit)]
/// Function to apply the inverse of the logistic transform
///
/// $$
/// x = \ln{\left[ \frac{y}{1 - y} \right]}
/// $$
pub fn logistic_inverse(y: f64) -> Result<f64, TransformError> {
    let x = (y / (1.0 - y)).ln();
    Ok(x)
}

#[cfg(test)]
mod tests {

    use anyhow;
    use approx::assert_relative_eq;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(-10.0)]
    #[case(-1.0)]
    #[case(0.0)]
    #[case(1.0)]
    #[case(10.0)]
    fn test_circular_behavior(#[case] x: f64) -> anyhow::Result<()> {
        assert_relative_eq!(x, logistic_inverse(logistic(x)?)?, epsilon = 1e-8);

        Ok(())
    }

    #[rstest]
    #[case(0.0, 0.5, 0.25_f64.ln())]
    #[case(
        1.2345,
        1.0 / (1.0 + (-1.2345_f64).exp()),
        ((-1.2345_f64).exp() / (1.0 + (-1.2345_f64).exp()).powi(2)).ln()
    )]
    fn test_logistic_log_jacobian(
        #[case] x: f64,
        #[case] target_y: f64,
        #[case] target_ll: f64,
    ) -> anyhow::Result<()> {
        let (y, ll) = logistic_log_jacobian(x)?;

        assert_eq!(y, target_y);
        assert_eq!(ll, target_ll);

        Ok(())
    }
}
