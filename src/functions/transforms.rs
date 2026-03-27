//! Helper functions to transform parameters from unconstrained spaces to constrained space
//! and apply the relevant Jacobian correction to the log-likelihood to account for the transform.

use ndarray::{Array1, ArrayView1};

use crate::functions::numerical::{log_diff_exp, log_sum_exp};

pub fn apply_transform_array1<F>(f: F, xs: ArrayView1<f64>, ll: &mut f64) -> Array1<f64>
where
    F: Fn(f64, &mut f64) -> f64,
{
    let mut out = Array1::<f64>::zeros(xs.len());

    for (o, &x) in out.iter_mut().zip(xs.iter()) {
        *o = f(x, ll);
    }

    out
}

// Function to apply the exponential change of variable
pub fn lj_exp(x: f64, ll: &mut f64) -> f64 {
    *ll += x;
    x.exp()
}

#[cfg_attr(all(doc, feature = "doc-math"), katexit::katexit)]
/// Apply the `softplus` transformation which converts any value -inf < x < +inf to 0 < y.
///
/// Useful for any parameter that cannot be <0 e.g. the standard deviation of a normal distribution.
///
/// The transformation is given by,
/// $$
/// y = \ln\left(1 + e^x \right)
/// $$
/// with the jacobian determinant given by
/// $$
/// \frac{dy}{dx} = \frac{e^x}{1 + e^x}
/// $$
/// which simplifies, when logarithm is taken, to
/// $$
/// \ln\left( \frac{dy}{dx} \right) = \ln\left( \frac{e^x}{1 + e^x} \right) = x - \ln\left( 1 + e^x \right) = x - y
/// $$
///
/// # Arguments
/// * `x` - the unconstrained parameter to be transformed.
/// * `&mut ll` - a reference to the mutable log-likelihood which will be updated as necessary
///
/// # Returns
/// * `y` - the parameter in the constrained space `0 <`
pub fn lj_softplus(x: f64, ll: &mut f64) -> f64 {
    let y = log_sum_exp(0.0, x).unwrap();

    *ll += x - y;

    y
}

/// Apply the inverse of the softplus.
/// [Can be useful for passing initial values in constrained space]
///
/// $$
/// x = \ln\left(e^y - 1\right)
/// $$
///
/// # Arguments:
/// * `y` - the constrained parameter to be transformed to unconstrained space
///
/// # Returns
/// * `x` - the corresponding value in unconstrained space
pub fn inv_softplus(y: f64) -> f64 {
    log_diff_exp(y, 0.0).unwrap()
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

/// Map an unconstrained variable to -1 < y < 1
///
/// $$
/// y = \tanh\left( x \right) = \frac{\sinh\left(x\right)}{\cosh\left( x \right)}
/// $$
pub fn lj_tanh(x: f64, ll: &mut f64) -> f64 {
    let y = x.tanh();

    *ll += (1.0 - y.powi(2)).ln();

    y
}

#[cfg(test)]
mod tests {

    use approx::assert_relative_eq;
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
    #[case(
        0.0,
        (1.0_f64 + (0.0_f64).exp()).ln(),
        0.0 - (1.0_f64 + (0.0_f64).exp()).ln()
    )]
    #[case(
        2.0,
        (1.0_f64 + (2.0_f64).exp()).ln(),
        2.0 - (1.0_f64 + (2.0_f64).exp()).ln()
    )]
    #[case(
        -4.567,
        (1.0_f64 + (-4.567_f64).exp()).ln(),
        -4.567 - (1.0_f64 + (-4.567_f64).exp()).ln()
    )]
    fn test_lj_softplus(#[case] x: f64, #[case] target_y: f64, #[case] target_ll: f64) {
        let mut ll: f64 = 0.0;
        let y = lj_softplus(x, &mut ll);

        assert_eq!(y, target_y);
        assert_eq!(ll, target_ll);
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

    #[rstest]
    #[case(
        1.2345,
        (1.2345_f64).tanh(),
        (((1.2345_f64).cosh().powi(2) - (1.2345_f64).sinh().powi(2)) / (1.2345_f64).cosh().powi(2)).ln()
    )]
    fn test_lj_tanh(#[case] x: f64, #[case] target_y: f64, #[case] target_ll: f64) {
        let mut ll: f64 = 0.0;
        let y = lj_tanh(x, &mut ll);

        assert_eq!(y, target_y);
        assert_relative_eq!(ll, target_ll, epsilon = 1e-8);
    }

    /// Test inverse softplus
    #[rstest]
    #[case(1.0)]
    #[case(2.0)]
    #[case(3.0)]
    #[case(1.2345678)]
    fn test_inv_softplus_consistency(#[case] x: f64) {
        let mut ll = 0.0;
        assert_relative_eq!(x, inv_softplus(lj_softplus(x, &mut ll)), epsilon = 1e-8);
    }

    #[test]
    fn test_identity_transform() {
        let xs = array![1.0, 2.0, 3.0];
        let mut ll = 0.0;

        let result = apply_transform_array1(
            |x, _ll| x, // identity
            xs.view(),
            &mut ll,
        );

        assert_eq!(result, xs);
        assert_eq!(ll, 0.0);
    }

    #[test]
    fn test_simple_scaling_transform() {
        let xs = array![1.0, 2.0, 3.0];
        let mut ll = 0.0;

        let result = apply_transform_array1(|x, _ll| 2.0 * x, xs.view(), &mut ll);

        let expected = array![2.0, 4.0, 6.0];
        assert_eq!(result, expected);
        assert_eq!(ll, 0.0);
    }

    #[test]
    fn test_log_likelihood_accumulation() {
        let xs = array![1.0, 2.0, 3.0];
        let mut ll = 0.0;

        let result = apply_transform_array1(
            |x, ll| {
                *ll += x;
                x + 1.0
            },
            xs.view(),
            &mut ll,
        );

        let expected = array![2.0, 3.0, 4.0];

        assert_eq!(result, expected);
        assert_eq!(ll, 6.0); // 1 + 2 + 3
    }
}
