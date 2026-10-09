use crate::functions::{
    numerical::{log_diff_exp, log_sum_exp},
    transforms::TransformError,
};

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
pub fn softplus(x: f64) -> Result<f64, TransformError> {
    let y = log_sum_exp(0.0, x).map_err(|source| TransformError::Transform {
        computation: "softplus",
        source,
    })?;

    Ok(y)
}

pub fn softplus_log_jacobian(x: f64) -> Result<(f64, f64), TransformError> {
    let y = softplus(x)?;

    Ok((y, x - y))
}

pub fn softplus_log_jacobian_inplace(x: &mut f64, ll: &mut f64) -> Result<(), TransformError> {
    let (s, lj) = softplus_log_jacobian(*x)?;

    *ll += lj;
    *x = s;

    Ok(())
}

#[cfg_attr(all(doc, feature = "doc-math"), katexit::katexit)]
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
pub fn softplus_inverse(y: f64) -> Result<f64, TransformError> {
    if y <= 0.0 {
        return Err(TransformError::InvalidSoftplusInput { value: y });
    }

    let x = log_diff_exp(y, 0.0).map_err(|source| TransformError::Transform {
        computation: "inverse-softplus",
        source,
    })?;

    Ok(x)
}

#[cfg(test)]
mod tests {

    use anyhow;
    use approx::assert_relative_eq;
    use rstest::rstest;

    use super::*;

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
    fn test_softplus_log_jacobian_inplace(
        #[case] mut x: f64,
        #[case] target_y: f64,
        #[case] target_ll: f64,
    ) -> anyhow::Result<()> {
        let mut ll: f64 = 0.0;
        softplus_log_jacobian_inplace(&mut x, &mut ll)?;

        assert_eq!(x, target_y);
        assert_eq!(ll, target_ll);

        Ok(())
    }

    /// Test inverse softplus
    #[rstest]
    #[case(1.0)]
    #[case(2.0)]
    #[case(3.0)]
    #[case(1.2345678)]
    fn test_softplus_inverse_consistency(#[case] x: f64) -> anyhow::Result<()> {
        assert_relative_eq!(x, softplus_inverse(softplus(x)?)?, epsilon = 1e-8);

        Ok(())
    }

    /// Test softplus failure modes
    #[test]
    fn test_softplus_error_handling() {
        let x = f64::NAN;
        let y = softplus(x);

        assert!(matches!(
            y,
            Err(TransformError::Transform {
                computation: _,
                source: _
            })
        ));
    }

    #[rstest]
    #[case(-1.0)]
    #[case(0.0)]
    fn test_softplus_inverse_error_handling(#[case] x: f64) {
        assert!(matches!(
            softplus_inverse(x),
            Err(TransformError::InvalidSoftplusInput { value: _ })
        ));
    }
}
