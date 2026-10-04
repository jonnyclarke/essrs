use crate::functions::transforms::TransformError;

#[cfg_attr(all(doc, feature = "doc-math"), katexit::katexit)]
/// Map an unconstrained variable to -1 < y < 1
///
/// $$
/// y = \tanh\left( x \right) = \frac{\sinh\left(x\right)}{\cosh\left( x \right)}
/// $$
pub fn tanh_x(x: f64) -> Result<f64, TransformError> {
    Ok(x.tanh())
}

#[cfg_attr(all(doc, feature = "doc-math"), katexit::katexit)]
/// Map an unconstrained variable to -1 < y < 1
///
/// $$
/// y = \tanh\left( x \right) = \frac{\sinh\left(x\right)}{\cosh\left( x \right)}
/// $$
pub fn tanh_log_jacobian(x: f64) -> Result<(f64, f64), TransformError> {
    let y = tanh_x(x)?;
    let lj = (1.0 - y.powi(2)).ln();
    Ok((y, lj))
}

pub fn tanh_log_jacobian_inplace(x: &mut f64, ll: &mut f64) -> Result<(), TransformError> {
    let (y, lj) = tanh_log_jacobian(*x)?;

    *x = y;
    *ll += lj;

    Ok(())
}

#[cfg_attr(all(doc, feature = "doc-math"), katexit::katexit)]
/// Inverse of the tanh(x) function, returning.
///
/// $$
/// x = \atanh\left( x \right)
/// $$
pub fn tanh_inverse(y: f64) -> Result<f64, TransformError> {
    if (y < -1.0) || (1.0 < y) {
        return Err(TransformError::InvalidAtanhInput { value: y });
    }

    Ok(y.atanh())
}

#[cfg(test)]
mod tests {

    use anyhow;
    use approx::assert_relative_eq;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(
        1.2345,
        (1.2345_f64).tanh(),
        (((1.2345_f64).cosh().powi(2) - (1.2345_f64).sinh().powi(2)) / (1.2345_f64).cosh().powi(2)).ln()
    )]
    fn test_tanh_log_jacobian(
        #[case] x: f64,
        #[case] target_y: f64,
        #[case] target_ll: f64,
    ) -> anyhow::Result<()> {
        let (y, ll) = tanh_log_jacobian(x)?;

        assert_eq!(y, target_y);
        assert_relative_eq!(ll, target_ll, epsilon = 1e-8);

        Ok(())
    }

    #[rstest]
    #[case(
        1.2345,
        (1.2345_f64).tanh(),
        (((1.2345_f64).cosh().powi(2) - (1.2345_f64).sinh().powi(2)) / (1.2345_f64).cosh().powi(2)).ln()
    )]
    fn test_tanh_log_jacobian_inplace(
        #[case] mut x: f64,
        #[case] target_x: f64,
        #[case] target_ll: f64,
    ) -> anyhow::Result<()> {
        let mut ll = 0.0;
        tanh_log_jacobian_inplace(&mut x, &mut ll)?;

        assert_eq!(x, target_x);
        assert_relative_eq!(ll, target_ll, epsilon = 1e-8);

        Ok(())
    }

    #[rstest]
    #[case(0.0, 0.0)]
    fn test_tanh_inverse(#[case] x: f64, #[case] y: f64) -> anyhow::Result<()> {
        let y_test = tanh_inverse(x)?;

        assert_eq!(y_test, y);

        Ok(())
    }

    #[rstest]
    #[case(-1.1)]
    #[case(1.1)]
    fn test_tanh_inverse_failure(#[case] x: f64) {
        assert!(matches!(
            tanh_inverse(x),
            Err(TransformError::InvalidAtanhInput { value: _ })
        ));
    }
}
