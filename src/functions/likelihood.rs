//! Pre-definition of common log-likelihood functions for simplicity.

#[cfg_attr(all(doc, feature = "doc-math"), katexit::katexit)]
/// Function to compute the normalised normal distribution.
///
/// $$
/// P\left(x\right) = \frac{1}{\sqrt{2 \pi \sigma^2}}  \exp\left(-\frac{1}{2} \left[\frac{x-\mu}{\sigma}\right]^2\right)
/// $$
///
/// # Arguments
/// * `x` - The value to be evaluated under the normal distribution.
/// * `mu` - The mean of the normal distribution.
/// * `sigma` - The standard deviation of the normal distribution.
///
/// # Returns
/// Probability of point `x` under the Normal distribution.
///
/// # Examples
///
/// ```
/// let example = essrs::functions::likelihood::normalised_gaussian(0.0, 0.0, 1.0);
/// assert_eq!(example, 1.0 / (2.0 * std::f64::consts::PI).sqrt());
/// ```
pub fn normalised_gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
    (-0.5 * ((x - mu) / sigma).powi(2)).exp() / ((2.0 * std::f64::consts::PI).sqrt() * sigma)
}

/// Function to compute the logarithm of the normalised Normal distribution.
///
/// # Arguments
/// * `x` - The value to be evaluated under the normal distribution.
/// * `mu` - The mean of the normal distribution.
/// * `sigma` - The standard deviation of the normal distribution.
///
/// # Returns
/// Logarithm of probability of `x` under the defined Normal distribution.
pub fn log_normalised_gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
    -0.5 * ((x - mu) / sigma).powi(2) + -0.5 * (2.0 * std::f64::consts::PI * sigma.powi(2)).ln()
}

/// Modification of logarithm of normal distribution to accept variance directly.
/// This can be more efficient in some heavy workflows to avoid the .powi(2) and .sqrt() calls.
///
/// # Arguments
/// * `x` - The value to be evaluated under the normal distribution.
/// * `mu` - The mean of the normal distribution.
/// * `sigma2` - The variance of the normal distribution.
///
/// # Returns
/// Logarithm of probability of `x` under the defined Normal distribution.
pub fn log_normalised_gaussian_s2(x: f64, mu: f64, sigma2: f64) -> f64 {
    -0.5 * ((x - mu).powi(2) / sigma2) + -0.5 * (2.0 * std::f64::consts::PI * sigma2).ln()
}

/// Evaluate the normal distribution in two dimensions; neglecting the 2pi factors.
///
/// # Arguments
/// `x` - x-coordinate of the point to be evaluated.
/// `y` - y-coordinate of the point to be evaluated.
/// `mu_x` - The mean of the 2D Normal distribution in the `x` dimension.
/// `mu_y` - The mean of the 2D Normal distribution in the `y` dimension.
/// `a` - The upper left value of the covariance matrix. Corresponds to the variance in `x` dimension.
/// `bc` - The upper right == lower left value of the covariance matrix. Corresponds to x:y correlation * standard deviation in x * standard deviation in y.
/// `d` - The lower right value of the covariance matrix. Corresponds to the variance in `y` dimension.
pub fn log_normal_2d(x: f64, y: f64, mu_x: f64, mu_y: f64, a: f64, bc: f64, d: f64) -> f64 {
    let dx = mu_x - x;
    let dy = mu_y - y;

    let det = a * d - bc.powi(2);

    -0.5 * det.ln() - 0.5 * (dx * (d * dx - bc * dy) + dy * (a * dy - bc * dx)) / det // constant factor of pi neglected
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(0.0, 0.0, 1.0, 1.0 / (2.0 * std::f64::consts::PI).sqrt())]
    #[case(1.0, 0.0, 1.0, (-0.5_f64).exp() / (2.0 * std::f64::consts::PI).sqrt())]
    #[case(-1.0, 0.0, 1.0, (-0.5_f64).exp() / (2.0 * std::f64::consts::PI).sqrt())]
    fn test_normalised_gaussian(
        #[case] x: f64,
        #[case] mu: f64,
        #[case] sigma: f64,
        #[case] expected: f64,
    ) {
        assert!(
            normalised_gaussian(x, mu, sigma) == expected,
            "Normalised Gaussian incorrect"
        );
    }

    #[rstest]
    #[case(0.0, 0.0, 1.0, -0.5 * (2.0 * std::f64::consts::PI).ln())]
    #[case(1.0, 0.0, 1.0, -0.5 + -0.5 * (2.0 * std::f64::consts::PI).ln())]
    #[case(-1.0, 0.0, 1.0, -0.5 + -0.5 * (2.0 * std::f64::consts::PI).ln())]
    fn test_log_normalised_gaussian(
        #[case] x: f64,
        #[case] mu: f64,
        #[case] sigma: f64,
        #[case] expected: f64,
    ) {
        assert!(
            log_normalised_gaussian(x, mu, sigma) == expected,
            "Normalised Gaussian incorrect"
        );
    }

    #[rstest]
    #[case(0.0, 0.0, 1.0)]
    #[case(2.0, 1.0, 1.4)]
    #[case(-1.32, 0.23, 1.976)]
    fn test_consistency(#[case] x: f64, #[case] mu: f64, #[case] sigma: f64) {
        assert_relative_eq!(
            normalised_gaussian(x, mu, sigma).ln(),
            log_normalised_gaussian(x, mu, sigma),
            epsilon = 1e-8
        );

        assert_relative_eq!(
            log_normalised_gaussian(x, mu, sigma),
            log_normalised_gaussian_s2(x, mu, sigma.powi(2)),
            epsilon = 1e-8
        );
    }

    #[rstest]
    #[case(
        0.0, 0.0,
        0.0, 0.0,
        2.0, 0.3, 1.5,
        -0.5_f64 * (1.5_f64 * 2.0_f64 - 0.3_f64 * 0.3_f64).ln()
    )] // test centroid value which is simply determinant
    #[case(
        1.0, 2.0,
        0.0, 0.0,
        1.0, 0.0, 1.0,
        -0.5 * (1.0 * 1.0 + 2.0 * 2.0)
    )] // determinant == 0
    fn test_log_normal_2d(
        #[case] x: f64,
        #[case] y: f64,
        #[case] mu_x: f64,
        #[case] mu_y: f64,
        #[case] a: f64,
        #[case] bc: f64,
        #[case] d: f64,
        #[case] target: f64,
    ) {
        let result = log_normal_2d(x, y, mu_x, mu_y, a, bc, d);

        assert_relative_eq!(result, target, epsilon = 1e-12);
    }

    #[rstest]
    #[case(0.7, -1.3, 0.0, 0.0, 2.0, 0.4, 1.0)]
    fn test_test_log_normal_2d_symmetry(
        #[case] x: f64,
        #[case] y: f64,
        #[case] mu_x: f64,
        #[case] mu_y: f64,
        #[case] a: f64,
        #[case] bc: f64,
        #[case] d: f64,
    ) {
        let v1 = log_normal_2d(mu_x + x, mu_y + y, mu_x, mu_y, a, bc, d);
        let v2 = log_normal_2d(mu_x - x, mu_y - y, mu_x, mu_y, a, bc, d);

        assert_relative_eq!(v1, v2, epsilon = 1e-12);
    }

    #[test]
    fn test_test_log_normal_2d_probability_decreases_with_distance() {
        let mu_x = 0.0;
        let mu_y = 0.0;

        let near = log_normal_2d(0.5, 0.5, mu_x, mu_y, 1.0, 0.0, 1.0);
        let far = log_normal_2d(2.0, 2.0, mu_x, mu_y, 1.0, 0.0, 1.0);

        assert!(far < near);
    }
}
