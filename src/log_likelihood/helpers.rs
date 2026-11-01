use anyhow::{Context, ensure};
use ndarray::{Array1, Array2};
use rand::rng;
use rand_distr::{Distribution, LogNormal, Normal};

/// Generate n random normally distributed points.
///
/// # Arguments
/// * `n`: the number of points to generate
/// * `mean`: centroid of the distribution to be sampled
/// * `std_dev`: standard deviation of the distribution to be sampled
///
/// # Returns
/// * array of normally distributed points
///
/// # Errors
/// * passed standard deviation is <=0
/// * sampling from normal distribution
pub fn random_normal_points(n: usize, mean: f64, std_dev: f64) -> anyhow::Result<Array1<f64>> {
    ensure!(
        std_dev.gt(&0.0_f64),
        "invalid standard deviation sigma(={std_dev})<0"
    );

    let normal = Normal::new(mean, std_dev).context("normal distribution sampling has failed")?;

    let mut rng = rng();

    let v: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();

    Ok(Array1::from(v))
}

/// Generate random points for benchmarking
/// TODO: re-write
///
/// # Errors
/// * sampling from normal/lognormal distribution
/// * conversion from a Vec to an Array2 type
pub fn helper_generate_random_gaussian_points(n: usize) -> anyhow::Result<Array2<f64>> {
    let mut data = Vec::<f64>::with_capacity(n * 2);

    let normal_v = Normal::<f64>::new(0.10, 0.80)?;
    let normal_e = LogNormal::<f64>::new(0.00, 0.10)?;

    let sampler = Normal::<f64>::new(0.00, 1.00)?;

    let mut rng = rand::rng();

    for _ in 0..n {
        let mut value = normal_v.sample(&mut rng);
        let sigma = normal_e.sample(&mut rng);

        // this adds convolution with the error into the final mean value
        value += sigma * sampler.sample(&mut rng);

        data.push(value);
        data.push(sigma);
    }

    let output = Array2::from_shape_vec((n, 2), data)?;

    Ok(output)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_random_normal_points() -> anyhow::Result<()> {
        let n = 100;
        let arr = random_normal_points(n, 1.0, 2.0)?;
        assert!(arr.iter().all(|x| x.is_finite()));

        Ok(())
    }

    #[test]
    fn test_helper_generate_random_gaussian_points_shape() -> anyhow::Result<()> {
        let n = 100;
        let arr = helper_generate_random_gaussian_points(n)?;
        assert_eq!(arr.shape(), &[n, 2]);

        Ok(())
    }

    #[test]
    fn test_helper_generate_random_gaussian_points_is_finite() -> anyhow::Result<()> {
        let n = 100;
        let arr = helper_generate_random_gaussian_points(n)?;
        assert!(arr.iter().all(|x| x.is_finite()));

        Ok(())
    }
}
