use ndarray::Array2;
use rand_distr::{Distribution, LogNormal, Normal};

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
