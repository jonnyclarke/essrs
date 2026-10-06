use std::time::Instant;

use essrs::moves::move_handlers::differential_handler::DifferentialHandler;
use essrs::{
    chains::static_buffer::StaticBuffer, // uses to store the walker positions with each iteration
    ess::EnsembleSliceSampler,           // ensemble slice sampler struct
    log_likelihood::normal_1d::{
        GaussianLl1dDataErrors, // Likelihood model for 1d Gaussian data with errors
        helper_generate_random_gaussian_points, // helper function to generate random gaussian points
    },
};
use ndarray::Array2;
use rand_distr::{Distribution, Normal};
use tracing_subscriber::fmt;

fn generate_cloud(anchor: &[f64], n_points: usize, std_dev: &[f64]) -> Array2<f64> {
    assert_eq!(anchor.len(), std_dev.len());

    let mut rng = rand::rng();
    let dimensions = anchor.len();

    let normals: Vec<Normal<f64>> = std_dev
        .iter()
        .map(|&sigma| Normal::new(0.0, sigma).unwrap())
        .collect();

    let mut points = Array2::<f64>::zeros((n_points, dimensions));

    for mut point in points.outer_iter_mut() {
        for (j, value) in point.iter_mut().enumerate() {
            *value = anchor[j] + normals[j].sample(&mut rng);
        }
    }

    points
}

fn main() -> anyhow::Result<()> {
    fmt()
        .with_timer(fmt::time::UtcTime::rfc_3339())
        .with_level(false)
        .with_target(false)
        .init();

    // Generate 1000 randomised data points
    // Sample a true position and an error. Then convolve position with a re-sampling of the error to get convolvedf data
    const N_DATA_POINTS: usize = 10_000;
    let data = helper_generate_random_gaussian_points(N_DATA_POINTS)?;

    const N_BURN_IN: usize = 500;
    const N_STEPS: usize = 1000; // number of steps for the walkers
    const N_PARAMETERS: usize = 2; // number of parameters in the fit: mean & standard deviation
    const N_WALKERS: usize = 100; // number of walkers used simultaneously NOTE: > 2 x N_PARAMETERS

    let chains = StaticBuffer::new(N_STEPS, N_WALKERS, N_PARAMETERS);
    let move_handler = DifferentialHandler::new();
    let model = GaussianLl1dDataErrors::new(&data);
    let mut ess =
        EnsembleSliceSampler::<StaticBuffer, DifferentialHandler, GaussianLl1dDataErrors>::new(
            N_STEPS,
            N_WALKERS,
            N_PARAMETERS,
            chains,
            move_handler,
            model,
        );

    let start = generate_cloud(&[0.1, 1.0], N_WALKERS, &[0.05, 0.05]);

    let start_time = Instant::now();

    ess.initialise(start.view())?; // compute log-likelihood of initial positions
    ess.run_sampler(N_BURN_IN)?; // run sampler

    let duration = start_time.elapsed();
    println!("sampler has taken {} seconds", duration.as_secs());
    ess.display_parameter_summaries();

    Ok(())
}
