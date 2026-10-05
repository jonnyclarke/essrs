use essrs::{
    chains::static_buffer::StaticBuffer, // uses to store the walker positions with each iteration
    ess::EnsembleSliceSampler,           // ensemble slice sampler struct
    log_likelihood::normal_1d::{
        GaussianLl1dDataErrors, // Likelihood model for 1d Gaussian data with errors
        helper_generate_random_gaussian_points, // helper function to generate random gaussian points
    },
    moves::MoveHandler, // struct defining which moves to use
};
use ndarray::array;
use tracing_subscriber::fmt;

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

    const N_STEPS: usize = 500; // number of steps for the walkers
    const N_PARAMETERS: usize = 2; // number of parameters in the fit: mean & standard deviation
    const N_WALKERS: usize = 12; // number of walkers used simultaneously NOTE: > 2 x N_PARAMETERS

    let chains = StaticBuffer::new(N_STEPS, N_WALKERS, N_PARAMETERS);
    let model = GaussianLl1dDataErrors::new(data);
    let mut ess = EnsembleSliceSampler::<StaticBuffer, GaussianLl1dDataErrors>::new(
        N_STEPS,
        N_WALKERS,
        N_PARAMETERS,
        chains,
        model,
    );

    let start = array![
        [0.102, 0.87],
        [0.103, 0.76],
        [0.101, 0.90],
        [0.999, 0.65],
        [0.994, 0.81],
        [0.108, 1.01],
        [0.997, 0.72],
        [0.995, 0.79],
        [0.993, 0.67],
        [0.102, 0.51],
        [0.998, 1.12],
        [0.101, 0.31],
    ];

    ess.initialise(&start)?; // compute log-likelihood of initial positions
    let move_handler = MoveHandler::default(); // use default move setup 90% differential + 10% gaussian 
    ess.run_sampler(100, move_handler)?; // run sampler

    ess.display_parameter_summaries();

    Ok(())
}
