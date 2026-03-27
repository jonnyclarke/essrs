use essrs::{
    chains::static_buffer::StaticBuffer, // uses to store the walker positions with each iteration
    ess::{EnsemblSliceSampler, EnsembleSliceSamplerConfig}, // ensemble slice sampler struct
    log_likelihood::gaussian_1d_data_err::{
        GaussianLl1dDataErrors, // Likelihood model for 1d Gaussian data with errors
        helper_generate_random_gaussian_points, // helper function to generate random gaussian points
    },
    moves::MoveHandler, // struct defining which moves to use
};
use ndarray::array;
use tracing_subscriber::fmt;

fn main() {
    fmt()
        .with_timer(fmt::time::UtcTime::rfc_3339())
        .with_level(false)
        .with_target(false)
        .init();

    // Generate 1000 randomised data points
    // Sample a true position and an error. Then convolve position with a re-sampling of the error to get convolvedf data
    const N_DATA_POINTS: usize = 1_000;
    let data = helper_generate_random_gaussian_points(N_DATA_POINTS);

    const N_STEPS: usize = 500; // number of steps for the walkers
    const N_PARAMETERS: usize = 2; // number of parameters in the fit: mean & standard deviation
    const N_WALKERS: usize = 12; // number of walkers used simultaneously NOTE: > 2 x N_PARAMETERS

    // Build config object which ensures all components are consistent
    type Config = EnsembleSliceSamplerConfig<N_STEPS, N_WALKERS, N_PARAMETERS>;

    let chains = StaticBuffer::<Config>::new();
    let model = GaussianLl1dDataErrors::new(data);
    let mut ess = EnsemblSliceSampler::<StaticBuffer<Config>, GaussianLl1dDataErrors, Config>::new(
        chains, model,
    );

    let start = array![
        [0.19, 0.001],
        [0.20, -0.32],
        [0.21, 0.33],
        [0.39, -0.34],
        [0.40, 0.35],
        [0.41, 0.006],
        [-0.19, 0.37],
        [-0.20, 0.008],
        [-0.21, -0.39],
        [-0.39, 0.0011],
        [-0.40, 0.312],
        [-0.41, -0.313],
    ];

    ess.initialise(&start); // compute log-likelihood of initial positions
    let move_handler = MoveHandler::default(); // use default move setup 90% differential + 10% gaussian 
    ess.run_sampler(move_handler); // run sampler

    ess.display_parameter_summaries()
}
