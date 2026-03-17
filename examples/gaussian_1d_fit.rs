use essrs::{
    chains::static_buffer::StaticBuffer,
    ess::{EnsemblSliceSampler, EnsembleSliceSamplerConfig},
    log_likelihood::gaussian_1d_data_err::{
        GaussianLl1dDataErrors, helper_generate_random_gaussian_points,
    },
    moves::MoveHandler,
};
use ndarray::array;
use tracing_subscriber::fmt;

fn main() {
    fmt()
        .with_timer(fmt::time::UtcTime::rfc_3339())
        .with_level(false)
        .with_target(false)
        .init();

    const N_DATA_POINTS: usize = 1_000;
    let data = helper_generate_random_gaussian_points(N_DATA_POINTS);

    const N_STEPS: usize = 500;
    const N_PARAMETERS: usize = 2;
    const N_WALKERS: usize = 12;

    type Config = EnsembleSliceSamplerConfig<N_STEPS, N_WALKERS, N_PARAMETERS, 0, 0, true>;

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

    ess.initialise(&start);
    let move_handler = MoveHandler::new(vec![1.0, 0.0]);
    ess.run_sampler(move_handler);

    ess.display_parameter_summaries()
}
