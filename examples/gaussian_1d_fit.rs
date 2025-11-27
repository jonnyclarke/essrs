use essrs::{
    chains::static_buffer::StaticBuffer,
    data::DataBuffer,
    ess::{EnsemblSliceSampler, EnsembleSliceSamplerConfig},
    log_likelihood::gaussian::GaussianLogLikelihood1D,
};
use ndarray::{Array1, array};
use rand_distr::{Distribution, Normal};
use tracing_subscriber::fmt;

fn main() {
    fmt()
        .with_timer(fmt::time::UtcTime::rfc_3339())
        .with_level(false)
        .with_target(false)
        .init();

    const N_DATA_POINTS: usize = 1_000_000;

    let normal = Normal::new(0.05, 0.94).unwrap();
    let mut rng = rand::rng();
    let arr = Array1::from(
        (0..N_DATA_POINTS)
            .map(|_| normal.sample(&mut rng))
            .collect::<Vec<_>>(),
    );

    // let std: f32 = arr.std(1.0);
    // println!("\n\n\nSTATISTICS :: mu = {}, sigma = {}, ln-sigma = {}", arr.mean().unwrap(), std, std.ln());
    let data = arr.into_shape_with_order((N_DATA_POINTS, 1)).unwrap();

    const N_STEPS: usize = 100;
    const N_PARAMETERS: usize = 2;
    const N_WALKERS: usize = 12;

    const N_DATA_DIMENS: usize = 1;

    type Config =
        EnsembleSliceSamplerConfig<N_STEPS, N_WALKERS, N_PARAMETERS, N_DATA_DIMENS, 0, 0, true>;

    let chains = StaticBuffer::<f32, Config>::new();
    let model = GaussianLogLikelihood1D::<f32, Config>::new();

    let mut ess = EnsemblSliceSampler::<
        f32,
        StaticBuffer<f32, Config>,
        GaussianLogLikelihood1D<f32, Config>,
        Config,
    >::new(DataBuffer::<f32, Config>::new(data), chains, model);

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
    ess.run_sampler();
}
