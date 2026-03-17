use criterion::{Criterion, criterion_group, criterion_main};
use essrs::{
    chains::static_buffer::StaticBuffer,
    ess::{EnsemblSliceSampler, EnsembleSliceSamplerConfig},
    log_likelihood::gaussian_1d_data_err::{
        GaussianLl1dDataErrors, helper_generate_random_gaussian_points,
    },
    moves::MoveHandler,
};
use ndarray::{Array2, array};

fn fit_gaussian(data: Array2<f64>) -> () {
    const N_STEPS: usize = 100;
    const N_PARAMETERS: usize = 2;
    const N_WALKERS: usize = 12;

    type Config = EnsembleSliceSamplerConfig<N_STEPS, N_WALKERS, N_PARAMETERS, 1, 0, true>;

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

    let move_handler = MoveHandler::default();
    ess.run_sampler(move_handler);
}

fn bench_gaussian_1d_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_mcmc");
    group.sample_size(10);
    group.bench_function("fit 1D Gaussian 1e6 points", |b| {
        b.iter(|| {
            let data = helper_generate_random_gaussian_points(1_000);
            fit_gaussian(std::hint::black_box(data));
        })
    });
    group.finish();
}

criterion_group!(benches, bench_gaussian_1d_fit);
criterion_main!(benches);
