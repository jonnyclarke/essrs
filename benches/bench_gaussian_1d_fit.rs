use criterion::{Criterion, criterion_group, criterion_main};
use essrs::{
    chains::static_buffer::StaticBuffer,
    ess::{EnsembleSliceSampler, EnsembleSliceSamplerConfig},
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

    type Config = EnsembleSliceSamplerConfig<N_STEPS, N_WALKERS, N_PARAMETERS>;

    let chains = StaticBuffer::<Config>::new();
    let model = GaussianLl1dDataErrors::new(data);

    let mut ess = EnsembleSliceSampler::<StaticBuffer<Config>, GaussianLl1dDataErrors, Config>::new(
        chains, model,
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

    ess.initialise(&start);

    let move_handler = MoveHandler::default();
    ess.run_sampler(0, move_handler);
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
