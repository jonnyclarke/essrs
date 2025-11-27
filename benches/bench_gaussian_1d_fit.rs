use criterion::{Criterion, criterion_group, criterion_main};
use essrs::{
    chains::static_buffer::StaticBuffer,
    data::DataBuffer,
    ess::{EnsemblSliceSampler, EnsembleSliceSamplerConfig},
    log_likelihood::gaussian::GaussianLogLikelihood1D,
};
use ndarray::{Array1, Array2, array};
// use rand::prelude::*;
use rand_distr::{Distribution, Normal};

fn generate_data(n: usize) -> Array2<f32> {
    let normal = Normal::new(0.05, 0.94).unwrap();
    let mut rng = rand::rng();
    let arr = Array1::from((0..n).map(|_| normal.sample(&mut rng)).collect::<Vec<_>>());

    // let std: f32 = arr.std(1.0);
    // println!("\n\n\nSTATISTICS :: mu = {}, sigma = {}, ln-sigma = {}", arr.mean().unwrap(), std, std.ln());
    arr.into_shape_with_order((n, 1)).unwrap()
}

fn fit_gaussian(data: Array2<f32>) -> () {
    const N_STEPS: usize = 100;
    const N_PARAMETERS: usize = 2;
    const N_WALKERS: usize = 12;

    type Config = EnsembleSliceSamplerConfig<N_STEPS, N_WALKERS, N_PARAMETERS, 1, 0, 0, true>;

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

fn bench_gaussian_1d_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_mcmc");
    group.sample_size(10);
    group.bench_function("fit 1D Gaussian 1e6 points", |b| {
        b.iter(|| {
            let data = generate_data(1_000_000);
            fit_gaussian(std::hint::black_box(data));
        })
    });
    group.finish();
}

criterion_group!(benches, bench_gaussian_1d_fit);
criterion_main!(benches);
