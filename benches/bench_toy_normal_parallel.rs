use anyhow;
use criterion::{Criterion, criterion_group, criterion_main};
use essrs::{
    ess::EnsembleSliceSampler,
    log_likelihood::{f_toy::normal::AnalyticNormalToyLogL, random_normal_points},
    moves::differential_move::DifferentialMove,
};
use ndarray::Axis;

fn bench_analytic_normal_parallel(n_threads: Option<usize>) -> anyhow::Result<()> {
    const TRUE_MU: f64 = 0.0;
    const TRUE_SIGMA: f64 = 1.0;

    const N_SAMPLES: usize = 1000;
    const N_WALKERS: usize = 10;
    const N_PARAMETERS: usize = 1;
    const N_BURN_IN: usize = 100;

    let move_handler = DifferentialMove::new();
    let model = AnalyticNormalToyLogL::new(TRUE_MU, TRUE_SIGMA);

    let mut ess = EnsembleSliceSampler::new(N_SAMPLES, N_WALKERS, N_PARAMETERS, n_threads, None);

    let start = random_normal_points(N_WALKERS, 1.0, 2.0);
    let start = start.insert_axis(Axis(1));

    ess.initialise(&model, start.view()).unwrap();

    ess.run_burn_in(&model, &DifferentialMove::new(), N_BURN_IN)
        .unwrap();

    ess.run_sampler(&model, &move_handler).unwrap(); // run sampler

    Ok(())
}

fn bench_analytic_toy_normal(c: &mut Criterion) {
    let mut group = c.benchmark_group("toy analytic normal");
    group.sample_size(100);

    group.bench_function("sequential", |b| {
        b.iter(|| {
            bench_analytic_normal_parallel(Some(1)).expect("Analytic toy normal has failed . . .");
        })
    });

    group.bench_function("parallel -- 4", |b| {
        b.iter(|| {
            bench_analytic_normal_parallel(Some(4)).expect("Analytic toy normal has failed . . .");
        })
    });

    group.bench_function("parallel -- 8", |b| {
        b.iter(|| {
            bench_analytic_normal_parallel(Some(8)).expect("Analytic toy normal has failed . . .");
        })
    });
    group.finish();
}

criterion_group!(benches, bench_analytic_toy_normal);
criterion_main!(benches);
