// use criterion::{Criterion, criterion_group, criterion_main};
// use essrs::{
//     log_likelihood::LogLikelihoodModel,
//     moves::{MoveHandler, dummy_move::DummyMove},
//     state::WalkerState,
// };
// use ndarray::ArrayView1;
// use rand::{SeedableRng, rngs::SmallRng};

// fn bench_move_handler(c: &mut Criterion) {
//     let mut group = c.benchmark_group("Move handler");

//     let mut rng = SmallRng::seed_from_u64(42);

//     let move_handler = MoveHandler::new(
//         vec![Box::new(DummyMove::new()), Box::new(DummyMove::new())],
//         vec![0.9, 1.0],
//     )
//     .expect("Move handler 'new' call has failed . . .");

//     struct DummyModel;

//     impl LogLikelihoodModel for DummyModel {
//         fn log_likelihood(&self, _: ArrayView1<f64>) -> anyhow::Result<f64> {
//             Ok(0.0)
//         }
//     }

//     let state_i = WalkerState::new(1, 1);
//     let mut state_j = WalkerState::new(1, 1);

//     group.sample_size(1000);
//     group.bench_function("Benchmark move handler with dummy move", |b| {
//         b.iter(|| move_handler.distribute_jump(&mut rng, &DummyModel, &state_i, &mut state_j))
//     });
//     group.finish();
// }

// criterion_group!(benches, bench_move_handler);
// criterion_main!(benches);
