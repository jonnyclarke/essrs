use anyhow;
use essrs::{
    chains::static_buffer::StaticBuffer,
    ess::EnsembleSliceSampler,
    log_likelihood::toy::rosenbrock::AnalyticRosenbrockToyLogL, // Likelihood model for 1d Gaussian data with errors
    moves::differential_move::DifferentialMove,
};
use ndarray::array;

#[test]
fn test_analytical_normal() -> anyhow::Result<()> {
    eprintln!("\n");

    const TARGET_A: f64 = 1.0;
    const TARGET_B: f64 = 3.0;

    eprintln!("True A: {}", TARGET_A);
    eprintln!("True B: {}", TARGET_B);

    const N_STEPS: usize = 2000;
    const N_PARAMETERS: usize = 2;
    const N_WALKERS: usize = 10;

    let chains = StaticBuffer::new(N_STEPS, N_WALKERS, N_PARAMETERS);
    let move_handler = DifferentialMove::new();
    let model = AnalyticRosenbrockToyLogL::new(TARGET_A, TARGET_B);

    let mut ess =
        EnsembleSliceSampler::<StaticBuffer, DifferentialMove, AnalyticRosenbrockToyLogL>::new(
            N_STEPS,
            N_WALKERS,
            N_PARAMETERS,
            chains,
            move_handler,
            model,
        );

    let start = array![
        [0.1, 0.4],
        [-0.2, 0.6],
        [-0.9, -0.7],
        [0.1, -0.9],
        [0.6, -0.6],
        [0.4, -0.5],
        [-0.3, 0.7],
        [-0.8, 0.2],
        [0.0, 0.6],
        [-0.5, 0.1]
    ];

    ess.initialise(start.view())?;

    ess.run_burn_in(1000, &DifferentialMove::new()).unwrap();
    ess.run_sampler().unwrap(); // run sampler

    ess.display_parameter_summaries();

    eprintln!("\n");
    Ok(())
}
