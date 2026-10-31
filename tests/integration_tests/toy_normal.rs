use anyhow;
use essrs::{
    ess::EnsembleSliceSampler,
    log_likelihood::f_toy::normal::AnalyticNormalToyLogL, // Likelihood model for 1d Gaussian data with errors
    moves::differential_move::DifferentialMove,
};
use ndarray::array;

#[test]
fn test_analytical_normal() -> anyhow::Result<()> {
    eprintln!("\n");
    const TARGET_MU: f64 = 2.0;
    const TARGET_SIGMA: f64 = 4.0;

    eprintln!("True Mean: {}", TARGET_MU);
    eprintln!("True Standard Deviation: {}", TARGET_SIGMA);

    const N_SAMPLES: usize = 2000;
    const N_PARAMETERS: usize = 1;
    const N_WALKERS: usize = 10;

    let move_handler = DifferentialMove::new();
    let model = AnalyticNormalToyLogL::new(TARGET_MU, TARGET_SIGMA);

    let mut ess = EnsembleSliceSampler::new(N_SAMPLES, N_WALKERS, N_PARAMETERS, None, None);

    let start = array![
        [-0.5],
        [0.0],
        [0.5],
        [1.0],
        [1.5],
        [2.5],
        [3.0],
        [3.5],
        [4.0],
        [4.5],
    ];

    ess.initialise(&model, start.view())?;

    ess.run_burn_in(&model, &DifferentialMove::new(), 2000)
        .unwrap();
    ess.run_sampler(&model, &move_handler).unwrap(); // run sampler

    ess.display_parameter_summaries();

    eprintln!("\n");
    Ok(())
}
