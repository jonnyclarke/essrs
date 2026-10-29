use anyhow;
use essrs::{
    ess::EnsembleSliceSampler,
    log_likelihood::f_scientific::GaussianLl1dDataErrors, // Likelihood model for 1d Gaussian data with errors
    log_likelihood::helper_generate_random_gaussian_points, // helper function to generate random gaussian points
    moves::differential_move::DifferentialMove,
};
use ndarray::array;

#[test]
fn test_normal_distribution_with_errors() -> anyhow::Result<()> {
    eprintln!("\n");
    let data = helper_generate_random_gaussian_points(100)?;

    const N_SAMPLES: usize = 200;
    const N_PARAMETERS: usize = 2;
    const N_WALKERS: usize = 12;

    let move_handler = DifferentialMove::new();
    let model = GaussianLl1dDataErrors::new(&data);

    let mut ess = EnsembleSliceSampler::new(N_SAMPLES, N_WALKERS, N_PARAMETERS, None, None);

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

    ess.initialise(&model, start.view())?;

    ess.run_burn_in(&model, &DifferentialMove::new(), 200)
        .unwrap();
    ess.run_sampler(&model, &move_handler).unwrap(); // run sampler

    ess.display_parameter_summaries();

    eprintln!("\n");
    Ok(())
}
