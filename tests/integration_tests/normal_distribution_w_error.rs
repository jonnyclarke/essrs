use anyhow;
use essrs::{
    chains::static_buffer::StaticBuffer,
    ess::EnsembleSliceSampler,
    log_likelihood::gaussian_1d_data_err::{
        GaussianLl1dDataErrors, helper_generate_random_gaussian_points,
    },
    moves::MoveHandler,
};
use ndarray::array;
use tempfile::NamedTempFile;

#[test]
fn test_normal_distribution_with_errors() -> anyhow::Result<()> {
    let data = helper_generate_random_gaussian_points(100)?;

    const N_STEPS: usize = 200;
    const N_PARAMETERS: usize = 2;
    const N_WALKERS: usize = 12;

    let chains = StaticBuffer::new(N_STEPS, N_WALKERS, N_PARAMETERS);
    let model = GaussianLl1dDataErrors::new(data);

    let mut ess = EnsembleSliceSampler::<StaticBuffer, GaussianLl1dDataErrors>::new(
        N_STEPS,
        N_WALKERS,
        N_PARAMETERS,
        chains,
        model,
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

    ess.initialise(&start)?;

    let move_handler = MoveHandler::default();
    ess.run_sampler(200, move_handler)?;

    let tmp_file = NamedTempFile::new()?;
    let path = tmp_file.path();

    let _ = ess.chains.dump_final_walkers(path);

    let tmp_file = NamedTempFile::new()?;
    let path = tmp_file.path();

    let _ = ess.save_log_likelihood(path);

    ess.display_parameter_summaries();

    Ok(())
}
