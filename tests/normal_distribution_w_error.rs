#[cfg(test)]
mod test {

    use essrs::{
        chains::static_buffer::StaticBuffer,
        ess::{EnsemblSliceSampler, EnsembleSliceSamplerConfig},
        log_likelihood::gaussian_1d_data_err::{
            GaussianLl1dDataErrors, helper_generate_random_gaussian_points,
        },
        moves::MoveHandler,
    };
    use ndarray::array;
    use tempfile::NamedTempFile;

    #[test]
    fn test() {
        let data = helper_generate_random_gaussian_points(50);

        const N_STEPS: usize = 100;
        const N_PARAMETERS: usize = 2;
        const N_WALKERS: usize = 12;

        type Config = EnsembleSliceSamplerConfig<N_STEPS, N_WALKERS, N_PARAMETERS>;

        let chains = StaticBuffer::<Config>::new();
        let model = GaussianLl1dDataErrors::new(data);

        let mut ess =
            EnsemblSliceSampler::<StaticBuffer<Config>, GaussianLl1dDataErrors, Config>::new(
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

        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path();

        let _ = ess.chains.dump_final_walkers(path);

        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path();

        let _ = ess.save_log_likelihood(path);

        ess.display_parameter_summaries()
    }
}
