#[cfg(test)]
pub mod helpers {

    use ndarray::array;

    use crate::{
        chains::static_buffer::StaticBuffer,
        data::DataBuffer,
        ess::{EnsemblSliceSampler, EnsembleSliceSamplerConfig},
        log_likelihood::gaussian::GaussianLogLikelihood1D,
    };

    type TestConfig = EnsembleSliceSamplerConfig<1, 5, 2, 1, 0, 0, true>;

    pub fn build_test_ess() -> EnsemblSliceSampler<
        f32,
        StaticBuffer<f32, TestConfig>,
        GaussianLogLikelihood1D<f32, TestConfig>,
        TestConfig,
    > {
        let data = array![[0.0_f32]];

        let chains = StaticBuffer::<f32, TestConfig>::new();
        let model = GaussianLogLikelihood1D::<f32, TestConfig>::new();

        let ess = EnsemblSliceSampler::<
            f32,
            StaticBuffer<f32, TestConfig>,
            GaussianLogLikelihood1D<f32, TestConfig>,
            TestConfig,
        >::new(DataBuffer::<f32, TestConfig>::new(data), chains, model);

        ess
    }
}
