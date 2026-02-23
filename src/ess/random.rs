use rand::{self, Rng};

use crate::{
    FloatExt,
    chains::ChainBuffer,
    ess::{EnsemblSliceSampler, EnsembleSliceSamplerConfigTrait},
    log_likelihood::LogLikelihoodModel,
};

impl<
    T: FloatExt,
    CHAINS: ChainBuffer<T>,
    MODEL: LogLikelihoodModel<T>,
    CONFIG: EnsembleSliceSamplerConfigTrait,
> EnsemblSliceSampler<T, CHAINS, MODEL, CONFIG>
{
    pub fn get_rn_not(&mut self, not_this: usize) -> usize {
        let mut random_number: usize = self.rng.random_range(0..CONFIG::N_WALKERS - 1);

        if random_number >= not_this {
            random_number += 1
        }

        random_number
    }

    pub fn get_rn_not_or(&mut self, not_this: usize, or_this: usize) -> usize {
        let (low, high) = if not_this < or_this {
            (not_this, or_this)
        } else {
            (or_this, not_this)
        };
        let mut random_number: usize = self.rng.random_range(0..CONFIG::N_WALKERS - 2);

        if random_number >= low {
            random_number += 1
        }

        if random_number >= high {
            random_number += 1
        }

        random_number
    }
}

#[cfg(test)]
mod tests {

    // use ndarray::{Array1, array};

    use crate::{
        chains::static_buffer::StaticBuffer,
        data::DataBuffer,
        ess::{EnsemblSliceSampler, EnsembleSliceSamplerConfig},
        log_likelihood::gaussian::GaussianLogLikelihood1D,
    };

    const N_DATA_DIMENSIONS: usize = 1;
    const N_WALKERS: usize = 3;

    #[test]
    fn test_gaussian() {
        type Config = EnsembleSliceSamplerConfig<1, N_WALKERS, 4, N_DATA_DIMENSIONS, 1, 1, true>;
        let chains = StaticBuffer::<f32, Config>::new();
        let model = GaussianLogLikelihood1D::<f32, Config>::new();

        let mut ess = EnsemblSliceSampler::<
            f32,
            StaticBuffer<f32, Config>,
            GaussianLogLikelihood1D<f32, Config>,
            Config,
        >::new(DataBuffer::<f32, Config>::test(), chains, model);

        let i: usize = 0;
        let j: usize = ess.get_rn_not(i);
        let k: usize = ess.get_rn_not_or(i, j);

        assert!(i != j);
        assert!(j != k);
        assert!(k != i);

        let i: usize = 0;
        let j: usize = 1;
        let k: usize = ess.get_rn_not_or(i, j);

        assert!(k == 2);

        let i: usize = 0;
        let j: usize = 1;
        let k: usize = ess.get_rn_not_or(j, i);

        assert!(k == 2);
    }
}
