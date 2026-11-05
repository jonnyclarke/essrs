// ! Generic implementation of a mixture mdoel

use std::sync::Arc;

use anyhow::ensure;
use ndarray::Array1;

use crate::{
    functions::numerical::log_sum_exp,
    log_likelihood::{
        LogLikelihoodModel, ParameterTransformer, Parametrised, ToyModelSamplerComponent,
    },
};

/// Wrapper to enable generic mixture models -- python binding
///
/// # Fields:
/// * theta: Array of component weights. All values should be positive. They should sum to < 1. The length should be {#components - 1} as weight of final component is (1 - sum(theta)).
/// * components: Vector of log-likelihood models. These should be toy 'distribution sampler' models that test P(data|theta).
pub struct DynToyMixModelDistributionSampler {
    theta: Array1<f64>,
    components: Vec<Arc<dyn ToyModelSamplerComponent>>,
}

impl DynToyMixModelDistributionSampler {
    /// Constructor
    ///
    /// # Errors
    /// * the len of passed `theta` array is 0
    /// * any of the passed `theta` values are <=0
    /// * the sum of `theta` array exceeds 1
    /// * the len of `theta` does not equal the len of `components` - 1
    /// * model `components` do not share identical dimensionality
    pub fn new(
        theta: Array1<f64>,
        components: Vec<Arc<dyn ToyModelSamplerComponent>>,
    ) -> anyhow::Result<Self> {
        ensure!(
            theta.len().ge(&1_usize),
            "Dyn Mixture Model -- theta array is empty"
        );

        ensure!(
            theta.iter().all(|&x| x > 0.0),
            "Dyn Mixture Model -- theta values are not all positive:\n\t{theta:#?}"
        );
        ensure!(
            theta.sum().le(&1.0),
            "Dyn Mixture Model -- theta values sum ={} > 1",
            theta.sum()
        );
        ensure!(
            (theta.len() + 1).eq(&components.len()),
            "Dyn Mixture Model -- n theta ({}) should equal k -1 components ({})",
            theta.len(),
            components.len()
        );

        let expected = components[0].get_n_parameters();

        ensure!(
            components
                .iter()
                .all(|model| { model.get_n_parameters() == expected }),
            "Dyn Mixture Model -- components do not share identical dimensionality:\n\t{:#?}",
            components
                .iter()
                .map(|m| m.get_n_parameters())
                .collect::<Vec<usize>>()
        );

        Ok(Self { theta, components })
    }
}

impl Parametrised for DynToyMixModelDistributionSampler {
    fn get_n_parameters(&self) -> usize {
        self.components[0].get_n_parameters()
    }
}

impl ParameterTransformer for DynToyMixModelDistributionSampler {}

impl LogLikelihoodModel for DynToyMixModelDistributionSampler {
    fn log_likelihood(&self, parameters: ndarray::prelude::ArrayView1<f64>) -> anyhow::Result<f64> {
        let n = self.theta.len() + 1;

        let mut ll = (1.0_f64 - self.theta.sum()).ln()
            + self.components[n - 1].log_likelihood(parameters)?;

        for i in 0..n - 1 {
            ll = log_sum_exp(
                ll,
                self.theta[i].ln() + self.components[i].log_likelihood(parameters)?,
            )?
        }

        Ok(ll)
    }
}

#[cfg(test)]
mod tests {

    use ndarray::array;
    use rstest::rstest;

    use super::*;
    use crate::{
        functions::likelihood::log_normalised_gaussian,
        log_likelihood::f_toy::{AnalyticNealsFunnelToyLogL, normal::AnalyticNormalToyLogL},
    };

    #[test]
    fn test_initialisation() -> anyhow::Result<()> {
        let _ = DynToyMixModelDistributionSampler::new(
            array![0.5],
            vec![
                Arc::new(AnalyticNormalToyLogL::new(0.0, 1.0)?),
                Arc::new(AnalyticNormalToyLogL::new(1.0, 2.0)?),
            ],
        )?;

        Ok(())
    }

    #[test]
    fn test_init_failure_theta_empty() -> anyhow::Result<()> {
        let result = DynToyMixModelDistributionSampler::new(
            array![],
            vec![
                Arc::new(AnalyticNormalToyLogL::new(0.0, 1.0)?),
                Arc::new(AnalyticNormalToyLogL::new(1.0, 2.0)?),
            ],
        );
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_init_failure_theta_too_large() -> anyhow::Result<()> {
        let result = DynToyMixModelDistributionSampler::new(
            array![1.5],
            vec![
                Arc::new(AnalyticNormalToyLogL::new(0.0, 1.0)?),
                Arc::new(AnalyticNormalToyLogL::new(1.0, 2.0)?),
            ],
        );
        assert!(result.is_err());

        Ok(())
    }

    #[rstest]
    #[case(array![0.2])]
    #[case(array![0.2, 0.1, 0.1])]
    fn test_init_failure_theta_wrong_length(#[case] theta: Array1<f64>) -> anyhow::Result<()> {
        let result = DynToyMixModelDistributionSampler::new(
            theta,
            vec![
                Arc::new(AnalyticNormalToyLogL::new(0.0, 1.0)?),
                Arc::new(AnalyticNormalToyLogL::new(1.0, 2.0)?),
                Arc::new(AnalyticNormalToyLogL::new(1.0, 2.0)?),
            ],
        );
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn theta_init_inconsistent_dimensionallity() -> anyhow::Result<()> {
        let result = DynToyMixModelDistributionSampler::new(
            array![1.0],
            vec![
                Arc::new(AnalyticNormalToyLogL::new(0.0, 1.0)?),
                Arc::new(AnalyticNealsFunnelToyLogL::new()),
            ],
        );
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_get_n_parameters() -> anyhow::Result<()> {
        let model = DynToyMixModelDistributionSampler::new(
            array![0.5],
            vec![
                Arc::new(AnalyticNormalToyLogL::new(0.0, 1.0)?),
                Arc::new(AnalyticNormalToyLogL::new(1.0, 2.0)?),
            ],
        )?;

        assert_eq!(model.get_n_parameters(), 1);

        Ok(())
    }

    #[test]
    fn test_log_likelihood() -> anyhow::Result<()> {
        let model = DynToyMixModelDistributionSampler::new(
            array![0.5],
            vec![
                Arc::new(AnalyticNormalToyLogL::new(0.0, 1.0)?),
                Arc::new(AnalyticNormalToyLogL::new(1.0, 2.0)?),
            ],
        )?;

        let physical = array![1.0];

        assert_eq!(
            model.log_likelihood(physical.view())?,
            log_sum_exp(
                0.5_f64.ln() + log_normalised_gaussian(1.0, 0.0, 1.0),
                0.5_f64.ln() + log_normalised_gaussian(1.0, 1.0, 2.0)
            )?
        );

        Ok(())
    }
}

// pub struct MixtureModelDistributionSampler<L: ToyModelSamplerComponent> {
//     n_components: usize,
//     n_sub_parameters: usize,
//     theta: Array1<f64>,
//     components: Vec<L>,
// }

// impl<L: ToyModelSamplerComponent> MixtureModelDistributionSampler<L> {
//     pub fn new(n_components: usize, theta: Array1<f64>, components: Vec<L>) -> Self {
//         assert_eq!(n_components, theta.len());
//         assert_eq!(n_components, components.len());
//         Self {
//             n_components,
//             n_sub_parameters: components[0].get_n_parameters(),

//             theta,
//             components,
//         }
//     }
// }

// impl<L: ToyModelSamplerComponent> LogLikelihoodModel for MixtureModelDistributionSampler<L> {
//     fn get_n_parameters(&self) -> usize {
//         self.n_components * self.n_sub_parameters
//     }

//     fn log_likelihood(&self, parameters: ndarray::prelude::ArrayView1<f64>) -> anyhow::Result<f64> {

//         let mut ll = (1.0 - self.theta.sum()).ln() + self.components[self.n_components - 1].log_likelihood(parameters)?;

//         for i in 0..self.n_components - 1 {
//             ll = log_sum_exp(
//                 ll,
//                 self.theta[i].ln() + self.components[i].log_likelihood(parameters)?
//             )?
//         }

//         Ok(ll)
//     }
// }
