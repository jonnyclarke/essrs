// ! Generic implementation of a mixture mdoel
// ! TODO: evaluate feasibility
// !
// ! * would require each function to also return a point-wise array-type of log-likelihoods

use std::sync::Arc;

use anyhow::ensure;
use ndarray::Array1;

use crate::{
    functions::numerical::log_sum_exp,
    log_likelihood::{LogLikelihoodModel, ProbDataGivenThetaModel},
};

pub struct DynMixtureModelDistributionSampler {
    theta: Array1<f64>,
    components: Vec<Arc<dyn ProbDataGivenThetaModel>>,
}

impl DynMixtureModelDistributionSampler {
    pub fn new(
        theta: Array1<f64>,
        components: Vec<Arc<dyn ProbDataGivenThetaModel>>,
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

impl LogLikelihoodModel for DynMixtureModelDistributionSampler {
    fn get_n_parameters(&self) -> usize {
        self.components[0].get_n_parameters()
    }

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
        let _ = DynMixtureModelDistributionSampler::new(
            array![0.5],
            vec![
                Arc::new(AnalyticNormalToyLogL::new(0.0, 1.0)),
                Arc::new(AnalyticNormalToyLogL::new(1.0, 2.0)),
            ],
        )?;

        Ok(())
    }

    #[test]
    fn test_init_failure_theta_empty() -> anyhow::Result<()> {
        let result = DynMixtureModelDistributionSampler::new(
            array![],
            vec![
                Arc::new(AnalyticNormalToyLogL::new(0.0, 1.0)),
                Arc::new(AnalyticNormalToyLogL::new(1.0, 2.0)),
            ],
        );
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_init_failure_theta_too_large() -> anyhow::Result<()> {
        let result = DynMixtureModelDistributionSampler::new(
            array![1.5],
            vec![
                Arc::new(AnalyticNormalToyLogL::new(0.0, 1.0)),
                Arc::new(AnalyticNormalToyLogL::new(1.0, 2.0)),
            ],
        );
        assert!(result.is_err());

        Ok(())
    }

    #[rstest]
    #[case(array![0.2])]
    #[case(array![0.2, 0.1, 0.1])]
    fn test_init_failure_theta_wrong_length(#[case] theta: Array1<f64>) -> anyhow::Result<()> {
        let result = DynMixtureModelDistributionSampler::new(
            theta,
            vec![
                Arc::new(AnalyticNormalToyLogL::new(0.0, 1.0)),
                Arc::new(AnalyticNormalToyLogL::new(1.0, 2.0)),
                Arc::new(AnalyticNormalToyLogL::new(1.0, 2.0)),
            ],
        );
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn theta_init_inconsistent_dimensionallity() -> anyhow::Result<()> {
        let result = DynMixtureModelDistributionSampler::new(
            array![1.0],
            vec![
                Arc::new(AnalyticNormalToyLogL::new(0.0, 1.0)),
                Arc::new(AnalyticNealsFunnelToyLogL::new()),
            ],
        );
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_get_n_parameters() -> anyhow::Result<()> {
        let model = DynMixtureModelDistributionSampler::new(
            array![0.5],
            vec![
                Arc::new(AnalyticNormalToyLogL::new(0.0, 1.0)),
                Arc::new(AnalyticNormalToyLogL::new(1.0, 2.0)),
            ],
        )?;

        assert_eq!(model.get_n_parameters(), 1);

        Ok(())
    }

    #[test]
    fn test_log_likelihood() -> anyhow::Result<()> {
        let model = DynMixtureModelDistributionSampler::new(
            array![0.5],
            vec![
                Arc::new(AnalyticNormalToyLogL::new(0.0, 1.0)),
                Arc::new(AnalyticNormalToyLogL::new(1.0, 2.0)),
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

// pub struct MixtureModelDistributionSampler<L: ProbDataGivenThetaModel> {
//     n_components: usize,
//     n_sub_parameters: usize,
//     theta: Array1<f64>,
//     components: Vec<L>,
// }

// impl<L: ProbDataGivenThetaModel> MixtureModelDistributionSampler<L> {
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

// impl<L: ProbDataGivenThetaModel> LogLikelihoodModel for MixtureModelDistributionSampler<L> {
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
