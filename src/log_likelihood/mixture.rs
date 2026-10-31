// ! Generic implementation of a mixture mdoel
// ! TODO: evaluate feasibility
// !
// ! * would require each function to also return a point-wise array-type of log-likelihoods

mod dynamic;
use std::sync::Arc;

pub use dynamic::DynNcompMixtureModel;
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
    pub fn new(theta: Array1<f64>, components: Vec<Arc<dyn ProbDataGivenThetaModel>>) -> Self {
        assert!(theta.iter().all(|&x| x > 0.0));
        assert!(theta.sum().le(&1.0));
        assert_eq!(theta.len() + 1, components.len());

        let expected = components[0].get_n_parameters();

        assert!(
            components
                .iter()
                .all(|model| { model.get_n_parameters() == expected })
        );

        Self { theta, components }
    }
}

impl LogLikelihoodModel for DynMixtureModelDistributionSampler {
    fn get_n_parameters(&self) -> usize {
        self.components[0].get_n_parameters()
        // let mut n: usize = 0;
        // for c in &self.components {
        //     n += c.get_n_parameters()
        // };

        // n
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
