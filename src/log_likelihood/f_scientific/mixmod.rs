use std::{any::Any, sync::Arc};

use anyhow::bail;
use ndarray::{Zip, s};

use crate::{
    functions::{
        numerical::log_sum_exp,
        transforms::{
            apply_transform_column,
            logistic::{logistic, logistic_inverse, logistic_log_jacobian_inplace},
        },
    },
    log_likelihood::{LogLikelihoodModel, ProbThetaGivenDataModel},
};

pub struct DynNcompMixtureModel {
    components: Vec<Arc<dyn ProbThetaGivenDataModel>>,
    n_comp: usize,
    n_theta: usize,
}

impl DynNcompMixtureModel {
    pub fn new(components: Vec<Arc<dyn ProbThetaGivenDataModel>>) -> anyhow::Result<Self> {
        let t = components[0].as_ref().type_id();

        for c in components.iter() {
            if t.ne(&c.as_ref().type_id()) {
                bail!("mixture model components are not identical")
            }
        }

        let n_comp = components.len();
        let n_theta = n_comp - 1;
        Ok(Self {
            components,
            n_comp,
            n_theta,
        })
    }
}

impl LogLikelihoodModel for DynNcompMixtureModel {
    fn get_n_parameters(&self) -> usize {
        let mut n: usize = self.n_theta;

        for c in &self.components {
            n += c.get_n_parameters();
        }

        n
    }

    fn columnar_transform_physical_to_internal(
        &self,
        physical: ndarray::prelude::ArrayView2<f64>,
        mut internal: ndarray::prelude::ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        internal.assign(&physical);
        let mut l = self.n_theta - self.components[0].get_n_parameters();
        let mut r = self.n_theta;

        for i in 0..r {
            apply_transform_column(logistic_inverse, &mut internal, i)?;
        }

        for c in &self.components {
            let n = c.get_n_parameters();
            l += n;
            r += n;

            c.columnar_transform_physical_to_internal(
                physical.slice(s![.., l..r]),
                internal.slice_mut(s![.., l..r]),
            )?
        }

        Ok(())
    }

    fn columnar_transform_internal_to_physical(
        &self,
        internal: ndarray::prelude::ArrayView2<f64>,
        mut physical: ndarray::prelude::ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        physical.assign(&internal);
        let mut l = self.n_theta - self.components[0].get_n_parameters();
        let mut r = self.n_theta;

        for i in 0..r {
            apply_transform_column(logistic, &mut physical, i)?;
        }

        for c in &self.components {
            let n = c.get_n_parameters();
            l += n;
            r += n;

            c.columnar_transform_internal_to_physical(
                internal.slice(s![.., l..r]),
                physical.slice_mut(s![.., l..r]),
            )?
        }

        Ok(())
    }

    fn convert_internal_to_physical_tracking_ll_warp(
        &self,
        log_jacobian: &mut f64,
        internal: ndarray::prelude::ArrayView1<f64>,
        physical: &mut ndarray::prelude::ArrayViewMut1<f64>,
    ) -> anyhow::Result<()> {
        physical.assign(&internal);
        let mut l = self.n_theta - self.components[0].get_n_parameters();
        let mut r = self.n_theta;

        for i in 0..r {
            logistic_log_jacobian_inplace(&mut physical[i], log_jacobian)?;
        }

        for c in &self.components {
            let n = c.get_n_parameters();
            l += n;
            r += n;
            c.convert_internal_to_physical_tracking_ll_warp(
                log_jacobian,
                internal.slice(s![l..r]),
                &mut physical.slice_mut(s![l..r]),
            )?
        }

        Ok(())
    }

    fn log_likelihood(&self, parameters: ndarray::prelude::ArrayView1<f64>) -> anyhow::Result<f64> {
        let thetas = parameters.slice(s![..self.n_theta]);

        let l = self.get_n_parameters() - self.components[self.n_comp - 1].get_n_parameters();
        let r = self.get_n_parameters();

        let mut ll_i = self.components[self.n_comp - 1]
            .log_likelihood_pointwise(parameters.slice(s![l..r]))?
            + (1.0_f64 - thetas.sum()).ln();

        if ll_i.iter().any(|x| x.is_infinite()) {
            return Ok(f64::NEG_INFINITY);
        }
        let mut l: usize = self.n_theta;
        for i in 0..self.n_comp - 1 {
            let r: usize = l + self.components[i].get_n_parameters();
            // TODO: this currently requires extensive allocation -- trade-off for dynamic usability?
            let ll_j = self.components[i].log_likelihood_pointwise(parameters.slice(s![l..r]))?
                + thetas[i].ln();
            l = r;

            if ll_j.iter().any(|x| x.is_infinite()) {
                return Ok(f64::NEG_INFINITY);
            }

            Zip::from(ll_i.view_mut())
                .and(ll_j.view())
                .for_each(|ll_cumulative, &ll_kth| {
                    *ll_cumulative =
                        log_sum_exp(*ll_cumulative, ll_kth).expect("failure in log_sum_exp");
                });
        }

        Ok(ll_i.sum())
    }
}
