use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

use crate::log_likelihood::{ParameterTransformerConfig, Parametrised};

/// Trait providing parameter transform functions
pub trait ParameterTransformer: Parametrised {
    /// Transform physical phase space to internal
    /// Used when ingesting initial conditions provided by the user
    ///
    /// The user is responsible for handling parameter ordering when implementing this function.
    /// If no transform is needed use the default implementation which handles book keeping of internal data structures.
    ///
    /// # Arguments
    /// * `physical`: initial conditions provided with shape [`N_walkers`, `N_parameters`]
    /// * `internal`: mutable array onto which the transformed parameters will be assigned
    ///
    /// # Errors
    /// * failure of user-defined (overwriting) transform function
    fn columnar_transform_physical_to_internal(
        &self,
        physical: ArrayView2<f64>,
        mut internal: ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        internal.assign(&physical);

        Ok(())
    }

    /// Transform internal phase space to physical
    /// Used when storing an iteration into the chain storage
    ///
    /// The user is responsible for handling parameter ordering when implementing this function.
    /// If no transform is needed use the default implementation which handles book keeping of internal data structures.
    ///
    /// # Arguments
    /// * `internal`: internal walker state of shape [`N_walkers`, `N_parameters`]
    /// * `physical`: mutable array onto which the transformed parameters will be assigned
    ///
    /// # Errors
    /// * failure of user-defined (overwriting) transformation function
    fn columnar_transform_internal_to_physical(
        &self,
        internal: ArrayView2<f64>,
        mut physical: ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        physical.assign(&internal);

        Ok(())
    }

    /// Transform parameter vector from internal to physical phase space and compute log-likelihood term from log-jacobian
    /// This function is used within the MCMC process to convert proposals back to physical space for log-likelihood function
    ///
    /// # Arguments
    /// * `log_jacobian`: mutable reference which should be modified according to log-likelihood warp
    /// * `internal`: array view of proposal parameters in internal phase space
    /// * `physical`: mutable array reference for assignment of proposal parameters in physical phase space
    ///
    /// # Errors
    /// * failure of user-defined (overwriting) conversion function
    fn convert_internal_to_physical_tracking_ll_warp(
        &self,
        log_jacobian: &mut f64,
        internal: ArrayView1<f64>,
        physical: &mut ArrayViewMut1<f64>,
    ) -> anyhow::Result<()> {
        *log_jacobian = 0.0;
        physical.assign(&internal);

        Ok(())
    }
}

impl<T: Parametrised + ParameterTransformerConfig> ParameterTransformer for T {
    fn columnar_transform_physical_to_internal(
        &self,
        physical: ArrayView2<f64>,
        mut internal: ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        internal.assign(&physical);

        if T::HAS_STICK_BREAK {
            self.default_stick_break_p2i(&mut internal, self.stick_break())
        }

        if T::HAS_ORDSEQ {
            self.default_ordinal_sequencing_p2i(&mut internal, self.ordseq())?
        }

        if T::HAS_LOGISTIC {
            self.default_logistic_p2i(&mut internal, self.logistic())?
        }

        if T::HAS_SOFTPLUS {
            self.default_softplus_p2i(&mut internal, self.softplus())?
        }

        if T::HAS_TANH {
            self.default_tanh_p2i(&mut internal, self.tanh())?
        }

        Ok(())
    }

    fn columnar_transform_internal_to_physical(
        &self,
        internal: ArrayView2<f64>,
        mut physical: ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        physical.assign(&internal);

        if T::HAS_LOGISTIC {
            self.default_logistic_i2p(&mut physical, self.logistic())?
        }

        if T::HAS_TANH {
            self.default_tanh_i2p(&mut physical, self.tanh())?;
        }

        if T::HAS_SOFTPLUS {
            self.default_softplus_i2p(&mut physical, self.softplus())?;
        }

        if T::HAS_STICK_BREAK {
            self.default_stick_break_i2p(&mut physical, self.stick_break())
        }

        if T::HAS_ORDSEQ {
            self.default_ordinal_sequencing_i2p(&mut physical, self.ordseq())?
        }

        Ok(())
    }

    fn convert_internal_to_physical_tracking_ll_warp(
        &self,
        log_jacobian: &mut f64,
        internal: ArrayView1<f64>,
        physical: &mut ArrayViewMut1<f64>,
    ) -> anyhow::Result<()> {
        physical.assign(&internal);

        if T::HAS_LOGISTIC {
            self.default_logistic_lj_i2p(physical, self.logistic(), log_jacobian)?;
        }

        if T::HAS_TANH {
            self.default_tanh_lj_i2p(physical, self.tanh(), log_jacobian)?;
        }

        if T::HAS_SOFTPLUS {
            self.default_softplus_lj_i2p(physical, self.softplus(), log_jacobian)?;
        }

        if T::HAS_STICK_BREAK {
            self.default_stick_break_lj_i2p(physical, self.logistic(), log_jacobian)?;
        }

        if T::HAS_ORDSEQ {
            self.default_ordinal_sequencing_lj_i2p(physical, self.ordseq())?;
        }

        Ok(())
    }
}
