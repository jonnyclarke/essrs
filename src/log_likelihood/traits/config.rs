use ndarray::{ArrayViewMut1, ArrayViewMut2};

use crate::functions::transforms::{
    apply_inplace_multi_column_transform, apply_transform_column,
    logistic::{logistic, logistic_inverse, logistic_log_jacobian_inplace},
    ordseq::{ordinal_sequencer, ordinal_sequencer_inverse},
    softplus::{softplus, softplus_inverse, softplus_log_jacobian_inplace},
    stick_break::{stick_break_1d, stick_break_array2_inplace, stick_break_inverse_array2_inplace},
    tanh::{tanh_inverse, tanh_log_jacobian_inplace, tanh_x},
};

/// Trait providing configuration for parameter transformations
pub trait ParameterTransformerConfig {
    /// Parameter Transformations requires `logistic`
    const HAS_LOGISTIC: bool;
    /// Parameter Transformations requires `softplus`
    const HAS_SOFTPLUS: bool;
    /// Parameter Transformations requires `tanh`
    const HAS_TANH: bool;
    /// Parameter Transformations requires `stick-break` (e.g. mixture model weights)
    const HAS_STICK_BREAK: bool;
    /// Parameter Transformations requires `ordinal-sequencing` (e.g. gaussian mixture model mean values)
    const HAS_ORDSEQ: bool;

    /// Obtain a vector for all columns to which `logistic` should be applied
    #[inline(always)]
    fn logistic(&self) -> &[usize] {
        &[]
    }

    /// Obtain a vector for all columns to which `softplus` should be applied
    #[inline(always)]
    fn softplus(&self) -> &[usize] {
        &[]
    }

    /// Obtain a vector for all columns to which `tanh` transform should be applied
    #[inline(always)]
    fn tanh(&self) -> &[usize] {
        &[]
    }

    /// Obtain a vector of vectors configuring stick break
    /// This applies a weights sum to 1 transform on mixture weights
    #[inline(always)]
    fn stick_break(&self) -> &[usize] {
        &[]
    }

    /// Obtain a vector of vectors configuring ordseq.
    /// Each vector lists indexes of parameters which, in physical space, should be ascending.
    #[inline(always)]
    fn ordseq(&self) -> &[usize] {
        &[]
    }

    // +---------------------------------------+
    // | DEFAULT IMPLEMENTATIONS OF TRANSFORMS |
    // +---------------------------------------+

    // physical-to-internal

    /// Default implementation of `stick-break` transform [physical to internal grid]
    #[inline(always)]
    fn default_stick_break_p2i(&self, arr: &mut ArrayViewMut2<f64>, column_indexes: &[usize]) {
        stick_break_inverse_array2_inplace(arr, column_indexes);
    }

    /// Default implementation of ordinal sequencing transform [physical to internal grid]
    ///
    /// # Errors
    /// * propagates internal function call errors
    #[inline(always)]
    fn default_ordinal_sequencing_p2i(
        &self,
        arr: &mut ArrayViewMut2<f64>,
        column_indexes: &[usize],
    ) -> anyhow::Result<()> {
        for (i, j) in column_indexes.windows(2).rev().map(|w| (w[0], w[1])) {
            apply_inplace_multi_column_transform(
                ordinal_sequencer_inverse,
                arr,
                i, // this value is subtracted
                j, // this value is modified
            )?;
        }

        Ok(())
    }

    /// Default implementation of `logistic` transform [physical to internal grid]
    ///
    /// # Errors
    /// * propagates internal function call errors
    #[inline(always)]
    fn default_logistic_p2i(
        &self,
        arr: &mut ArrayViewMut2<f64>,
        column_indexes: &[usize],
    ) -> anyhow::Result<()> {
        for &i in column_indexes {
            apply_transform_column(logistic_inverse, arr, i)?;
        }

        Ok(())
    }

    /// Default implementation of `softplus` transform [physical to internal grid]
    ///
    /// # Errors
    /// * propagates internal function call errors
    #[inline(always)]
    fn default_softplus_p2i(
        &self,
        arr: &mut ArrayViewMut2<f64>,
        column_indexes: &[usize],
    ) -> anyhow::Result<()> {
        for &i in column_indexes {
            apply_transform_column(softplus_inverse, arr, i)?;
        }

        Ok(())
    }

    /// Default implementation of `tanh` transform [physical to internal grid]
    ///
    /// # Errors
    /// * propagates internal function call errors
    #[inline(always)]
    fn default_tanh_p2i(
        &self,
        arr: &mut ArrayViewMut2<f64>,
        column_indexes: &[usize],
    ) -> anyhow::Result<()> {
        for &i in column_indexes {
            apply_transform_column(tanh_inverse, arr, i)?;
        }

        Ok(())
    }

    // internal-to-physical

    /// Default implementation of `stick-break` transform [internal to physical grid]
    ///
    /// # Errors
    /// * propagates internal function call errors
    #[inline(always)]
    fn default_stick_break_i2p(&self, arr: &mut ArrayViewMut2<f64>, column_indexes: &[usize]) {
        stick_break_array2_inplace(arr, column_indexes);
    }

    /// Default implementation of `ordinal-sequencing` transform [internal to physical grid]
    ///
    /// # Errors
    /// * propagates internal function call errors
    #[inline(always)]
    fn default_ordinal_sequencing_i2p(
        &self,
        arr: &mut ArrayViewMut2<f64>,
        column_indexes: &[usize],
    ) -> anyhow::Result<()> {
        for (i, j) in column_indexes.windows(2).map(|w| (w[0], w[1])) {
            apply_inplace_multi_column_transform(ordinal_sequencer, arr, i, j)?;
        }

        Ok(())
    }

    /// Default implementation of `logistic` transform [internal to physical grid]
    ///
    /// # Errors
    /// * propagates internal function call errors
    #[inline(always)]
    fn default_logistic_i2p(
        &self,
        arr: &mut ArrayViewMut2<f64>,
        column_indexes: &[usize],
    ) -> anyhow::Result<()> {
        for &i in column_indexes {
            apply_transform_column(logistic, arr, i)?;
        }

        Ok(())
    }

    /// Default implementation of `softplus` transform [internal to physical grid]
    ///
    /// # Errors
    /// * propagates internal function call errors
    #[inline(always)]
    fn default_softplus_i2p(
        &self,
        arr: &mut ArrayViewMut2<f64>,
        column_indexes: &[usize],
    ) -> anyhow::Result<()> {
        for &i in column_indexes {
            apply_transform_column(softplus, arr, i)?;
        }

        Ok(())
    }

    /// Default implementation of `tanh` transform [internal to physical grid]
    ///
    /// # Errors
    /// * propagates internal function call errors
    #[inline(always)]
    fn default_tanh_i2p(
        &self,
        arr: &mut ArrayViewMut2<f64>,
        column_indexes: &[usize],
    ) -> anyhow::Result<()> {
        for &i in column_indexes {
            apply_transform_column(tanh_x, arr, i)?;
        }

        Ok(())
    }

    // internal-to-physical -- log jacobian

    /// Default implementation of `stick-break` transform [internal to physical -- log jacobian]
    ///
    /// # Errors
    /// * propagates internal function call errors
    #[inline(always)]
    fn default_stick_break_lj_i2p(
        &self,
        arr: &mut ArrayViewMut1<f64>,
        column_indexes: &[usize],
        lj: &mut f64,
    ) -> anyhow::Result<()> {
        stick_break_1d(arr, column_indexes, lj);

        Ok(())
    }

    /// Default implementation of `ordinal-sequencing` transform [internal to physical -- log jacobian]
    ///
    /// # Errors
    /// * propagates internal function call errors
    #[inline(always)]
    fn default_ordinal_sequencing_lj_i2p(
        &self,
        arr: &mut ArrayViewMut1<f64>,
        column_indexes: &[usize],
    ) -> anyhow::Result<()> {
        for (i, j) in column_indexes.windows(2).map(|w| (w[0], w[1])) {
            ordinal_sequencer(arr[i], &mut arr[j])?
        }
        Ok(())
    }

    /// Default implementation of `logistic` transform [internal to physical -- log jacobian]
    ///
    /// # Errors
    /// * propagates internal function call errors
    #[inline(always)]
    fn default_logistic_lj_i2p(
        &self,
        arr: &mut ArrayViewMut1<f64>,
        column_indexes: &[usize],
        lj: &mut f64,
    ) -> anyhow::Result<()> {
        for &i in column_indexes {
            logistic_log_jacobian_inplace(&mut arr[i], lj)?;
        }

        Ok(())
    }

    /// Default implementation of `softplus` transform [internal to physical -- log jacobian]
    ///
    /// # Errors
    /// * propagates internal function call errors
    #[inline(always)]
    fn default_softplus_lj_i2p(
        &self,
        arr: &mut ArrayViewMut1<f64>,
        column_indexes: &[usize],
        lj: &mut f64,
    ) -> anyhow::Result<()> {
        for &i in column_indexes {
            softplus_log_jacobian_inplace(&mut arr[i], lj)?;
        }

        Ok(())
    }

    /// Default implementation of `tanh` transform [internal to physical -- log jacobian]
    ///
    /// # Errors
    /// * propagates internal function call errors
    #[inline(always)]
    fn default_tanh_lj_i2p(
        &self,
        arr: &mut ArrayViewMut1<f64>,
        column_indexes: &[usize],
        lj: &mut f64,
    ) -> anyhow::Result<()> {
        for &i in column_indexes {
            tanh_log_jacobian_inplace(&mut arr[i], lj)?;
        }

        Ok(())
    }
}
