pub mod gaussian;

use ndarray::Array1;

use crate::{FloatExt, data::DataBufferTrait};

pub trait LogLikelihoodModel<T: FloatExt> {
    const N_PARAMETERS: usize;
    const N_DATA_DIMENS: usize;

    fn n_parameters(&self) -> usize;
    fn log_likelihood<DataType: DataBufferTrait<T>>(
        &self,
        data: &DataType,
        parameters: &Array1<T>,
    ) -> T;
    fn jacobian_change_of_variables(&self, _parameters: &Array1<T>) -> T;
}
