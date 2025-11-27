use std::marker::PhantomData;

use ndarray::{Array2, ArrayView1};

use crate::{FloatExt, ess::EnsembleSliceSamplerConfigTrait};

pub trait DataBufferTrait<T: FloatExt> {
    fn n_dimensions(&self) -> usize;
    fn n_data_points(&self) -> usize;
    fn get_dimension<'a>(&'a self, i: usize) -> ArrayView1<'a, T>;
}

pub struct DataBuffer<T: FloatExt, CONFIG: EnsembleSliceSamplerConfigTrait> {
    data: Array2<T>,
    _config: PhantomData<CONFIG>,
}

impl<T: FloatExt, CONFIG: EnsembleSliceSamplerConfigTrait> DataBuffer<T, CONFIG> {
    pub fn new(data: Array2<T>) -> Self {
        let shape = data.shape();
        assert_eq!(shape[1], CONFIG::N_DATA_DIMENS);

        Self {
            data,
            _config: PhantomData,
        }
    }

    #[cfg(test)]
    pub fn test() -> Self {
        Self {
            data: Array2::<T>::zeros((1, CONFIG::N_DATA_DIMENS)),
            _config: PhantomData,
        }
    }
}

impl<T: FloatExt, CONFIG: EnsembleSliceSamplerConfigTrait> DataBufferTrait<T>
    for DataBuffer<T, CONFIG>
{
    fn n_data_points(&self) -> usize {
        self.data.shape()[0]
    }

    fn n_dimensions(&self) -> usize {
        CONFIG::N_DATA_DIMENS
    }

    fn get_dimension<'a>(&'a self, i: usize) -> ArrayView1<'a, T> {
        self.data.column(i)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::{data::DataBuffer, ess::EnsembleSliceSamplerConfig};

    // --- Mock Config ---
    type TestConfig = EnsembleSliceSamplerConfig<10, 2, 2, 3, 1, 1, true>;

    #[test]
    fn test_data_buffer_creation() {
        // 2 data points, 3 dimensions
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let buffer = DataBuffer::<f64, TestConfig>::new(data);

        assert_eq!(buffer.n_data_points(), 2);
        assert_eq!(buffer.n_dimensions(), 3);
    }

    #[test]
    fn test_get_dimension() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let buffer = DataBuffer::<f64, TestConfig>::new(data);

        let dim0 = buffer.get_dimension(0); // first column
        let dim1 = buffer.get_dimension(1); // second column
        let dim2 = buffer.get_dimension(2); // third column

        assert_eq!(dim0.to_vec(), vec![1.0, 4.0]);
        assert_eq!(dim1.to_vec(), vec![2.0, 5.0]);
        assert_eq!(dim2.to_vec(), vec![3.0, 6.0]);
    }

    #[test]
    fn test_test_constructor() {
        let buffer = DataBuffer::<f64, TestConfig>::test();
        assert_eq!(buffer.n_data_points(), 1);
        assert_eq!(buffer.n_dimensions(), 3);
        // The test constructor should return zeros
        for i in 0..3 {
            let dim = buffer.get_dimension(i);
            for &val in dim.iter() {
                assert_eq!(val, 0.0);
            }
        }
    }
}
