use ndarray::{ArrayView2, Axis};

use crate::chains::{ChainError, static_buffer::StaticBuffer};

impl StaticBuffer {
    pub fn get_final_state(&self) -> Result<ArrayView2<'_, f64>, ChainError> {
        if self.n_stored == 0 {
            return Err(ChainError::EmptyChainBufferError);
        }
        Ok(self.chains.index_axis(Axis(0), self.n_stored - 1))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::testing::make_test_static_buffer;

    #[test]
    fn test_get_final_state() -> anyhow::Result<()> {
        let buffer = make_test_static_buffer();

        let expected_fs = array![[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0],];

        assert_eq!(expected_fs, buffer.get_final_state()?);

        Ok(())
    }
}
