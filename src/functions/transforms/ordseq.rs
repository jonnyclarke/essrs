// TODO: error implementation for these functions.

use crate::functions::transforms::TransformError;

/// This is a method used for achieving ordered parameters, e.g. in gaussian mixture modelling
pub fn ordinal_sequencer(x0: f64, x1: &mut f64) -> Result<(), TransformError> {
    *x1 += x0;
    Ok(())
}

pub fn ordinal_sequencer_inverse(x0: f64, x1_prime: &mut f64) -> Result<(), TransformError> {
    *x1_prime -= x0;
    Ok(())
}

#[cfg(test)]
mod tests {

    use anyhow;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(1.0, 2.0, 3.0)]
    fn test_ordinal_sequencer(
        #[case] x0: f64,
        #[case] mut x: f64,
        #[case] t: f64,
    ) -> anyhow::Result<()> {
        ordinal_sequencer(x0, &mut x)?;

        assert_eq!(x, t);
        Ok(())
    }

    #[rstest]
    #[case(1.0, 2.0, 1.0)]
    fn test_ordinal_sequencer_inverse(
        #[case] x0: f64,
        #[case] mut x: f64,
        #[case] t: f64,
    ) -> anyhow::Result<()> {
        ordinal_sequencer_inverse(x0, &mut x)?;

        assert_eq!(x, t);
        Ok(())
    }
}
