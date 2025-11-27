use core::f32;
use std::ops::{AddAssign, SubAssign};

use ndarray::ScalarOperand;
use num_traits::Float;
use rand_distr::uniform::SampleUniform;

pub trait FloatExt:
    Float
    + ScalarOperand
    + SampleUniform
    + SubAssign
    + AddAssign
    + std::iter::Sum<Self>
    + std::iter::Sum<&'static Self>
{
    fn my_pi() -> Self;
    fn my_min_positive() -> Self;

    fn get_neg_one() -> Self;
    fn get_zero() -> Self;
    fn get_one() -> Self;
    fn get_two() -> Self;
}

impl FloatExt for f32 {
    fn my_pi() -> Self {
        std::f32::consts::PI
    }
    fn my_min_positive() -> Self {
        f32::MIN_POSITIVE
    }

    fn get_neg_one() -> Self {
        -1.0
    }
    fn get_zero() -> Self {
        0.0
    }
    fn get_one() -> Self {
        1.0
    }
    fn get_two() -> Self {
        2.0
    }
}

impl FloatExt for f64 {
    fn my_pi() -> Self {
        std::f64::consts::PI
    }
    fn my_min_positive() -> Self {
        f64::MIN_POSITIVE
    }

    fn get_neg_one() -> Self {
        -1.0
    }
    fn get_zero() -> Self {
        0.0
    }
    fn get_one() -> Self {
        1.0
    }
    fn get_two() -> Self {
        2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_constants() {
        assert_eq!(f32::my_pi(), std::f32::consts::PI);
        assert_eq!(f32::my_min_positive(), f32::MIN_POSITIVE);
        assert_eq!(f32::get_neg_one(), -1.0_f32);
        assert_eq!(f32::get_zero(), 0.0_f32);
        assert_eq!(f32::get_one(), 1.0_f32);
        assert_eq!(f32::get_two(), 2.0_f32);
    }

    #[test]
    fn test_f64_constants() {
        assert_eq!(f64::my_pi(), std::f64::consts::PI);
        assert_eq!(f64::my_min_positive(), f64::MIN_POSITIVE);
        assert_eq!(f64::get_neg_one(), -1.0_f64);
        assert_eq!(f64::get_zero(), 0.0_f64);
        assert_eq!(f64::get_one(), 1.0_f64);
        assert_eq!(f64::get_two(), 2.0_f64);
    }
}
