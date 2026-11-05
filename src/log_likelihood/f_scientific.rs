//! Scientific style log-likelihood functions: P(theta|data)

mod normal_1d;
mod normal_2d_mix_err_deconv;

pub use normal_1d::{GaussianLl1dDataErrors, Normal1DimMixErrDeconv};
pub use normal_2d_mix_err_deconv::GaussianMixModel2DimUnDeConv;
