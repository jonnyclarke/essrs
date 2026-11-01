mod mixmod;
mod normal_1d_err_deconv;
mod normal_2d_mix_err_deconv;

pub use mixmod::DynNcompMixtureModel;
pub use normal_1d_err_deconv::GaussianLl1dDataErrors;
pub use normal_2d_mix_err_deconv::GaussianMixModel2DimUnDeConv;
