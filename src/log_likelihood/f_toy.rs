mod neals_funnel;
mod normal_nd;
mod normal;
mod rosenbrock;

pub use neals_funnel::AnalyticNealsFunnelToyLogL;
pub use normal::{
    AnalyticNormalToyLogL,
    AnalyticNormalMixtureOneD,
    BuildAnalyticNormalMixtureOneD,
};
pub use rosenbrock::AnalyticRosenbrockToyLogL;
pub use normal_nd::{
    AnalyticNormalNDimensional,
    BuildAnalyticNormalNDimensional,
};
