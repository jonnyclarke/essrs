//! Analytical toy log-likelihood functions: P(data|theta)

mod beta;
mod mixmod;
mod neals_funnel;
pub mod normal;
mod rosenbrock;

pub use beta::AnalyticBetaDistToyLogL;
pub use mixmod::DynToyMixModelDistributionSampler;
pub use neals_funnel::AnalyticNealsFunnelToyLogL;
pub use rosenbrock::AnalyticRosenbrockToyLogL;
