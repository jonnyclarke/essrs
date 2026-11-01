//! Analytical toy log-likelihood functions: P(data|theta)

mod mixmod;
mod neals_funnel;
pub mod normal;
mod rosenbrock;

pub use mixmod::DynNcompMixtureModelProbData;
pub use neals_funnel::AnalyticNealsFunnelToyLogL;
pub use rosenbrock::AnalyticRosenbrockToyLogL;
