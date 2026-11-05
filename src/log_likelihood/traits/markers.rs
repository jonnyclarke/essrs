use crate::log_likelihood::LogLikelihoodModel;

/// Trait used to annotate models which are inferring P(x|theta) rather than P(theta|x)
///
/// The critical difference is that, rather than being passed parameter values to infer from data,
/// the model stores parameter values and we infer samples from the distribution.
///
/// Such a model is far less complex.
/// A P(theta|x) model generates a log-likelihood for each `x_i` which needs to be handled n context of a mixture.
/// This system simply returns a log-likelihood for that sample of distribution space.
pub trait ToyModelSamplerComponent: LogLikelihoodModel {}
