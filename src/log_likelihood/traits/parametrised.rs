/// Minimal function required of a log-likelihood function
pub trait Parametrised {
    /// Get the number of parameters required by the model
    fn get_n_parameters(&self) -> usize;
}
