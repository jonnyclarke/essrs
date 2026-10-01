//! Module providing the different move (jump) options and handlers for managing combinations of move options

pub mod differential_move;
pub mod dummy_move;
pub mod gaussian_move;

use anyhow;
use rand::{Rng, RngCore};
use thiserror::Error;

use crate::{
    log_likelihood::LogLikelihoodModel,
    moves::{differential_move::DifferentialMove, gaussian_move::GaussianMove},
    state::WalkerState,
};

pub trait EnsembleMove {
    fn get_likelihood_floor(&self, rng: &mut dyn RngCore) -> f64 {
        rng.random_range(0.0_f64..1.0_f64).ln()
    }
    fn jump(
        &self,
        rng: &mut dyn RngCore,
        log_likelihood_model: &dyn LogLikelihoodModel,
        state_i: &WalkerState,
        state_j: &mut WalkerState,
    ) -> anyhow::Result<()>;
}

pub struct MoveHandler {
    vec_move: Vec<Box<dyn EnsembleMove>>,
    cumulative: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum MoveHandlerError {
    #[error("at least one move is required")]
    EmptyMoves,

    #[error("at least one probability is required")]
    EmptyProbabilities,

    #[error("number of moves ({moves}) does not match number of probabilities ({probabilities})")]
    LengthMismatch { moves: usize, probabilities: usize },

    #[error(
        "probability at index {index} is invalid: {value}; probabilities must be finite and non-negative"
    )]
    InvalidProbability { index: usize, value: f64 },

    #[error("sum of probabilities must be greater than zero")]
    ZeroTotalProbability,

    #[error("unable to index against cumulative probability weights")]
    InvalidCumulativeIndexing,
}

impl MoveHandler {
    pub fn new(
        vec_move: Vec<Box<dyn EnsembleMove>>,
        probability: Vec<f64>,
    ) -> Result<Self, MoveHandlerError> {
        if vec_move.is_empty() {
            return Err(MoveHandlerError::EmptyMoves);
        }

        if probability.is_empty() {
            return Err(MoveHandlerError::EmptyProbabilities);
        }

        if probability.len() != vec_move.len() {
            return Err(MoveHandlerError::LengthMismatch {
                moves: vec_move.len(),
                probabilities: probability.len(),
            });
        }

        for (index, &p) in probability.iter().enumerate() {
            if !p.is_finite() || p < 0.0 {
                return Err(MoveHandlerError::InvalidProbability { index, value: p });
            }
        }

        let total: f64 = probability.iter().sum();

        if !total.is_finite() || total <= 0.0 {
            return Err(MoveHandlerError::ZeroTotalProbability);
        }

        let cumulative = probability
            .iter()
            .scan(0.0, |sum, &p| {
                *sum += p / total;
                Some(*sum)
            })
            .collect::<Vec<_>>();

        Ok(Self {
            vec_move,
            cumulative,
        })
    }

    pub fn choose_move(&self, r: f64) -> Result<usize, MoveHandlerError> {
        self.cumulative
            .iter()
            .position(|&c| r < c)
            .ok_or(MoveHandlerError::InvalidCumulativeIndexing)
    }

    pub fn distribute_jump(
        &self,
        rng: &mut impl Rng,
        log_likelihood_model: &dyn LogLikelihoodModel,
        state_i: &WalkerState,
        state_j: &mut WalkerState,
    ) -> anyhow::Result<()> {
        let r: f64 = rng.random();

        let i = self.choose_move(r)?;

        self.vec_move[i].jump(rng, log_likelihood_model, state_i, state_j)?;

        Ok(())
    }
}

const DIFFERENTIAL_MOVE_WEIGHT: f64 = 0.9;
const GAUSSIAN_MOVE_WEIGHT: f64 = 0.1;

impl Default for MoveHandler {
    fn default() -> Self {
        Self::new(
            vec![
                Box::new(DifferentialMove::new()),
                Box::new(GaussianMove::new()),
            ],
            vec![DIFFERENTIAL_MOVE_WEIGHT, GAUSSIAN_MOVE_WEIGHT],
        )
        .expect("built-in default move configuration is invalid")
    }
}

#[cfg(test)]
mod tests {

    use rand::Rng;

    use crate::moves::{DIFFERENTIAL_MOVE_WEIGHT, GAUSSIAN_MOVE_WEIGHT, MoveHandler};

    #[test]
    fn move_distributions() -> anyhow::Result<()> {
        let mut rng = rand::rng();

        let move_handler = MoveHandler::default();

        let mut counts = [0usize; 2];

        const N_ITERATIONS: usize = 1_000_000;
        let n_iterations_f64: f64 = N_ITERATIONS as f64;

        for _ in 0..N_ITERATIONS {
            let r: f64 = rng.random();
            let i = move_handler.choose_move(r)?;
            counts[i] += 1;
        }

        let p0 = counts[0] as f64 / n_iterations_f64;
        let p1 = counts[1] as f64 / n_iterations_f64;

        assert!((p0 - DIFFERENTIAL_MOVE_WEIGHT).abs() < 0.002);
        assert!((p1 - GAUSSIAN_MOVE_WEIGHT).abs() < 0.002);

        Ok(())
    }
}

pub struct MoveHandlerDifferentialGaussian {
    i_gaussian_move: u64, // this is the number of differential moves per gaussian move

    move_differential: DifferentialMove, // differential move struct
    move_gaussian: GaussianMove,         // gaussian move struct
    i: u64,
}

impl MoveHandlerDifferentialGaussian {
    pub fn new(n_diff_per_gauss: u64) -> Self {
        Self {
            i_gaussian_move: n_diff_per_gauss + 1,
            move_differential: DifferentialMove::new(),
            move_gaussian: GaussianMove::new(),
            i: 1,
        }
    }

    pub fn choose_move(&mut self) -> bool {
        let use_gaussian_move = self.i.is_multiple_of(self.i_gaussian_move);
        self.i += 1;

        use_gaussian_move
    }

    pub fn distribute_jump(
        &mut self,
        rng: &mut impl Rng,
        log_likelihood_model: &dyn LogLikelihoodModel,
        state_i: &WalkerState,
        state_j: &mut WalkerState,
    ) -> anyhow::Result<()> {
        let use_gaussian_move: bool = self.choose_move();

        match use_gaussian_move {
            true => self
                .move_gaussian
                .jump(rng, log_likelihood_model, state_i, state_j)?,
            false => self
                .move_differential
                .jump(rng, log_likelihood_model, state_i, state_j)?,
        }

        Ok(())
    }
}

const DEFAULT_N_DIFFERENTIAL_PER_GAUSSIAN: u64 = 9;

impl Default for MoveHandlerDifferentialGaussian {
    fn default() -> Self {
        Self::new(DEFAULT_N_DIFFERENTIAL_PER_GAUSSIAN)
    }
}

#[cfg(test)]
mod test_move_handler_differential_gaussian {

    use crate::moves::MoveHandlerDifferentialGaussian;

    #[test]
    fn move_distributions() {
        let mut move_handler = MoveHandlerDifferentialGaussian::default();

        let mut counts = [0usize; 2];

        const N_ITERATIONS: usize = 10_000;

        for _ in 0..N_ITERATIONS {
            let use_gaussian_move = move_handler.choose_move();
            match use_gaussian_move {
                true => counts[1] += 1,
                false => counts[0] += 1,
            }
        }

        // this time the selection is deterministic
        assert!(counts[0] == 9_000);
        assert!(counts[1] == 1_000);
    }
}
