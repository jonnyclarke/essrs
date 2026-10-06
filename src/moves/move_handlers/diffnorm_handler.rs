use crate::{
    moves::{
        DifferentialMove, EnsembleMove, EnsembleMoveHandler, MoveType, WrappedLogLikelihoodModel,
        slice_sampler::gaussian_move::GaussianMove,
    },
    state::WalkerState,
};

const DEFAULT_N_MOVE_PER_NORMAL: usize = 10;

pub struct DiffNormHandler {
    rand: rand::rngs::ThreadRng,

    move_differential: DifferentialMove, // differential move struct
    move_gaussian: GaussianMove,         // gaussian move struct

    n_move_per_norm: usize,
}

impl DiffNormHandler {
    pub fn new(n_move_per_norm: usize) -> Self {
        Self {
            rand: rand::rng(),
            move_differential: DifferentialMove::new(),
            move_gaussian: GaussianMove::new(),

            n_move_per_norm,
        }
    }
}

impl Default for DiffNormHandler {
    fn default() -> Self {
        Self::new(DEFAULT_N_MOVE_PER_NORMAL)
    }
}

impl EnsembleMoveHandler for DiffNormHandler {
    fn choose_move(&self, iteration: usize) -> MoveType {
        match iteration.is_multiple_of(self.n_move_per_norm) {
            true => MoveType::Gaussian,
            false => MoveType::Differential,
        }
    }

    fn distribute_jump<L: WrappedLogLikelihoodModel>(
        &mut self,
        iteration: usize,
        model: &L,
        state_i: &WalkerState,
        state_j: &mut WalkerState,
    ) -> anyhow::Result<()> {
        let switch = self.choose_move(iteration);
        match switch {
            MoveType::Gaussian => {
                self.move_gaussian
                    .jump(&mut self.rand, model, state_i, state_j)?
            }
            MoveType::Differential => {
                self.move_differential
                    .jump(&mut self.rand, model, state_i, state_j)?
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod test_move_handler_differential_gaussian {

    use super::*;

    #[test]
    fn move_distributions() {
        let move_handler = DiffNormHandler::default();

        let mut counts = [0usize; 2];

        const N_ITERATIONS: usize = 10_000;

        for iteration in 0..N_ITERATIONS {
            let use_gaussian_move = move_handler.choose_move(iteration + 1);
            match use_gaussian_move {
                MoveType::Gaussian => counts[1] += 1,
                MoveType::Differential => counts[0] += 1,
            }
        }

        // this time the selection is deterministic
        assert!(counts[0] == 9_000);
        assert!(counts[1] == 1_000);
    }
}
