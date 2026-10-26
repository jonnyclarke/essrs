//! Hybrid move considering one secondary move for every (N-1) primary moves.

pub enum WhichMove {
    Primary,
    Secondary,
}

use crate::{log_likelihood::LogLikelihoodModel, moves::EnsembleMove, state::WalkerState};

pub struct HybridOneInNMove<P: EnsembleMove, S: EnsembleMove> {
    primary: P,
    secondary: S,
    one_in_n: usize,
}

impl<P: EnsembleMove, S: EnsembleMove> HybridOneInNMove<P, S> {
    pub fn new(primary: P, secondary: S, one_in_n: usize) -> Self {
        Self {
            primary,
            secondary,
            one_in_n,
        }
    }

    pub fn choose_move(&self, iteration: usize) -> bool {
        iteration.is_multiple_of(self.one_in_n)
    }
}

impl<P: EnsembleMove, S: EnsembleMove> EnsembleMove for HybridOneInNMove<P, S> {
    fn jump<L: LogLikelihoodModel>(
        &self,
        iteration: usize,
        model: &L,
        state: &mut WalkerState,
    ) -> anyhow::Result<()> {
        match self.choose_move(iteration) {
            false => self.primary.jump(iteration, model, state)?,
            true => self.secondary.jump(iteration, model, state)?,
        }

        Ok(())
    }
}

#[cfg(test)]
mod test_move_handler_diff_diffconv {

    use super::*;
    use crate::moves::{diffconv_move::DiffConvMove, differential_move::DifferentialMove};

    #[test]
    fn move_distributions() {
        let diff_move = DifferentialMove::new();
        let diff_conv_move = DiffConvMove::new();
        let move_handler = HybridOneInNMove::new(diff_move, diff_conv_move, 5);

        let mut counts = [0usize; 2];

        const N_ITERATIONS: usize = 10_000;

        for iteration in 0..N_ITERATIONS {
            let which_move = move_handler.choose_move(iteration + 1);
            match which_move {
                true => counts[1] += 1,
                false => counts[0] += 1,
            }
        }

        // this time the selection is deterministic
        assert!(counts[0] == 8_000);
        assert!(counts[1] == 2_000);
    }
}
