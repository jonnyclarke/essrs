use std::hint::black_box;

use rand::RngCore;

use crate::{log_likelihood::LogLikelihoodModel, moves::EnsembleMove, state::WalkerState};

pub struct DummyMove {}

impl DummyMove {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for DummyMove {
    fn default() -> Self {
        Self::new()
    }
}

impl EnsembleMove for DummyMove {
    #[inline(never)]
    fn jump(
        &self,
        _rng: &mut dyn RngCore,
        _model: &dyn LogLikelihoodModel,
        _state_i: &WalkerState,
        _state_j: &mut WalkerState,
    ) {
        black_box(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dummy_move_can_be_constructed() {
        let _ = DummyMove::new();
        let _ = DummyMove::default();
    }

    // #[test]
    // fn dummy_move_can_jump() {
    //     let dummy = DummyMove::default();

    //     // Replace these with however you normally construct these
    //     // in your project.
    //     let mut rng = rand::thread_rng();
    //     let model = /* your test LogLikelihoodModel */;
    //     let state_i = /* your test WalkerState */;
    //     let mut state_j = /* your test WalkerState */;

    //     dummy.jump(
    //         &mut rng,
    //         &model,
    //         &state_i,
    //         &mut state_j,
    //     );
    // }
}
