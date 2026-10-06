use crate::{
    log_likelihood::WrappedLogLikelihoodModel,
    moves::{DifferentialMove, EnsembleMove, EnsembleMoveHandler, MoveType},
};

pub struct DifferentialHandler {
    move_differential: DifferentialMove,
}

impl DifferentialHandler {
    pub fn new() -> Self {
        Self {
            move_differential: DifferentialMove::new(),
        }
    }
}

impl Default for DifferentialHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl EnsembleMoveHandler for DifferentialHandler {
    fn choose_move(&self, _iteration: usize) -> MoveType {
        MoveType::Differential
    }

    fn distribute_jump<L: WrappedLogLikelihoodModel>(
        &mut self,
        _iteration: usize,
        model: &L,
        state_i: &crate::state::WalkerState,
        state_j: &mut crate::state::WalkerState,
    ) -> anyhow::Result<()> {
        self.move_differential.jump(model, state_i, state_j)?;

        Ok(())
    }
}
