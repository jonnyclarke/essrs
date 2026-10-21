use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

pub struct WalkerState {
    current: Array2<f64>,
    next: Array2<f64>,

    ll_current: Array1<f64>,
    ll_next: Array1<f64>,

    n: usize,
}

impl WalkerState {
    pub fn new(n_walkers: usize, n_parameters: usize) -> Self {
        Self {
            current: Array2::<f64>::zeros((n_walkers, n_parameters)),
            next: Array2::<f64>::zeros((n_walkers, n_parameters)),

            ll_current: Array1::<f64>::zeros(n_walkers),
            ll_next: Array1::<f64>::zeros(n_walkers),

            n: n_walkers,
        }
    }

    pub fn n_walkers(&self) -> usize {
        self.n
    }
}

impl WalkerState {
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
        std::mem::swap(&mut self.ll_current, &mut self.ll_next);
    }

    // Assignment
    pub fn assign_state_matrix(&mut self, a: ArrayView2<f64>) {
        self.next.assign(&a)
    }

    pub fn assign_ith_state_vector(&mut self, index: usize, a: ArrayView1<f64>) {
        self.next.row_mut(index).assign(&a)
    }

    pub fn assign_ll_vector(&mut self, ll: ArrayView1<f64>) {
        self.ll_next.assign(&ll)
    }

    pub fn assign_ith_ll(&mut self, index: usize, ll: f64) {
        self.ll_next[index] = ll
    }

    // Get views
    pub fn get_state_matrix_view<'a>(&'a self) -> ArrayView2<'a, f64> {
        self.current.view()
    }

    pub fn get_ith_state_vector_view<'a>(&'a self, i: usize) -> ArrayView1<'a, f64> {
        self.current.row(i)
    }

    pub fn get_ll_vector_view<'a>(&'a self) -> ArrayView1<'a, f64> {
        self.ll_current.view()
    }

    pub fn get_ith_ll_value(&self, i: usize) -> f64 {
        self.ll_current[i]
    }
}


#[cfg(test)]
mod tests {
    use ndarray::array;
    use anyhow;

    use super::*;

    use crate::testing::WalkerStateBuilder;

    fn build_test_state() -> WalkerState {
        let state = WalkerStateBuilder::new(3, 2)
            .set_state_matrix(array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            .set_ll_vector(array![0.2, 0.3, 0.4])
            .swap()
            .set_state_matrix(array![[7.0, 8.0], [9.0, 1.0], [2.0, 3.0]])
            .set_ll_vector(array![0.5, 0.6, 0.7])
            .construct();

        state
    }

    #[test]
    fn test_n_walker() {
        let state = build_test_state();
        assert_eq!(state.n_walkers(), 3);
    }

    #[test]
    fn test_state_assignment() -> anyhow::Result<()> {
        let mut state = build_test_state();
        // current
        assert_eq!(
            state.get_state_matrix_view(),
            array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        );
        assert_eq!(
            state.get_ll_vector_view(),
            array![0.2, 0.3, 0.4]
        );

        state.swap();
        // next
        assert_eq!(
            state.get_state_matrix_view(),
            array![[7.0, 8.0], [9.0, 1.0], [2.0, 3.0]]
        );
        assert_eq!(
            state.get_ll_vector_view(),
            array![0.5, 0.6, 0.7]
        );

        Ok(())
    }

    #[test]
    fn test_state_manipulation() -> anyhow::Result<()> {
        let mut state = build_test_state();
        
        state.assign_ith_state_vector(0, array![10.0, 11.0].view());
        state.assign_ith_ll(0, -0.1);

        state.swap();

        assert_eq!(
            state.get_state_matrix_view(),
            array![[10.0, 11.0], [9.0, 1.0], [2.0, 3.0]]
        );
        assert_eq!(
            state.get_ll_vector_view(),
            array![-0.1, 0.6, 0.7]
        );

        assert_eq!(
            state.get_ith_state_vector_view(1),
            array![9.0, 1.0]
        );

        assert_eq!(
            state.get_ith_ll_value(1),
            0.6
        );

        Ok(())
    }

}
