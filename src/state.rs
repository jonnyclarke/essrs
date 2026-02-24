mod save_ll;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

pub struct WalkerState {
    state: Array2<f64>,
    ll: Array1<f64>,
    n: usize,
}

impl WalkerState {
    pub fn new(n_walkers: usize, n_parameters: usize) -> Self {
        Self {
            state: Array2::<f64>::zeros((n_walkers, n_parameters)),
            ll: Array1::<f64>::zeros(n_walkers),
            n: n_walkers,
        }
    }

    pub fn n_walkers(&self) -> usize {
        self.n
    }
}

impl WalkerState {
    pub fn get_state_matrix<'a>(&'a self) -> ArrayView2<'a, f64> {
        self.state.view()
    }

    pub fn get_mut_state_matrix<'a>(&'a mut self) -> ArrayViewMut2<'a, f64> {
        self.state.view_mut()
    }
}

impl WalkerState {
    pub fn get_ll_vector<'a>(&'a self) -> ArrayView1<'a, f64> {
        self.ll.view()
    }

    pub fn get_mut_ll_vector<'a>(&'a mut self) -> ArrayViewMut1<'a, f64> {
        self.ll.view_mut()
    }
}

impl WalkerState {
    pub fn get_ith_state_vector<'a>(&'a self, i: usize) -> ArrayView1<'a, f64> {
        self.state.row(i)
    }

    pub fn get_mut_ith_state_vector(&mut self, i: usize) -> ArrayViewMut1<'_, f64> {
        self.state.row_mut(i)
    }
}

impl WalkerState {
    pub fn get_ith_ll(&self, i: usize) -> f64 {
        self.ll[i]
    }

    pub fn get_mut_ith_ll(&mut self, i: usize) -> &mut f64 {
        &mut self.ll[i]
    }
}

impl WalkerState {
    pub fn panic_on_invalid_ll(&self) {
        if self.ll.iter().any(|x| x.is_nan()) {
            panic!("NaN detected in Log-Likelihood!");
        }

        if self.ll.iter().any(|x| x.is_infinite()) {
            panic!("Infinite detected in Log-Likelihood!");
        }
    }
}

#[cfg(test)]
mod tests {
    use core::f64;

    use ndarray::array;
    use rstest::rstest;

    use super::*;

    fn helper_new() -> WalkerState {
        WalkerState::new(3, 2)
    }

    fn helper_get_mut_state_matrix() -> WalkerState {
        let mut state = helper_new();
        state
            .get_mut_state_matrix()
            .assign(&array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        state
    }

    fn helper_get_mut_ll_vector() -> WalkerState {
        let mut state = helper_get_mut_state_matrix();
        state.get_mut_ll_vector().assign(&array![0.2, 0.3, 0.4]);
        state
    }

    fn helper_get_mut_ith_state_vector() -> WalkerState {
        let mut state = helper_get_mut_ll_vector();
        state.get_mut_ith_state_vector(1).assign(&array![7.0, 8.0]);
        state
    }

    fn helper_get_mut_ith_ll() -> WalkerState {
        let mut state = helper_get_mut_ith_state_vector();
        *state.get_mut_ith_ll(1) = -2.2;
        state
    }

    #[test]
    fn test_n_walker() {
        let state = helper_new();
        assert_eq!(state.n_walkers(), 3);
    }

    #[test]
    fn test_walker_state_new() {
        _ = helper_new();
    }

    #[test]
    fn test_get_mut_state_matrix() {
        _ = helper_get_mut_state_matrix();
    }

    #[test]
    fn test_get_mut_ll_vector() {
        _ = helper_get_mut_ll_vector();
    }

    #[test]
    fn test_get_mut_ith_state_vector() {
        _ = helper_get_mut_ith_state_vector();
    }

    #[test]
    fn test_get_mut_ith_ll() {
        _ = helper_get_mut_ith_ll();
    }

    #[test]
    fn test_get_state_matrix() {
        let state = helper_get_mut_ith_ll();
        assert_eq!(
            state.get_state_matrix(),
            array![[1.0, 2.0], [7.0, 8.0], [5.0, 6.0]]
        )
    }

    #[test]
    fn test_get_ll_vector() {
        let state = helper_get_mut_ith_ll();
        assert_eq!(state.get_ll_vector(), array![0.2, -2.2, 0.4])
    }

    #[test]
    fn test_get_ith_state_vector() {
        let state = helper_get_mut_ith_ll();
        assert_eq!(state.get_ith_state_vector(0), array![1.0, 2.0]);
        assert_eq!(state.get_ith_state_vector(1), array![7.0, 8.0]);
        assert_eq!(state.get_ith_state_vector(2), array![5.0, 6.0]);
    }

    #[test]
    fn test_get_ith_ll() {
        let state = helper_get_mut_ith_ll();
        assert_eq!(state.get_ith_ll(0), 0.2);
        assert_eq!(state.get_ith_ll(1), -2.2);
        assert_eq!(state.get_ith_ll(2), 0.4);
    }

    #[test]
    #[should_panic]
    fn test_panic_on_invalid_ll_nan() {
        let state = WalkerState {
            state: array![[1.0], [2.0]],
            ll: array![f64::NAN, 0.0],
            n: 2,
        };
        state.panic_on_invalid_ll();
    }

    #[rstest]
    #[should_panic]
    #[case(f64::NEG_INFINITY)]
    #[should_panic]
    #[case(f64::INFINITY)]
    fn test_panic_on_invalid_ll_inf(#[case] ll: f64) {
        let state = WalkerState {
            state: array![[1.0], [2.0]],
            ll: array![ll, 0.0],
            n: 2,
        };
        state.panic_on_invalid_ll();
    }
}
