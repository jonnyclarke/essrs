use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

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
    
    pub fn get_state_matrix_view<'a>(&'a self) -> ArrayView2<'a, f64> {
        self.current.view()
    }

}

impl WalkerState {

    pub fn get_mut_state_matrix<'a>(&'a mut self) -> ArrayViewMut2<'a, f64> {
        self.next.view_mut()
    }

}

impl WalkerState {
    pub fn get_ll_vector<'a>(&'a self) -> ArrayView1<'a, f64> {
        self.ll_current.view()
    }

    pub fn get_mut_ll_vector<'a>(&'a mut self) -> ArrayViewMut1<'a, f64> {
        self.ll_next.view_mut()
    }
}

impl WalkerState {
    pub fn get_ith_state_vector<'a>(&'a self, i: usize) -> ArrayView1<'a, f64> {
        self.current.row(i)
    }

    pub fn get_mut_ith_state_vector(&mut self, i: usize) -> ArrayViewMut1<'_, f64> {
        self.next.row_mut(i)
    }
}

impl WalkerState {
    pub fn get_ith_ll(&self, i: usize) -> f64 {
        self.ll_current[i]
    }

    pub fn get_mut_ith_ll(&mut self, i: usize) -> &mut f64 {
        &mut self.ll_next[i]
    }
}


// #[cfg(test)]
// mod tests {
//     use ndarray::array;
//     use anyhow;

//     use super::*;


//     fn helper_new() -> WalkerState {
//         WalkerState::new(3, 2)
//     }

//     fn helper_get_mut_state_matrix() -> WalkerState {
//         let mut state = helper_new();
//         state
//             .get_mut_state_matrix()
//             .assign(&array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
//         state
//     }

//     fn helper_get_mut_ll_vector() -> WalkerState {
//         let mut state = helper_get_mut_state_matrix();
//         state.get_mut_ll_vector().assign(&array![0.2, 0.3, 0.4]);
//         state.swap();
//         state
//     }

//     fn helper_get_mut_ith_state_vector() -> WalkerState {
//         let mut state = helper_get_mut_ll_vector();
//         state.get_mut_ith_state_vector(1).assign(&array![7.0, 8.0]);
//         state
//     }

//     fn helper_get_mut_ith_ll() -> WalkerState {
//         let mut state = helper_get_mut_ith_state_vector();
//         *state.get_mut_ith_ll(1) = -2.2;
//         state
//     }

//     #[test]
//     fn test_n_walker() {
//         let state = helper_new();
//         assert_eq!(state.n_walkers(), 3);
//     }

//     #[test]
//     fn test_walker_state_new() {
//         _ = helper_new();
//     }

//     #[test]
//     fn test_get_mut_state_matrix() {
//         _ = helper_get_mut_state_matrix();
//     }

//     #[test]
//     fn test_get_mut_ll_vector() {
//         _ = helper_get_mut_ll_vector();
//     }

//     #[test]
//     fn test_get_mut_ith_state_vector() {
//         _ = helper_get_mut_ith_state_vector();
//     }

//     #[test]
//     fn test_get_mut_ith_ll() {
//         _ = helper_get_mut_ith_ll();
//     }

//     #[test]
//     fn test_get_state_matrix_view() -> anyhow::Result<()> {
//         let mut state = helper_get_mut_ith_ll();
//         // current
//         assert_eq!(
//             state.get_state_matrix_view(),
//             array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
//         );
//         assert_eq!(
//             state.get_ll_vector(),
//             array![0.0, 0.0, 0.0]
//         );
//         // next
//         assert_eq!(
//             state.get_mut_state_matrix(),
//             array![[1.0, 2.0], [7.0, 8.0], [5.0, 6.0]]
//         );
//         assert_eq!(
//             state.get_mut_ll_vector(),
//             array![0.0, 0.0, 0.0]
//         );

//         Ok(())
//     }

//     #[test]
//     fn test_get_ll_vector() {
//         let state = helper_get_mut_ith_ll();
//         assert_eq!(state.get_ll_vector(), array![0.2, -2.2, 0.4])
//     }

//     #[test]
//     fn test_get_ith_state_vector() {
//         let state = helper_get_mut_ith_ll();
//         assert_eq!(state.get_ith_state_vector(0), array![1.0, 2.0]);
//         assert_eq!(state.get_ith_state_vector(1), array![7.0, 8.0]);
//         assert_eq!(state.get_ith_state_vector(2), array![5.0, 6.0]);
//     }

//     #[test]
//     fn test_get_ith_ll() {
//         let state = helper_get_mut_ith_ll();
//         assert_eq!(state.get_ith_ll(0), 0.2);
//         assert_eq!(state.get_ith_ll(1), -2.2);
//         assert_eq!(state.get_ith_ll(2), 0.4);
//     }
// }
