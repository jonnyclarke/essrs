//! Thread management for parallelised MCMC fitting.

use std::sync::Mutex;

use ndarray::Array1;
use rand::{SeedableRng, rngs::StdRng};
use rayon::{ThreadPool, ThreadPoolBuilder};

/// Set of scratch arrays used within the implementation of 'jump' functions to avoid excessive allocation
pub struct ThreadScratch {
    /// scratch array for computation of direction vector
    pub direction: Array1<f64>,
    /// scratch array for computation of internal
    pub internal: Array1<f64>,
    /// scratch array for computation of parameters into physical units
    pub physical: Array1<f64>,
}

impl ThreadScratch {
    /// Constructor
    ///
    /// # Arguments
    /// * `n_parameters`: Number of parameters being fit; 'jump' functions require scratch arrays of length `n_parameters` in order to store direction and proposals.
    pub fn new(n_parameters: usize) -> Self {
        Self {
            direction: Array1::<f64>::zeros(n_parameters),
            internal: Array1::<f64>::zeros(n_parameters),
            physical: Array1::<f64>::zeros(n_parameters),
        }
    }
}

/// Thread specific struct containing random number generator and scratch arrays
pub struct ThreadWorker {
    /// Random number generator
    pub rng: StdRng,

    /// Scratch arrays required by ensemble slice sampler
    pub scratch: ThreadScratch,
}

/// Object managing per-thread level allocations
pub struct Threading {
    /// Pool of worked availabel for use in parallel processing
    pub pool: ThreadPool,
    /// Worker structs containing random number generators and scratch arrays
    pub workers: Vec<Mutex<ThreadWorker>>,
    n_threads: usize,
}

impl Threading {
    /// Constructor
    ///
    /// # Arguments:
    /// * `requested_threads`: the number of threads desired to be used. If not given will default to using all available cores
    /// * `n_parameters`: the number of parameters being fit. Needed in order to allocate correctly sized scratch arrays
    ///
    /// # Errors
    /// * thread count is not > 0
    /// * requested thread count exceeds machine capabilities
    /// * thread pool has failed to build
    pub fn new(requested_threads: Option<usize>, n_parameters: usize) -> anyhow::Result<Self> {
        let max_n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let n_threads = match requested_threads {
            Some(x) => x,
            None => max_n_threads,
        };

        anyhow::ensure!(n_threads > 0, "thread count must be positive, {n_threads}");
        anyhow::ensure!(
            n_threads <= max_n_threads,
            "requested thread count, {n_threads}, exceeds machine capabilities, {max_n_threads}"
        );

        let pool = ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .thread_name(|i| format!("sampler-{i}"))
            .build()?;

        let workers = (0..n_threads)
            .map(|i| {
                // TODO: make this robust and reproducible
                let seed = rand::random::<u64>() ^ (i as u64);

                Mutex::new(ThreadWorker {
                    rng: StdRng::seed_from_u64(seed),
                    scratch: ThreadScratch::new(n_parameters),
                })
            })
            .collect();

        Ok(Self {
            pool,
            workers,
            n_threads,
        })
    }

    /// Get the number of threads available for use
    pub fn n_threads(&self) -> usize {
        self.n_threads
    }

    /// Wrapper function which passes a locked mutex, containing thread-level rng and scratch arrays, to the internal function which uses them.
    /// The mutex lock is released after the function has concluded and the call goes out of scope.
    ///
    /// # Panics
    /// * panic upon attempt to use a locked mutex
    pub fn with_worker<R>(&self, worker_index: usize, f: impl FnOnce(&mut ThreadWorker) -> R) -> R {
        let mut worker = self.workers[worker_index]
            .lock()
            .expect("worker mutex poisoned");

        f(&mut worker)
    }
}
