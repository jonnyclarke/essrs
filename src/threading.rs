//! Thread management for parallelised MCMC fitting.

use std::sync::Mutex;

use ndarray::Array1;
use rand::{SeedableRng, rngs::StdRng};
use rayon::{ThreadPool, ThreadPoolBuilder};

pub struct ThreadScratch {
    pub direction: Array1<f64>,
    pub internal: Array1<f64>,
    pub physical: Array1<f64>,
}

impl ThreadScratch {
    pub fn new(n_parameters: usize) -> Self {
        Self {
            direction: Array1::<f64>::zeros(n_parameters),
            internal: Array1::<f64>::zeros(n_parameters),
            physical: Array1::<f64>::zeros(n_parameters),
        }
    }
}

pub struct ThreadWorker {
    pub rng: StdRng,
    pub scratch: ThreadScratch,
}

pub struct Threading {
    pub pool: ThreadPool,
    pub workers: Vec<Mutex<ThreadWorker>>,
    n_threads: usize,
}

impl Threading {
    pub fn new(requested_threads: Option<usize>, n_parameters: usize) -> anyhow::Result<Self> {
        let max_n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let n_threads = match requested_threads {
            Some(x) => x,
            None => max_n_threads,
        };

        anyhow::ensure!(n_threads > 0, "thread count must be positive");
        anyhow::ensure!(
            n_threads <= max_n_threads,
            "requestedthread count exceeds machine capabilities"
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

    pub fn n_threads(&self) -> usize {
        self.n_threads
    }

    pub fn with_worker<R>(&self, worker_index: usize, f: impl FnOnce(&mut ThreadWorker) -> R) -> R {
        let mut worker = self.workers[worker_index]
            .lock()
            .expect("worker mutex poisoned");

        f(&mut worker)
    }
}
