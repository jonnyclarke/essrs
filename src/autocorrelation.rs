// use rustfft::{FftPlanner, num_complex::Complex};
// use std::f64;
// use rand_distr::Distribution;

// /// Compute the autocorrelation function using the naive O(N^2) approach.
// fn autocorrelation_naive(x: &[f64]) -> Vec<f64> {
//     let n = x.len();
//     let mean: f64 = x.iter().sum::<f64>() / n as f64;

//     let mut acf = vec![0.0; n];
//     let mut c0 = 0.0;

//     // Compute C(0)
//     for &v in x.iter() {
//         c0 += (v - mean) * (v - mean);
//     }

//     // Compute C(k) for k = 0..n-1
//     for k in 0..n {
//         let mut sum = 0.0;
//         for t in 0..(n - k) {
//             sum += (x[t] - mean) * (x[t + k] - mean);
//         }
//         acf[k] = sum / c0;
//     }

//     acf
// }

// fn autocorrelation_fft(x: &[f64]) -> Vec<f64> {
//     let n = x.len();
//     let mean: f64 = x.iter().sum::<f64>() / n as f64;

//     // Zero-mean
//     let mut x_centered: Vec<Complex<f64>> = x.iter()
//         .map(|&v| Complex{ re: v - mean, im: 0.0 })
//         .collect();

//     // Zero-padding to 2n
//     x_centered.extend(vec![Complex{ re: 0.0, im: 0.0 }; n]);

//     let mut planner = FftPlanner::new();
//     let fft = planner.plan_fft_forward(x_centered.len());
//     let ifft = planner.plan_fft_inverse(x_centered.len());

//     let mut buffer = x_centered.clone();
//     fft.process(&mut buffer);

//     // Power spectrum
//     for c in buffer.iter_mut() {
//         *c = *c * c.conj();
//     }

//     // Inverse FFT to get autocovariance
//     ifft.process(&mut buffer);

//     // Normalize by number of points and C(0)
//     let c0 = buffer[0].re / (2.0 * n as f64);
//     let acf: Vec<f64> = buffer[..n].iter().map(|c| c.re / (2.0 * n as f64) / c0).collect();

//     acf
// }

// fn integrated_autocorrelation_time(acf: &[f64]) -> f64 {
//     let mut tau = 0.5; // start with 0.5 to include lag 0
//     for &rho in acf.iter().skip(1) {
//         if rho <= 0.0 {
//             break; // truncate when correlation becomes non-positive
//         }
//         tau += rho;
//     }
//     tau
// }

// fn main() {
//     // Example: AR(1) process with rho = 0.8
//     let n = 10000;
//     let mut x = vec![0.0; n];
//     let mut rng = rand::rng();
//     let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

//     x[0] = normal.sample(&mut rng);
//     for i in 1..n {
//         x[i] = 0.8 * x[i-1] + normal.sample(&mut rng);
//     }

//     let acf = autocorrelation_fft(&x);
//     let tau = integrated_autocorrelation_time(&acf);
//     println!("Estimated autocorrelation time: {}", tau);
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     use rand_distr::Normal;

//     #[test]
//     fn test_autocorrelation_naive_ar1_iid() {
//         let n = 10000;
//         let mut rng = rand::rng();
//         let normal = Normal::new(0.0, 1.0).unwrap();

//         // -------- Test 1: i.i.d. noise --------
//         let x_iid: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
//         let acf_iid = autocorrelation_naive(&x_iid);
//         let tau_iid = integrated_autocorrelation_time(&acf_iid);

//         println!("i.i.d. noise τ (should be ~0.5): {:.4}", tau_iid);
//         // Assert τ is close to 0.5 (allow 10% tolerance)
//         assert!((tau_iid - 0.5).abs() < 0.05, "i.i.d. τ not within expected range");

//         // -------- Test 2: AR(1) process --------
//         let phi = 0.8;
//         let mut x_ar = vec![0.0; n];
//         x_ar[0] = normal.sample(&mut rng);
//         for i in 1..n {
//             x_ar[i] = phi * x_ar[i - 1] + normal.sample(&mut rng);
//         }
//         let acf_ar = autocorrelation_naive(&x_ar);
//         let tau_ar = integrated_autocorrelation_time(&acf_ar);

//         let tau_true = (1.0 + phi) / (1.0 - phi) / 2.0;
//         println!("AR(1) τ estimated: {:.4}, true: {:.4}", tau_ar, tau_true);
//         // Assert τ is within 5% of theoretical value
//         assert!((tau_ar - tau_true).abs() / tau_true < 0.05, "AR(1) τ deviates too much");
//     }
// }
