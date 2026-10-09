// Function to apply the exponential change of variable
pub fn lj_exp(x: f64, ll: &mut f64) -> f64 {
    *ll += x;
    x.exp()
}
