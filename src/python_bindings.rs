// python bindings
mod normal_1d;
mod normal_2d;

use pyo3::prelude::*;

use crate::python_bindings::{
    normal_1d::normal_1d_deconv::normal_1d_deconv,
    normal_2d::normal_2d_gmm_deconv::normal_2d_gmm_deconv,
};

#[pymodule]
fn _essrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let normal_1d = PyModule::new(m.py(), "normal_1d")?;
    normal_1d.add_function(wrap_pyfunction!(normal_1d_deconv, &normal_1d)?)?;
    m.add_submodule(&normal_1d)?;

    let normal_2d = PyModule::new(m.py(), "normal_2d")?;
    normal_2d.add_function(wrap_pyfunction!(normal_2d_gmm_deconv, &normal_2d)?)?;
    m.add_submodule(&normal_2d)?;

    Ok(())
}
