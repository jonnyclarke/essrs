// python bindings
mod normal_1d;
mod normal_2d;

mod science_archive;
mod toy;

use pyo3::prelude::*;

use crate::python_bindings::{
    normal_1d::normal_1d_deconv::normal_1d_deconv,
    normal_2d::normal_2d_gmm_deconv::normal_2d_gmm_deconv,
    // +---------------------+
    // | COMMUNITY FUNCTIONS |
    // +---------------------+
    // > Clarke 2027
    // -- proper motion fits
    science_archive::clarke2027::virac2_pm_rc_rgbb::virac2_pm_rc_rgbb,
    science_archive::clarke2027::virac2_pm_rgb::virac2_pm_rgb,
    // -- density fits
    science_archive::clarke2027::virac2_rho_rc::virac2_rho_rc,
    science_archive::clarke2027::virac2_rho_rc_rgbb::virac2_rho_rc_rgbb,
    science_archive::clarke2027::virac2_rho_rc_rgbb_2peak::virac2_rho_rc_rgbb_2peak,
    toy::{neals_funnel::toy_neals_funnel, rosenbrock::toy_rosenbrock},
};

#[pymodule]
fn _essrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let normal_1d = PyModule::new(m.py(), "normal_1d")?;
    normal_1d.add_function(wrap_pyfunction!(normal_1d_deconv, &normal_1d)?)?;
    m.add_submodule(&normal_1d)?;

    let normal_2d = PyModule::new(m.py(), "normal_2d")?;
    normal_2d.add_function(wrap_pyfunction!(normal_2d_gmm_deconv, &normal_2d)?)?;
    m.add_submodule(&normal_2d)?;

    // Toy Models -- for evaluation
    let toy = PyModule::new(m.py(), "toy")?;
    toy.add_function(wrap_pyfunction!(toy_rosenbrock, &toy)?)?;
    toy.add_function(wrap_pyfunction!(toy_neals_funnel, &toy)?)?;

    m.add_submodule(&toy)?;

    // +---------------------+
    // | COMMUNITY FUNCTIONS |
    // +---------------------+
    let clarke2027 = PyModule::new(m.py(), "clarke_2027")?;

    clarke2027.add_function(wrap_pyfunction!(virac2_rho_rc, &clarke2027)?)?; // the RC only fit. Rapid convergence?
    clarke2027.add_function(wrap_pyfunction!(virac2_rho_rc_rgbb, &clarke2027)?)?;
    clarke2027.add_function(wrap_pyfunction!(virac2_rho_rc_rgbb_2peak, &clarke2027)?)?;

    clarke2027.add_function(wrap_pyfunction!(virac2_pm_rgb, &clarke2027)?)?;
    clarke2027.add_function(wrap_pyfunction!(virac2_pm_rc_rgbb, &clarke2027)?)?;

    let science_archive = PyModule::new(m.py(), "scilibrary")?;

    science_archive.add_submodule(&clarke2027)?;

    m.add_submodule(&science_archive)?;

    Ok(())
}
