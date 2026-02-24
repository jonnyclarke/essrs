use std::{fs::File, io::BufWriter, path::Path};

use crate::state::WalkerState;

impl WalkerState {
    pub fn write_final_log_likelihood<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        for (j, value) in self.get_ll_vector().iter().enumerate() {
            if j > 0 {
                writeln!(writer)?;
            }
            write!(writer, "{}", value)?;
        }
        writeln!(writer)?;

        Ok(())
    }

    pub fn dump_final_log_likelihood<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.write_final_log_likelihood(&mut writer)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use ndarray::array;
    use tempfile::NamedTempFile;

    use crate::state::WalkerState;

    fn make_test_state_obj() -> WalkerState {
        WalkerState {
            state: array![[1.0, 2.0], [3.0, 4.0]],
            ll: array![-0.1, -0.2],
            n: 2,
        }
    }
    #[test]
    fn test_write_final_walkers_to_buffer() {
        let state = make_test_state_obj();
        let mut out = Vec::new();

        state.write_final_log_likelihood(&mut out).unwrap();

        let s = String::from_utf8(out).unwrap();

        let expected = "-0.1\n-0.2\n";

        assert_eq!(s, expected);
    }

    #[test]
    fn test_dump_final_walkers_to_file() {
        let state = make_test_state_obj();

        // Use a temporary file for testing
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path();

        state.dump_final_log_likelihood(path).unwrap();

        let contents = fs::read_to_string(path).unwrap();

        // TODO: Replace this with your actual expected output
        let expected = "-0.1\n-0.2\n";

        assert_eq!(contents, expected);
    }
}
