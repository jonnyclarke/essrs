use std::{fs::File, io::BufWriter, path::Path};

use ndarray::Axis;

use crate::{chains::static_buffer::StaticBuffer, ess::EnsembleSliceSamplerConfigTrait};

impl<CONFIG: EnsembleSliceSamplerConfigTrait> StaticBuffer<CONFIG> {
    pub fn write_final_walkers<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        if self.n_stored == 0 {
            return Ok(());
        }

        let final_state = self.chains.index_axis(Axis(0), self.n_stored - 1);

        for walker in final_state.rows() {
            for (j, value) in walker.iter().enumerate() {
                if j > 0 {
                    write!(writer, " ")?;
                }
                write!(writer, "{}", value)?;
            }
            writeln!(writer)?;
        }

        Ok(())
    }

    pub fn dump_final_walkers<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.write_final_walkers(&mut writer)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::NamedTempFile;

    use crate::testing::helpers::make_test_static_buffer;

    #[test]
    fn test_write_final_walkers_to_buffer() {
        let buffer = make_test_static_buffer();

        let mut out = Vec::new();
        buffer.write_final_walkers(&mut out).unwrap();

        let s = String::from_utf8(out).unwrap();

        let expected = "21 22\n23 24\n25 26\n27 28\n";

        assert_eq!(s, expected);
    }

    #[test]
    fn test_dump_final_walkers_to_file() {
        let buffer = make_test_static_buffer();

        // Use a temporary file for testing
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path();

        buffer.dump_final_walkers(path).unwrap();

        let contents = fs::read_to_string(path).unwrap();

        // TODO: Replace this with your actual expected output
        let expected = "21 22\n23 24\n25 26\n27 28\n";

        assert_eq!(contents, expected);
    }

    #[test]
    fn test_write_final_walkers_empty() {
        // Test the case with n_stored = 0
        let mut buffer = make_test_static_buffer();
        buffer.n_stored = 0;

        let mut out = Vec::new();
        let res = buffer.write_final_walkers(&mut out);

        assert!(res.is_ok());
        assert!(out.is_empty());
    }
}
