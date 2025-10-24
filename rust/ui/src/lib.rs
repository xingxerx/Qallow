//! Telemetry utilities shared by the Rust front-ends.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use thiserror::Error;

/// Row representation for generic telemetry CSV data.
pub type TelemetryRecord = BTreeMap<String, String>;

/// Errors that can surface when loading telemetry artefacts.
#[derive(Debug, Error)]
pub enum TelemetryError {
    #[error("telemetry file not found at {path}")]
    NotFound { path: PathBuf },
    #[error("failed to parse telemetry csv at {path}: {source}")]
    Csv {
        path: PathBuf,
        #[source]
        source: csv::Error,
    },
}

/// Load every record from a telemetry CSV file into memory.
///
pub fn load_csv<P: AsRef<Path>>(path: P) -> Result<Vec<TelemetryRecord>, TelemetryError> {
    let path_ref = path.as_ref();
    let mut reader = csv::Reader::from_path(path_ref)
        .map_err(|err| map_csv_error(path_ref, err))?;

    let headers = reader
        .headers()
        .map_err(|err| map_csv_error(path_ref, err))?
        .clone();

    let mut records = Vec::new();

    for result in reader.records() {
        let record = result.map_err(|err| map_csv_error(path_ref, err))?;
        let mut row = TelemetryRecord::new();

        for (header, value) in headers.iter().zip(record.iter()) {
            if !header.is_empty() {
                row.insert(header.to_owned(), value.to_owned());
            }
        }

        if record.len() > headers.len() {
            for idx in headers.len()..record.len() {
                if let Some(value) = record.get(idx) {
                    row.insert(format!("column_{idx}"), value.to_owned());
                }
            }
        }

        records.push(row);
    }

    Ok(records)
}

/// Return the newest `count` records from the telemetry CSV, preserving order.
pub fn tail_csv<P: AsRef<Path>>(path: P, count: usize) -> Result<Vec<TelemetryRecord>, TelemetryError> {
    let mut records = load_csv(path)?;
    if count == 0 {
        records.clear();
        return Ok(records);
    }

    if count >= records.len() {
        return Ok(records);
    }

    let split_at = records.len() - count;
    Ok(records.split_off(split_at))
}

/// Render a telemetry record into a prettified JSON blob.
pub fn record_to_pretty_json(record: &TelemetryRecord) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(record)
}

fn map_csv_error(path: &Path, err: csv::Error) -> TelemetryError {
    if let Some(io_err) = err.io_error() {
        if io_err.kind() == std::io::ErrorKind::NotFound {
            return TelemetryError::NotFound {
                path: path.to_path_buf(),
            };
        }
    }

    TelemetryError::Csv {
        path: path.to_path_buf(),
        source: err,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn load_csv_reads_records() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        writeln!(
            file,
            "metric,value\nphase,13\nthroughput,42.5"
        )
        .expect("write csv");

        let records = load_csv(file.path()).expect("load csv");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].get("metric").unwrap(), "phase");
        assert_eq!(records[0].get("value").unwrap(), "13");
    }

    #[test]
    fn tail_csv_limits_records() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        writeln!(
            file,
            "metric,value\nphase,13\nthroughput,42.5\nlatency,99"
        )
        .expect("write csv");

        let records = tail_csv(file.path(), 2).expect("tail csv");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].get("metric").unwrap(), "throughput");
        assert_eq!(records[1].get("metric").unwrap(), "latency");
    }

    #[test]
    fn missing_file_is_reported() {
        let err = load_csv("/tmp/qallow-non-existent.csv").expect_err("should fail");
        match err {
            TelemetryError::NotFound { .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn record_json_round_trip() {
        let mut record = TelemetryRecord::new();
        record.insert("phase".to_string(), "13".to_string());
        record.insert("status".to_string(), "ready".to_string());

        let json = record_to_pretty_json(&record).expect("json");
        let value: serde_json::Value = serde_json::from_str(&json).expect("parse");
        assert_eq!(value["phase"], "13");
        assert_eq!(value["status"], "ready");
    }
}
