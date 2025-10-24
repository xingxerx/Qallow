//! GUI entry point for the Rust-based Qallow application surface.

#[cfg(feature = "gui")]
mod fltk_gui;

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum, ValueHint};
use qallow_ui::{record_to_pretty_json, tail_csv, TelemetryRecord};

#[cfg(feature = "gui")]
use fltk_gui::run_gui;

#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
enum OutputFormat {
    Table,
    Json,
}

#[derive(Parser, Debug, Clone)]
#[command(
    author,
    version,
    about = "Qallow unified dashboard",
    disable_help_subcommand = true
)]
struct Args {
    /// Run in CLI mode instead of launching GUI
    #[arg(long)]
    cli: bool,

    /// Override the telemetry CSV path
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    telemetry: Option<PathBuf>,

    /// Number of rows to read from the tail of the telemetry file
    #[arg(short = 'n', long = "rows", default_value_t = 5)]
    rows: usize,

    /// Output format: table or json
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Table)]
    format: OutputFormat,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // If --cli flag is provided, run in CLI mode; otherwise launch GUI (if available)
    if args.cli {
        run_cli_mode(args)
    } else {
        #[cfg(feature = "gui")]
        {
            run_gui_mode()
        }
        #[cfg(not(feature = "gui"))]
        {
            eprintln!("GUI feature not compiled. Running in CLI mode.");
            run_cli_mode(args)
        }
    }
}

#[cfg(feature = "gui")]
fn run_gui_mode() -> Result<()> {
    run_gui()
}

fn run_cli_mode(args: Args) -> Result<()> {
    let telemetry_path = args
        .telemetry
        .unwrap_or_else(|| PathBuf::from("data/logs/telemetry_stream.csv"));

    let records = tail_csv(&telemetry_path, args.rows)
        .with_context(|| format!("loading telemetry from {}", telemetry_path.display()))?;

    if records.is_empty() {
        println!(
            "no telemetry rows available in {}",
            telemetry_path.display()
        );
        return Ok(());
    }

    match args.format {
        OutputFormat::Table => print_table(&records),
        OutputFormat::Json => print_json(&records)?,
    }

    Ok(())
}

fn print_table(records: &[TelemetryRecord]) {
    let mut headers = BTreeSet::new();
    for record in records {
        headers.extend(record.keys().cloned());
    }

    let headers: Vec<String> = headers.into_iter().collect();
    let widths: Vec<usize> = headers
        .iter()
        .map(|header| {
            let value_width = records
                .iter()
                .filter_map(|record| record.get(header))
                .map(|value| value.len())
                .max()
                .unwrap_or(0);
            header.len().max(value_width)
        })
        .collect();

    let mut line = String::new();
    for (header, width) in headers.iter().zip(widths.iter()) {
        line.push_str(&format!("{:>width$} ", header, width = width));
    }
    println!("{}", line.trim_end());

    for record in records {
        line.clear();
        for (header, width) in headers.iter().zip(widths.iter()) {
            let value = record.get(header).map(String::as_str).unwrap_or("-");
            line.push_str(&format!("{:>width$} ", value, width = width));
        }
        println!("{}", line.trim_end());
    }
}

fn print_json(records: &[TelemetryRecord]) -> Result<()> {
    for record in records {
        let json = record_to_pretty_json(record)?;
        println!("{}", json);
    }
    Ok(())
}
