use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use poker_core::EquityCache;

const TOTAL_PAIRS: u64 = 169 * 169;

fn main() -> ExitCode {
    let (iterations, output) = match parse_args(env::args().skip(1).collect()) {
        Ok(values) => values,
        Err(message) => {
            eprintln!("{message}");
            eprintln!("usage: build_cache [--iterations N] [--output PATH]");
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Generating equity cache with {iterations} iterations per valid combo pair...");
    let started = Instant::now();
    let cache = EquityCache::generate_with_progress(iterations, report_progress);
    let elapsed = started.elapsed();

    if let Err(error) = cache.save(Path::new(&output)) {
        eprintln!("failed to save cache to {}: {error}", output.display());
        return ExitCode::FAILURE;
    }

    eprintln!(
        "Saved {} ({} bytes) in {:.1}s",
        output.display(),
        169 * 169 * 4,
        elapsed.as_secs_f64()
    );

    ExitCode::SUCCESS
}

fn parse_args(args: Vec<String>) -> Result<(u64, PathBuf), String> {
    let mut iterations = 100_000_u64;
    let mut output = PathBuf::from("equity_cache.bin");
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--iterations" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--iterations requires a value".to_string())?;
                iterations = value
                    .parse()
                    .map_err(|_| format!("invalid iterations value: {value}"))?;
            }
            "--output" => {
                index += 1;
                output = PathBuf::from(
                    args.get(index)
                        .ok_or_else(|| "--output requires a path".to_string())?,
                );
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        index += 1;
    }

    if iterations == 0 {
        return Err("iterations must be greater than zero".to_string());
    }

    Ok((iterations, output))
}

fn report_progress(completed: u64, started: Instant) {
    if completed == 0 || completed == TOTAL_PAIRS || completed.is_multiple_of(TOTAL_PAIRS / 20) {
        let percent = (completed as f64 / TOTAL_PAIRS as f64) * 100.0;
        let elapsed = started.elapsed().as_secs_f64();
        let eta_min = if completed == 0 {
            0.0
        } else {
            (elapsed / completed as f64) * (TOTAL_PAIRS - completed) as f64 / 60.0
        };
        eprintln!("Done {completed}/{TOTAL_PAIRS} pairs ({percent:.0}%), ETA: {eta_min:.1} min");
    }
}
