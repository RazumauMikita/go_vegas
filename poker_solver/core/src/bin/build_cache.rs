use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use poker_core::{EquityCache, UNIQUE_PAIR_COUNT};

fn main() -> ExitCode {
    match run(env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_usage();
        return Ok(());
    }

    let config = parse_args(args)?;

    if let Some(sample_count) = config.sample {
        run_sample(config.iterations, sample_count);
        return Ok(());
    }

    eprintln!(
        "Generating equity cache with {} iterations per pair ({} unique pairs)...",
        config.iterations, UNIQUE_PAIR_COUNT
    );
    let started = Instant::now();
    let cache = EquityCache::generate_with_progress(config.iterations, |completed, started_at| {
        report_progress(completed, UNIQUE_PAIR_COUNT as u64, started_at);
    });
    let elapsed = started.elapsed();

    cache.save(Path::new(&config.output)).map_err(|error| {
        format!(
            "failed to save cache to {}: {error}",
            config.output.display()
        )
    })?;

    eprintln!(
        "Saved {} ({} bytes) in {:.1}s",
        config.output.display(),
        169 * 169 * 4,
        elapsed.as_secs_f64()
    );

    Ok(())
}

struct Config {
    iterations: u64,
    output: PathBuf,
    sample: Option<usize>,
}

fn run_sample(iterations: u64, sample_count: usize) {
    eprintln!(
        "Sample mode: generating {sample_count} unique pairs with {iterations} iterations each..."
    );
    let started = Instant::now();
    let _cache = EquityCache::generate_sample_with_progress(
        iterations,
        sample_count,
        |completed, started_at| {
            report_progress(completed, sample_count as u64, started_at);
        },
    );
    let elapsed = started.elapsed();
    let seconds_per_pair = elapsed.as_secs_f64() / sample_count as f64;
    let full_eta_secs = seconds_per_pair * UNIQUE_PAIR_COUNT as f64;

    eprintln!();
    eprintln!("Sample pairs:      {sample_count}");
    eprintln!("Elapsed:           {:.2}s", elapsed.as_secs_f64());
    eprintln!("Seconds per pair:  {:.3}s", seconds_per_pair);
    eprintln!("Unique pairs:      {UNIQUE_PAIR_COUNT}");
    eprintln!(
        "Estimated full ETA: {:.1} min ({:.0}s)",
        full_eta_secs / 60.0,
        full_eta_secs
    );
}

fn parse_args(args: Vec<String>) -> Result<Config, String> {
    let mut iterations = 100_000_u64;
    let mut output = PathBuf::from("equity_cache.bin");
    let mut sample = None;
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
            "--sample" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--sample requires a value".to_string())?;
                sample = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid sample value: {value}"))?,
                );
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        index += 1;
    }

    if iterations == 0 {
        return Err("iterations must be greater than zero".to_string());
    }

    if let Some(sample_count) = sample {
        if sample_count == 0 {
            return Err("sample must be greater than zero".to_string());
        }
    }

    Ok(Config {
        iterations,
        output,
        sample,
    })
}

fn report_progress(completed: u64, total: u64, started: Instant) {
    if completed == 0 || completed == total || completed.is_multiple_of((total / 20).max(1)) {
        let percent = (completed as f64 / total as f64) * 100.0;
        let elapsed = started.elapsed().as_secs_f64();
        let eta_min = if completed == 0 {
            0.0
        } else {
            (elapsed / completed as f64) * (total - completed) as f64 / 60.0
        };
        eprintln!("Done {completed}/{total} pairs ({percent:.0}%), ETA: {eta_min:.1} min");
    }
}

fn print_usage() {
    eprintln!(
        "usage:
  build_cache [--iterations N] [--output PATH]
  build_cache --sample N [--iterations N]

Examples:
  build_cache --iterations 100000 --output equity_cache.bin
  build_cache --sample 20 --iterations 100000"
    );
}
