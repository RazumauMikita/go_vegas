#![forbid(unsafe_code)]

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use poker_core::{ThreeWayRankCache, THREE_WAY_RANK_CACHE_BYTES, THREE_WAY_RANK_CACHE_SIZE};
use rayon::current_num_threads;

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
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_usage();
        return Ok(());
    }

    let config = parse_args(args)?;
    let total = if config.limit == 0 {
        THREE_WAY_RANK_CACHE_SIZE
    } else {
        config.limit.min(THREE_WAY_RANK_CACHE_SIZE)
    };

    eprintln!("Building 3-way rank cache");
    eprintln!("Samples per triple: {}", config.samples);
    eprintln!("Triples:            {total} / {THREE_WAY_RANK_CACHE_SIZE}");
    eprintln!("Rayon threads:      {}", current_num_threads());
    eprintln!("Output:             {}", config.output.display());

    let started = Instant::now();
    let (cache, stats) = ThreeWayRankCache::generate_with_progress(
        config.samples,
        config.limit,
        |completed, started_at| {
            report_progress(completed, total as u64, started_at);
        },
    );
    let elapsed = started.elapsed();

    cache.save(Path::new(&config.output)).map_err(|error| {
        format!(
            "failed to save cache to {}: {error}",
            config.output.display()
        )
    })?;

    let size = std::fs::metadata(&config.output)
        .map(|meta| meta.len())
        .unwrap_or(THREE_WAY_RANK_CACHE_BYTES as u64);

    eprintln!();
    eprintln!("Valid triples:   {}", stats.valid);
    eprintln!("Invalid triples: {}", stats.invalid);
    eprintln!("Considered:      {}", stats.total);
    eprintln!(
        "Elapsed:         {:.1}s ({:.1} min)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / 60.0
    );
    eprintln!(
        "Throughput:      {:.0} triples/s",
        stats.total as f64 / elapsed.as_secs_f64().max(1e-9)
    );
    eprintln!(
        "Saved {} ({size} bytes, expected {THREE_WAY_RANK_CACHE_BYTES})",
        config.output.display()
    );

    Ok(())
}

struct Config {
    samples: u64,
    output: PathBuf,
    limit: usize,
}

fn parse_args(args: Vec<String>) -> Result<Config, String> {
    let mut samples = 100_u64;
    let mut output = PathBuf::from("three_way_rank_cache.bin");
    let mut limit = 0_usize;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--samples" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--samples requires a value".to_string())?;
                samples = value
                    .parse()
                    .map_err(|_| format!("invalid samples value: {value}"))?;
            }
            "--output" => {
                index += 1;
                output = PathBuf::from(
                    args.get(index)
                        .ok_or_else(|| "--output requires a path".to_string())?,
                );
            }
            "--limit" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--limit requires a value".to_string())?;
                limit = value
                    .parse()
                    .map_err(|_| format!("invalid limit value: {value}"))?;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        index += 1;
    }

    if samples == 0 {
        return Err("samples must be greater than zero".to_string());
    }

    Ok(Config {
        samples,
        output,
        limit,
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
        eprintln!("Done {completed}/{total} triples ({percent:.0}%), ETA: {eta_min:.1} min");
    }
}

fn print_usage() {
    eprintln!(
        "usage:
  build_rank_cache [--samples N] [--output PATH] [--limit N]

Examples:
  build_rank_cache --samples 100 --output three_way_rank_cache.bin
  build_rank_cache --limit 1000 --samples 2 --output /tmp/rank_cache_sample.bin"
    );
}
