#![forbid(unsafe_code)]

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use poker_core::four_way_rank_cache::{CACHE_SIZE, FILE_BYTES};
use poker_core::{Bucketing, EquityCache, FourWayRankCache};
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
        CACHE_SIZE
    } else {
        config.limit.min(CACHE_SIZE)
    };
    let mc_samples = config.samples.saturating_mul(config.samples);

    println!("Loading bucketing...");
    let bucketing = load_bucketing(&config)?;

    println!(
        "Generating 4-way cache: {} combos × {mc_samples} MC samples...",
        format_count(total as u64)
    );
    eprintln!("Rayon threads: {}", current_num_threads());
    eprintln!("Output:        {}", config.output.display());

    let started = Instant::now();
    let (cache, stats) = FourWayRankCache::generate_with_progress(
        &bucketing,
        config.samples,
        config.limit,
        |completed, started_at| {
            report_progress(completed, total as u64, started_at);
        },
    );
    let generate_elapsed = started.elapsed();

    cache.save(Path::new(&config.output)).map_err(|error| {
        format!(
            "failed to save cache to {}: {error}",
            config.output.display()
        )
    })?;
    let elapsed = started.elapsed();

    let size = std::fs::metadata(&config.output)
        .map(|meta| meta.len())
        .unwrap_or(FILE_BYTES as u64);

    println!();
    println!(
        "Done. Saved {} ({:.1} MB).",
        config.output.display(),
        size as f64 / (1024.0 * 1024.0)
    );
    println!("Time: {}.", format_duration(elapsed));
    eprintln!("Valid:   {}", stats.valid);
    eprintln!("Invalid: {}", stats.invalid);
    eprintln!("Total:   {}", stats.total);
    eprintln!(
        "Throughput: {:.0} combos/s",
        stats.total as f64 / generate_elapsed.as_secs_f64().max(1e-9)
    );
    let full_eta = Duration::from_secs_f64(
        CACHE_SIZE as f64 / (stats.total as f64 / generate_elapsed.as_secs_f64().max(1e-9)),
    );
    eprintln!("Full-cache ETA: {}", format_duration(full_eta));

    Ok(())
}

struct Config {
    bucketing: PathBuf,
    cache: PathBuf,
    output: PathBuf,
    samples: u64,
    limit: usize,
}

fn load_bucketing(config: &Config) -> Result<Bucketing, String> {
    match Bucketing::load(&config.bucketing) {
        Ok(bucketing) => Ok(bucketing),
        Err(error) => {
            eprintln!(
                "failed to load {}: {error}; falling back to equity cache",
                config.bucketing.display()
            );
            let cache = EquityCache::load(&config.cache).map_err(|cache_error| {
                format!(
                    "failed to load bucketing from {} ({error}) and cache from {}: {cache_error}",
                    config.bucketing.display(),
                    config.cache.display()
                )
            })?;
            Ok(Bucketing::new(&cache))
        }
    }
}

fn parse_args(args: Vec<String>) -> Result<Config, String> {
    let mut bucketing = PathBuf::from("bucketing.bin");
    let mut cache = PathBuf::from("equity_cache.bin");
    let mut output = PathBuf::from("four_way_rank_cache.bin");
    let mut samples = 20_u64;
    let mut limit = 0_usize;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--bucketing" => {
                index += 1;
                bucketing = PathBuf::from(
                    args.get(index)
                        .ok_or_else(|| "--bucketing requires a path".to_string())?,
                );
            }
            "--cache" => {
                index += 1;
                cache = PathBuf::from(
                    args.get(index)
                        .ok_or_else(|| "--cache requires a path".to_string())?,
                );
            }
            "--output" => {
                index += 1;
                output = PathBuf::from(
                    args.get(index)
                        .ok_or_else(|| "--output requires a path".to_string())?,
                );
            }
            "--samples" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--samples requires a value".to_string())?;
                samples = value
                    .parse()
                    .map_err(|_| format!("invalid samples value: {value}"))?;
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
        bucketing,
        cache,
        output,
        samples,
        limit,
    })
}

fn report_progress(completed: u64, total: u64, started: Instant) {
    let percent = if total == 0 {
        100.0
    } else {
        (completed as f64 / total as f64) * 100.0
    };
    let filled = ((percent / 5.0).round() as usize).min(20);
    let bar = format!(
        "[{}{}]",
        "=".repeat(filled.saturating_sub(1)) + if filled > 0 { ">" } else { "" },
        " ".repeat(20 - filled)
    );
    let elapsed = started.elapsed().as_secs_f64();
    let eta = if completed == 0 {
        Duration::ZERO
    } else {
        Duration::from_secs_f64((elapsed / completed as f64) * (total - completed) as f64)
    };
    eprintln!(
        "{bar} {}/{} ({percent:.0}%) ETA: {}",
        format_count(completed),
        format_count(total),
        format_duration(eta)
    );
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else {
        format!("{n}")
    }
}

fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    if secs < 1.0 {
        return format!("{secs:.1}s");
    }
    let total = duration.as_secs();
    let hours = total / 3600;
    let mins = (total % 3600) / 60;
    let rem = total % 60;
    if hours > 0 {
        if mins == 0 {
            format!("{hours}h")
        } else {
            format!("{hours}h {mins}m")
        }
    } else if mins > 0 {
        format!("{mins}m")
    } else {
        format!("{rem}s")
    }
}

fn print_usage() {
    eprintln!(
        "usage:
  build_4way_cache --bucketing PATH --cache PATH --output PATH [--samples N] [--limit N]

Examples:
  build_4way_cache --bucketing bucketing.bin --cache equity_cache.bin --output four_way_rank_cache.bin --samples 20
  build_4way_cache --bucketing bucketing.bin --cache equity_cache.bin --output test.bin --samples 20 --limit 1000"
    );
}
