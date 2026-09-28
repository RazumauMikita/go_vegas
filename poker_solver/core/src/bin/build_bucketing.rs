#![forbid(unsafe_code)]

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use poker_core::{combo_label, Bucketing, EquityCache, NUM_BUCKETS};

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
    let cache = EquityCache::load(&config.cache).map_err(|error| {
        format!(
            "failed to load cache from {}: {error}",
            config.cache.display()
        )
    })?;

    println!("Computing avg equity for 169 hands...");
    let hand_eq = avg_equity_table(&cache);
    println!("Sorting and bucketing into 50 groups...");
    let bucketing = Bucketing::new(&cache);

    bucketing.save(&config.output).map_err(|error| {
        format!(
            "failed to save bucketing to {}: {error}",
            config.output.display()
        )
    })?;

    println!();
    print_bucket_preview(&bucketing, &hand_eq, 0, "weakest");
    print_bucket_preview(&bucketing, &hand_eq, 25, "middle");
    print_bucket_preview(&bucketing, &hand_eq, 49, "strongest");

    println!();
    println!("Bucket sizes:");
    for (bucket, &size) in bucketing.bucket_sizes.iter().enumerate() {
        println!("  {bucket:>2}: {size}");
    }

    println!();
    print_top_hands(&bucketing, &hand_eq, 0, 3);
    print_top_hands(&bucketing, &hand_eq, 25, 3);
    print_top_hands(&bucketing, &hand_eq, 49, 3);

    let size = std::fs::metadata(&config.output)
        .map(|meta| meta.len())
        .unwrap_or(0);
    println!();
    println!("Saved to {} ({size} bytes).", config.output.display());

    Ok(())
}

struct Config {
    cache: PathBuf,
    output: PathBuf,
}

fn parse_args(args: Vec<String>) -> Result<Config, String> {
    let mut cache = PathBuf::from("equity_cache.bin");
    let mut output = PathBuf::from("bucketing.bin");
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
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
            other => return Err(format!("unknown argument: {other}")),
        }
        index += 1;
    }

    Ok(Config { cache, output })
}

fn avg_equity_table(cache: &EquityCache) -> [f64; 169] {
    let mut avg = [0.0_f64; 169];
    for hero in 0..169_u8 {
        let mut sum = 0.0;
        for opponent in 0..169_u8 {
            if opponent != hero {
                sum += cache.equity(hero, opponent);
            }
        }
        avg[hero as usize] = sum / 168.0;
    }
    avg
}

fn hands_in_bucket(map: &[u8; 169], bucket: u8) -> Vec<u8> {
    (0..169_u8)
        .filter(|&hand| map[hand as usize] == bucket)
        .collect()
}

fn sorted_hands(map: &[u8; 169], hand_eq: &[f64; 169], bucket: u8) -> Vec<u8> {
    let mut hands = hands_in_bucket(map, bucket);
    hands.sort_by(|&left, &right| {
        let order = hand_eq[left as usize]
            .partial_cmp(&hand_eq[right as usize])
            .unwrap_or(std::cmp::Ordering::Equal);
        if bucket == (NUM_BUCKETS as u8 - 1) {
            order.reverse().then(left.cmp(&right))
        } else {
            order.then(left.cmp(&right))
        }
    });
    hands
}

fn print_bucket_preview(bucketing: &Bucketing, hand_eq: &[f64; 169], bucket: u8, label: &str) {
    let hands = sorted_hands(&bucketing.map, hand_eq, bucket);
    let names: Vec<String> = hands.iter().copied().map(combo_label).collect();
    println!(
        "  Bucket {bucket} ({label}):  {}  (avg eq {:.2})",
        names.join(", "),
        bucketing.bucket_avg_equity[bucket as usize]
    );
}

fn print_top_hands(bucketing: &Bucketing, hand_eq: &[f64; 169], bucket: u8, count: usize) {
    let hands = sorted_hands(&bucketing.map, hand_eq, bucket);
    let names: Vec<String> = hands.iter().copied().take(count).map(combo_label).collect();
    println!("Top-{count} in bucket {bucket}: {}", names.join(", "));
}

fn print_usage() {
    eprintln!(
        "usage:
  build_bucketing --cache PATH --output PATH

Examples:
  build_bucketing --cache equity_cache.bin --output bucketing.bin"
    );
}
