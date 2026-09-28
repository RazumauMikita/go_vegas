#![forbid(unsafe_code)]

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use poker_core::four_way_rank_cache::{CACHE_SIZE, FILE_BYTES, NUM_PERMS};
use poker_core::{Bucketing, EquityCache, FourWayRankCache, NUM_BUCKETS};
use rayon::current_num_threads;

const RANK_PERMUTATIONS: [[usize; 4]; NUM_PERMS] = [
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [1, 2, 0, 3],
    [1, 2, 3, 0],
    [1, 3, 0, 2],
    [1, 3, 2, 0],
    [2, 0, 1, 3],
    [2, 0, 3, 1],
    [2, 1, 0, 3],
    [2, 1, 3, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
    [3, 0, 1, 2],
    [3, 0, 2, 1],
    [3, 1, 0, 2],
    [3, 1, 2, 0],
    [3, 2, 0, 1],
    [3, 2, 1, 0],
];

fn main() -> ExitCode {
    match run(env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_usage();
        return Ok(());
    }

    let config = parse_args(args).inspect_err(|_| print_usage())?;
    if let Some(path) = config.validate {
        return validate(&path);
    }

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
    validate: Option<PathBuf>,
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
    let mut validate = None;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--validate" => {
                index += 1;
                validate = Some(PathBuf::from(
                    args.get(index)
                        .ok_or_else(|| "--validate requires a path".to_string())?,
                ));
            }
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
        validate,
    })
}

fn validate(path: &Path) -> Result<(), String> {
    println!("Loading {}...", path.display());
    let meta = std::fs::metadata(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let actual = meta.len();
    println!(
        "Size: {:.1} MB (ожидается {:.1} MB).",
        actual as f64 / (1024.0 * 1024.0),
        FILE_BYTES as f64 / (1024.0 * 1024.0)
    );
    let size_ok = actual == FILE_BYTES as u64;

    let cache = FourWayRankCache::load(path)
        .map_err(|error| format!("failed to load {}: {error}", path.display()))?;
    println!("Combos loaded: {}.", with_commas(CACHE_SIZE));
    println!();

    let mut failed = !size_ok;
    if !size_ok {
        println!("Size check: FAIL");
        failed = true;
    }

    let mut rng = Rng::new(0x5641_4C31_4441_5445);
    let mut random_ok = 0_usize;
    for _ in 0..20 {
        let combo = random_combo(&mut rng);
        match check_combo(&cache, combo) {
            Ok(_) => random_ok += 1,
            Err(reason) => {
                failed = true;
                println!(
                    "Random combo ({}, {}, {}, {}): FAIL ({reason})",
                    combo[0], combo[1], combo[2], combo[3]
                );
            }
        }
    }

    let zero = cache.lookup(0, 0, 0, 0);
    match check_combo(&cache, [0, 0, 0, 0]) {
        Ok(sum) => println!(
            "Random combo (0, 0, 0, 0): sum = {sum:.2}, top-5 = [{}]",
            format_top5(&zero)
        ),
        Err(reason) => {
            failed = true;
            println!("Random combo (0, 0, 0, 0): FAIL ({reason})");
        }
    }

    let aa = cache.lookup(49, 49, 49, 49);
    let aa_sum: f64 = aa.iter().sum();
    let uniform = aa
        .iter()
        .all(|&p| (p - 1.0 / NUM_PERMS as f64).abs() <= 0.1);
    if !uniform {
        failed = true;
    }
    match check_combo(&cache, [49, 49, 49, 49]) {
        Ok(_) => println!("Random combo (49, 49, 49, 49): sum = {aa_sum:.2}, uniform = {uniform}"),
        Err(reason) => {
            failed = true;
            println!("Random combo (49, 49, 49, 49): FAIL ({reason})");
        }
    }

    let symmetry_ok = check_symmetry(&cache);
    if !symmetry_ok {
        failed = true;
    }
    println!(
        "Symmetry check: {}",
        if symmetry_ok { "OK" } else { "FAIL" }
    );
    if random_ok < 20 {
        println!("Random combos: {random_ok}/20 OK");
    }
    println!();

    if failed {
        println!("Validation: FAILED.");
        Err("validation failed".to_string())
    } else {
        println!("Validation: PASSED.");
        Ok(())
    }
}

fn check_combo(cache: &FourWayRankCache, combo: [u8; 4]) -> Result<f64, String> {
    let dist = cache.lookup(combo[0], combo[1], combo[2], combo[3]);
    let sum: f64 = dist.iter().sum();
    if (sum - 1.0).abs() > 0.05 {
        return Err(format!("sum {sum:.4}"));
    }
    if dist.iter().any(|&p| !(0.0..=1.0).contains(&p)) {
        return Err("probability out of range".to_string());
    }
    let mut sorted = dist;
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    if (0..4).any(|i| sorted[i] + 1e-12 < sorted[i + 1]) {
        return Err("top-5 not descending".to_string());
    }
    Ok(sum)
}

fn format_top5(dist: &[f64; NUM_PERMS]) -> String {
    let mut sorted = *dist;
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    sorted
        .iter()
        .take(5)
        .map(|p| format!("{p:.2}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn check_symmetry(cache: &FourWayRankCache) -> bool {
    let forward = cache.lookup(0, 1, 2, 3);
    let reversed = cache.lookup(3, 2, 1, 0);
    (0..NUM_PERMS).all(|perm| {
        let mapped = reverse_perm_index(perm);
        (forward[perm] - reversed[mapped]).abs() <= 0.1
    })
}

fn reverse_perm_index(perm: usize) -> usize {
    let order = RANK_PERMUTATIONS[perm];
    let mapped = [3 - order[0], 3 - order[1], 3 - order[2], 3 - order[3]];
    RANK_PERMUTATIONS
        .iter()
        .position(|&candidate| candidate == mapped)
        .expect("reversed permutation exists")
}

fn random_combo(rng: &mut Rng) -> [u8; 4] {
    [
        rng.gen_range(NUM_BUCKETS) as u8,
        rng.gen_range(NUM_BUCKETS) as u8,
        rng.gen_range(NUM_BUCKETS) as u8,
        rng.gen_range(NUM_BUCKETS) as u8,
    ]
}

fn with_commas(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, ch) in digits.chars().rev().enumerate() {
        if i != 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn gen_range(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max
    }
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
  build_4way_cache --validate PATH

Examples:
  build_4way_cache --bucketing bucketing.bin --cache equity_cache.bin --output four_way_rank_cache.bin --samples 20
  build_4way_cache --bucketing bucketing.bin --cache equity_cache.bin --output test.bin --samples 20 --limit 1000
  build_4way_cache --validate four_way_rank_cache.bin"
    );
}
