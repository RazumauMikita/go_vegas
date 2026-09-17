use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use poker_core::EquityCache;

const AA: u8 = 0;
const KK: u8 = 1;

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
    let path = parse_path(args)?;
    let cache = EquityCache::load(&path)
        .map_err(|error| format!("failed to load cache from {}: {error}", path.display()))?;

    let aa_vs_kk = cache.equity(AA, KK);
    let kk_vs_aa = cache.equity(KK, AA);
    let aa_vs_aa = cache.equity(AA, AA);

    println!("Cache: {}", path.display());
    println!("Size:  {} bytes", 169 * 169 * 4);
    println!();
    println!("equity(AA, KK) = {aa_vs_kk:.4}  (expected ~0.81 ±0.02)");
    println!("equity(KK, AA) = {kk_vs_aa:.4}  (expected ~0.19 ±0.02)");
    println!("equity(AA, AA) = {aa_vs_aa:.4}  (same-type pair, not 1.0)");
    println!();

    let aa_kk_ok = (aa_vs_kk - 0.81).abs() <= 0.02;
    let kk_aa_ok = (kk_vs_aa - 0.19).abs() <= 0.02;
    let symmetry_ok = (aa_vs_kk + kk_vs_aa - 1.0).abs() <= 0.01;

    if aa_kk_ok && kk_aa_ok && symmetry_ok {
        println!("Result: OK — cache looks correct.");
        Ok(())
    } else {
        println!("Result: MISMATCH — check cache generation.");
        if !aa_kk_ok {
            println!("  - equity(AA, KK) out of range");
        }
        if !kk_aa_ok {
            println!("  - equity(KK, AA) out of range");
        }
        if !symmetry_ok {
            println!(
                "  - symmetry broken: AA/KK + KK/AA = {:.4}",
                aa_vs_kk + kk_vs_aa
            );
        }
        Err("cache verification failed".to_string())
    }
}

fn parse_path(args: Vec<String>) -> Result<PathBuf, String> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Err(
            "usage: verify_cache [PATH]\n  default: equity_cache.bin in current directory".into(),
        );
    }

    if let Some(path) = args.first() {
        return Ok(PathBuf::from(path));
    }

    Ok(PathBuf::from("equity_cache.bin"))
}
