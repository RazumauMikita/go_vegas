use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use poker_core::{
    combo_index, combo_label, index_to_ranks, Card, EquityCache, SolverInput, SolverOutput,
    ThreeMaxRanges,
};

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
    let cache = EquityCache::load(&config.cache).map_err(|error| {
        format!(
            "failed to load cache from {}: {error}",
            config.cache.display()
        )
    })?;

    let input = SolverInput {
        stacks: config.stacks,
        payouts: config.payouts,
        small_blind: config.small_blind,
        big_blind: config.big_blind,
        ante: config.ante,
        button_index: config.button_index,
        max_iterations: config.max_iterations,
        tolerance: config.tolerance,
        num_players: config.num_players,
        verbose_convergence: config.verbose_convergence,
    };

    let output = poker_core::solve(&input, &cache);
    print_output(&input, &output);

    Ok(())
}

struct Config {
    stacks: Vec<f64>,
    payouts: Vec<f64>,
    small_blind: f64,
    big_blind: f64,
    ante: f64,
    button_index: usize,
    max_iterations: usize,
    tolerance: f64,
    num_players: usize,
    cache: PathBuf,
    verbose_convergence: bool,
}

fn parse_args(args: Vec<String>) -> Result<Config, String> {
    let mut stacks = None;
    let mut payouts = None;
    let mut blinds = None;
    let mut ante = 0.0;
    let mut button_index = 0_usize;
    let mut max_iterations = 50_usize;
    let mut tolerance = 0.001;
    let mut cache = PathBuf::from("equity_cache.bin");
    let mut verbose_convergence = false;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--stacks" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--stacks requires a value".to_string())?;
                stacks = Some(parse_number_list(value, "stack")?);
            }
            "--payouts" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--payouts requires a value".to_string())?;
                payouts = Some(parse_number_list(value, "payout")?);
            }
            "--blinds" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--blinds requires a value".to_string())?;
                blinds = Some(parse_number_list(value, "blind")?);
            }
            "--ante" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--ante requires a value".to_string())?;
                ante = value
                    .parse()
                    .map_err(|_| format!("invalid ante value: {value}"))?;
            }
            "--button" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--button requires a value".to_string())?;
                button_index = value
                    .parse()
                    .map_err(|_| format!("invalid button value: {value}"))?;
            }
            "--iterations" | "--max-iterations" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--iterations requires a value".to_string())?;
                max_iterations = value
                    .parse()
                    .map_err(|_| format!("invalid iterations value: {value}"))?;
            }
            "--tolerance" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--tolerance requires a value".to_string())?;
                tolerance = value
                    .parse()
                    .map_err(|_| format!("invalid tolerance value: {value}"))?;
            }
            "--cache" => {
                index += 1;
                cache = PathBuf::from(
                    args.get(index)
                        .ok_or_else(|| "--cache requires a path".to_string())?,
                );
            }
            "--verbose-convergence" => {
                verbose_convergence = true;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        index += 1;
    }

    let stacks = stacks.ok_or_else(|| "missing --stacks".to_string())?;
    let payouts = payouts.ok_or_else(|| "missing --payouts".to_string())?;
    let blinds = blinds.ok_or_else(|| "missing --blinds".to_string())?;

    if stacks.len() != 2 && stacks.len() != 3 {
        return Err("solver requires 2 or 3 stacks".to_string());
    }
    if blinds.len() != 2 {
        return Err("--blinds requires SB,BB".to_string());
    }
    if button_index >= stacks.len() {
        return Err("button index out of range".to_string());
    }
    if max_iterations == 0 {
        return Err("iterations must be greater than zero".to_string());
    }
    if tolerance <= 0.0 {
        return Err("tolerance must be greater than zero".to_string());
    }

    Ok(Config {
        num_players: stacks.len(),
        stacks,
        payouts,
        small_blind: blinds[0],
        big_blind: blinds[1],
        ante,
        button_index,
        max_iterations,
        tolerance,
        cache,
        verbose_convergence,
    })
}

fn parse_number_list(input: &str, label: &str) -> Result<Vec<f64>, String> {
    let values: Result<Vec<f64>, String> = input
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<f64>()
                .map_err(|_| format!("invalid {label} value: {part}"))
        })
        .collect();

    let values = values?;
    if values.is_empty() {
        return Err(format!("{label} list cannot be empty"));
    }
    Ok(values)
}

fn print_output(input: &SolverInput, output: &SolverOutput) {
    if input.stacks.len() == 3 {
        if let Some(ranges) = output.three_max.as_ref() {
            print_output_3max(input, output, ranges);
            return;
        }
    }
    print_output_hu(input, output);
}

fn print_output_hu(input: &SolverInput, output: &SolverOutput) {
    let sb = input.button_index;
    let bb = 1 - sb;

    println!("Players: {}", input.stacks.len());
    println!("Stacks:  SB={} BB={}", input.stacks[sb], input.stacks[bb]);
    println!(
        "Blinds:  {}/{} ante={}",
        input.small_blind, input.big_blind, input.ante
    );
    println!(
        "Iterations: {}  converged={}",
        output.iterations_used, output.converged
    );
    println!();
    println!("SB $EV: {:.2}%", output.equities[sb] * 100.0);
    println!("BB $EV: {:.2}%", output.equities[bb] * 100.0);
    println!();
    print_range_share("SB push", &output.push_ranges[sb]);
    print_range_share("BB call", &output.call_ranges[bb]);
    println!();

    println!("SB top-20 push hands:");
    print_top_hands(&output.push_ranges[sb], 20);
    println!();
    println!("BB top-20 call hands:");
    print_top_hands(&output.call_ranges[bb], 20);
    println!();

    println!("SB push 13x13 (A..2, suited above diagonal, offsuit below):");
    print_matrix(&output.push_ranges[sb]);
    println!();
    println!("BB call 13x13 (A..2, suited above diagonal, offsuit below):");
    print_matrix(&output.call_ranges[bb]);
}

fn print_output_3max(input: &SolverInput, output: &SolverOutput, ranges: &ThreeMaxRanges) {
    let btn = input.button_index;
    let sb = (btn + 1) % 3;
    let bb = (btn + 2) % 3;

    println!("Players: 3");
    println!(
        "Stacks:  BTN={} SB={} BB={}",
        input.stacks[btn], input.stacks[sb], input.stacks[bb]
    );
    println!(
        "Blinds:  {}/{} ante={}",
        input.small_blind, input.big_blind, input.ante
    );
    println!(
        "Iterations: {}  converged={}",
        output.iterations_used, output.converged
    );
    println!();
    println!("BTN $EV: {:.2}%", output.equities[btn] * 100.0);
    println!("SB $EV:  {:.2}%", output.equities[sb] * 100.0);
    println!("BB $EV:  {:.2}%", output.equities[bb] * 100.0);
    println!();
    print_range_share("BTN push", &ranges.btn_push);
    print_range_share("SB call vs BTN push", &ranges.sb_call_vs_btn);
    print_range_share("BB call vs BTN push (SB fold)", &ranges.bb_call_vs_btn);
    print_range_share(
        "BB call vs BTN push (SB call)",
        &ranges.bb_call_vs_btn_and_sb,
    );
    print_range_share("SB push (BTN fold)", &ranges.sb_push);
    print_range_share("BB call vs SB push", &ranges.bb_call_vs_sb);
    println!();

    println!("BTN push 13x13 (A..2, suited above diagonal, offsuit below):");
    print_matrix(&ranges.btn_push);
    println!();
    println!("SB call vs BTN 13x13:");
    print_matrix(&ranges.sb_call_vs_btn);
    println!();
    println!("BB call vs BTN (SB fold) 13x13:");
    print_matrix(&ranges.bb_call_vs_btn);
    println!();
    println!("BB call vs BTN+SB 13x13:");
    print_matrix(&ranges.bb_call_vs_btn_and_sb);
    println!();
    println!("SB push (BTN fold) 13x13:");
    print_matrix(&ranges.sb_push);
    println!();
    println!("BB call vs SB 13x13:");
    print_matrix(&ranges.bb_call_vs_sb);
}

fn print_range_share(label: &str, range: &[f64; 169]) {
    let combo_share = combo_share(range);
    let type_share = range.iter().sum::<f64>() / range.len() as f64;
    println!(
        "{label}: combo-share {:.1}%  type-share {:.1}%",
        combo_share * 100.0,
        type_share * 100.0
    );
}

fn combo_share(range: &[f64; 169]) -> f64 {
    const TOTAL_COMBOS: f64 = 1326.0;
    let weighted: f64 = range
        .iter()
        .enumerate()
        .map(|(idx, &freq)| freq.clamp(0.0, 1.0) * combo_weight(idx as u8) as f64)
        .sum();
    weighted / TOTAL_COMBOS
}

fn combo_weight(idx: u8) -> u8 {
    let (high, low, suited) = index_to_ranks(idx);
    if high == low {
        6
    } else if suited {
        4
    } else {
        12
    }
}

fn print_top_hands(range: &[f64; 169], count: usize) {
    let mut hands: Vec<(u8, f64)> = range
        .iter()
        .enumerate()
        .map(|(idx, &freq)| (idx as u8, freq))
        .collect();
    hands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (rank, (idx, freq)) in hands.into_iter().take(count).enumerate() {
        println!("  {:>2}. {:<4}  {:.3}", rank + 1, combo_label(idx), freq);
    }
}

fn print_matrix(range: &[f64; 169]) {
    for row in 0..13_u8 {
        let mut cells = Vec::with_capacity(13);
        for col in 0..13_u8 {
            let idx = matrix_combo_index(row, col);
            cells.push(format!("{:.1}", range[idx as usize]));
        }
        println!("{}", cells.join(" "));
    }
}

fn matrix_combo_index(row: u8, col: u8) -> u8 {
    let rank_row = 14 - row;
    let rank_col = 14 - col;
    if row == col {
        combo_index([Card::new(rank_row, 0), Card::new(rank_row, 1)])
    } else if row < col {
        combo_index([Card::new(rank_row, 0), Card::new(rank_col, 0)])
    } else {
        combo_index([Card::new(rank_row, 0), Card::new(rank_col, 1)])
    }
}

fn print_usage() {
    eprintln!(
        "usage:
  solve --stacks 1000,1000 --payouts 0.5,0.3 --blinds 50,100 --cache equity_cache.bin
  solve --stacks 1000,1000,1000 --payouts 0.5,0.3,0.2 --blinds 50,100 --cache equity_cache.bin

Options:
  --ante N
  --button N
  --iterations N   (default 50)
  --tolerance X    (default 0.001)"
    );
}
