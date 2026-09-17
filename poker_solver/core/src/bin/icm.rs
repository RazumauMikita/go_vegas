use std::env;
use std::process::ExitCode;

use poker_core::icm_equity;

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

    let mut stacks = None;
    let mut payouts = None;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--stacks" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--stacks requires a comma-separated list".to_string())?;
                stacks = Some(parse_number_list(value, "stack")?);
            }
            "--payouts" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--payouts requires a comma-separated list".to_string())?;
                payouts = Some(parse_number_list(value, "payout")?);
            }
            token => return Err(format!("unknown argument: {token}")),
        }
        index += 1;
    }

    let stacks = stacks.ok_or_else(|| "missing --stacks".to_string())?;
    let payouts = payouts.ok_or_else(|| "missing --payouts".to_string())?;

    if stacks.is_empty() {
        return Err("stacks list cannot be empty".to_string());
    }
    if payouts.is_empty() {
        return Err("payouts list cannot be empty".to_string());
    }
    if stacks.iter().any(|&stack| stack < 0.0) {
        return Err("stacks must be non-negative".to_string());
    }
    if !stacks.iter().any(|&stack| stack > 0.0) {
        return Err("at least one stack must be positive".to_string());
    }

    let payout_sum: f64 = payouts.iter().sum();
    if (payout_sum - 1.0).abs() > 1e-9 {
        return Err(format!("payouts must sum to 1.0, got {payout_sum}"));
    }

    let equities = icm_equity(&stacks, &payouts);
    print_results(&stacks, &payouts, &equities);

    Ok(())
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

fn print_results(stacks: &[f64], payouts: &[f64], equities: &[f64]) {
    let stack_sum: f64 = stacks.iter().sum();

    println!("Stacks:  {}", format_list(stacks));
    println!("Payouts: {}", format_list(payouts));
    println!("Total stack: {stack_sum:.0}");
    println!();

    for (index, (&stack, &equity)) in stacks.iter().zip(equities.iter()).enumerate() {
        let chip_share = if stack_sum > 0.0 {
            stack / stack_sum * 100.0
        } else {
            0.0
        };
        println!(
            "Player {}: stack {:>8.0} ({:>5.1}% chips) -> $EV {:>6.2}%",
            index + 1,
            stack,
            chip_share,
            equity * 100.0,
        );
    }

    let total_ev: f64 = equities.iter().sum();
    println!();
    println!("Sum of $EV: {:.4}%", total_ev * 100.0);
}

fn format_list(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| {
            if value.fract() == 0.0 {
                format!("{value:.0}")
            } else {
                format!("{value}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn print_usage() {
    eprintln!(
        "usage:
  icm --stacks 5000,3000,2000 --payouts 0.5,0.3,0.2
  icm --stacks 1000,1000 --payouts 0.65,0.35

Notes:
  - stacks: chip stacks for each player
  - payouts: prize pool fractions, must sum to 1.0
  - output $EV is share of total prize pool (sum = 100%)"
    );
}
