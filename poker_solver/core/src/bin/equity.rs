use std::env;
use std::process::ExitCode;
use std::str::FromStr;

use poker_core::{equity_exact, equity_monte_carlo, parse_range, Card, EquityResult, HandRange};

const DEFAULT_ITERATIONS: u64 = 100_000;

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

    let mut hero_arg = None;
    let mut villain_arg = None;
    let mut board_tokens = Vec::new();
    let mut force_exact = false;
    let mut force_mc = false;
    let mut iterations = DEFAULT_ITERATIONS;

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--board" => {
                index += 1;
                while index < args.len() && !args[index].starts_with("--") {
                    board_tokens.push(args[index].clone());
                    index += 1;
                }
                continue;
            }
            "--exact" => force_exact = true,
            "--mc" => force_mc = true,
            "--iterations" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--iterations requires a value".to_string())?;
                iterations = value
                    .parse()
                    .map_err(|_| format!("invalid iterations value: {value}"))?;
                if iterations == 0 {
                    return Err("iterations must be greater than zero".to_string());
                }
            }
            token if hero_arg.is_none() => hero_arg = Some(token.to_string()),
            token if villain_arg.is_none() => villain_arg = Some(token.to_string()),
            token => return Err(format!("unexpected argument: {token}")),
        }
        index += 1;
    }

    let hero_text = hero_arg.ok_or_else(|| "missing hero hand or range".to_string())?;
    let villain_text = villain_arg.ok_or_else(|| "missing villain hand or range".to_string())?;

    let hero_range = parse_hand_or_range(&hero_text)?;
    let villain_range = parse_hand_or_range(&villain_text)?;
    let board = parse_board(&board_tokens)?;

    if ranges_overlap(&hero_range, &villain_range) {
        return Err("hero and villain ranges contain overlapping cards".to_string());
    }

    let hero_is_specific = is_specific_hand(&hero_text);
    let villain_is_specific = is_specific_hand(&villain_text);
    let use_exact = if force_mc {
        false
    } else if force_exact {
        true
    } else {
        hero_is_specific && villain_is_specific
    };

    let ranges = [hero_range, villain_range];
    let results = if use_exact {
        equity_exact(&ranges, &board)
    } else {
        equity_monte_carlo(&ranges, &board, iterations)
    };

    print_results(
        &hero_text,
        &villain_text,
        &board,
        &results,
        use_exact,
        iterations,
    );

    Ok(())
}

fn parse_hand_or_range(input: &str) -> Result<HandRange, String> {
    let tokens: Vec<&str> = input.split_whitespace().collect();
    if tokens.len() == 2 {
        let first = Card::from_str(tokens[0])
            .map_err(|error| format!("failed to parse card '{}': {error:?}", tokens[0]))?;
        let second = Card::from_str(tokens[1])
            .map_err(|error| format!("failed to parse card '{}': {error:?}", tokens[1]))?;
        return Ok(vec![[first, second]]);
    }

    let range = parse_range(input);
    if range.is_empty() {
        return Err(format!("invalid hand or range notation: {input}"));
    }

    Ok(range)
}

fn is_specific_hand(input: &str) -> bool {
    input.split_whitespace().count() == 2
}

fn parse_board(tokens: &[String]) -> Result<Vec<Card>, String> {
    if tokens.len() > 5 {
        return Err(format!(
            "board cannot contain more than 5 cards, got {}",
            tokens.len()
        ));
    }

    let mut board = Vec::with_capacity(tokens.len());
    for token in tokens {
        let card = Card::from_str(token)
            .map_err(|error| format!("failed to parse board card '{token}': {error:?}"))?;
        if board.contains(&card) {
            return Err(format!("duplicate board card: {token}"));
        }
        board.push(card);
    }

    Ok(board)
}

fn ranges_overlap(left: &HandRange, right: &HandRange) -> bool {
    for hand_left in left {
        for hand_right in right {
            if hands_overlap(*hand_left, *hand_right) {
                return true;
            }
        }
    }
    false
}

fn hands_overlap(h1: [Card; 2], h2: [Card; 2]) -> bool {
    h1[0] == h2[0] || h1[0] == h2[1] || h1[1] == h2[0] || h1[1] == h2[1]
}

fn print_results(
    hero: &str,
    villain: &str,
    board: &[Card],
    results: &[EquityResult],
    exact: bool,
    iterations: u64,
) {
    println!("Hero:    {hero}");
    println!("Villain: {villain}");
    if board.is_empty() {
        println!("Board:   (preflop)");
    } else {
        let board_text: Vec<String> = board.iter().map(|card| card.to_string()).collect();
        println!("Board:   {}", board_text.join(" "));
    }

    if exact {
        println!("Method:  exact");
    } else {
        println!("Method:  monte carlo ({iterations} iterations)");
    }

    print_player_line("Hero", &results[0]);
    if results.len() > 1 {
        print_player_line("Villain", &results[1]);
    }
}

fn print_player_line(label: &str, result: &EquityResult) {
    println!(
        "{label}: equity {:.2}% | win {:.2}% | tie {:.2}% | lose {:.2}%",
        result.equity() * 100.0,
        result.win * 100.0,
        result.tie * 100.0,
        result.lose * 100.0,
    );
}

fn print_usage() {
    eprintln!(
        "usage:
  equity \"Ah Ad\" \"Kh Kd\"
  equity \"Ah Ad\" \"Kh Kd\" --board Ts 7d 2c
  equity \"AA\" \"KK\"
  equity \"Ah Ad\" \"KK\" --mc --iterations 50000
  equity \"Ah Ad\" \"Kh Kd\" --exact

Notes:
  - Two cards in quotes = specific hand (uses exact enumeration by default)
  - Range notation = AA, AKs, AKo, 77+, AQo+
  - Ranges use Monte Carlo unless --exact is passed"
    );
}
