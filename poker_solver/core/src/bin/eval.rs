use std::env;
use std::process::ExitCode;
use std::str::FromStr;

use poker_core::{evaluate_hand, Card};

fn main() -> ExitCode {
    let input: String = env::args().skip(1).collect::<Vec<_>>().join(" ");

    if input.trim().is_empty() {
        eprintln!("usage: eval <7 cards, e.g. \"As Ks Qs Js Ts 2c 3d\">");
        return ExitCode::FAILURE;
    }

    let tokens: Vec<&str> = input.split_whitespace().collect();
    if tokens.len() != 7 {
        eprintln!("expected exactly 7 cards, got {}: {}", tokens.len(), input);
        return ExitCode::FAILURE;
    }

    let mut cards = [Card::new(2, 0); 7];
    for (index, token) in tokens.iter().enumerate() {
        match Card::from_str(token) {
            Ok(card) => cards[index] = card,
            Err(error) => {
                eprintln!("failed to parse card '{token}': {error:?}");
                return ExitCode::FAILURE;
            }
        }
    }

    let hand_rank = evaluate_hand(&cards);
    println!(
        "cards: {input}\ncategory: {:?}\nvalue: {}",
        hand_rank.category(),
        hand_rank.value()
    );

    ExitCode::SUCCESS
}
