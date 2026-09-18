use std::str::FromStr;

use poker_core::Card;

pub fn parse_number_list(input: &str, label: &str) -> Result<Vec<f64>, String> {
    let values: Result<Vec<f64>, String> = input
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<f64>()
                .map_err(|_| format!("некорректное значение {label}: {part}"))
        })
        .collect();

    let values = values?;
    if values.is_empty() {
        return Err(format!("список {label} не может быть пустым"));
    }
    Ok(values)
}

pub fn parse_hand(input: &str) -> Result<[Card; 2], String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("рука не задана".to_string());
    }

    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    if tokens.len() == 2 {
        let first = Card::from_str(tokens[0])
            .map_err(|error| format!("не удалось разобрать карту '{}': {:?}", tokens[0], error))?;
        let second = Card::from_str(tokens[1])
            .map_err(|error| format!("не удалось разобрать карту '{}': {:?}", tokens[1], error))?;
        if first == second {
            return Err("рука содержит одинаковые карты".to_string());
        }
        return Ok([first, second]);
    }

    if trimmed.len() == 4 {
        let first = Card::from_str(&trimmed[0..2]).map_err(|error| {
            format!(
                "не удалось разобрать карту '{}': {:?}",
                &trimmed[0..2],
                error
            )
        })?;
        let second = Card::from_str(&trimmed[2..4]).map_err(|error| {
            format!(
                "не удалось разобрать карту '{}': {:?}",
                &trimmed[2..4],
                error
            )
        })?;
        if first == second {
            return Err("рука содержит одинаковые карты".to_string());
        }
        return Ok([first, second]);
    }

    Err(format!("некорректный формат руки: {trimmed}"))
}

pub fn parse_board(input: &str) -> Result<Vec<Card>, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    if tokens.len() > 5 {
        return Err(format!(
            "борд не может содержать больше 5 карт, получено {}",
            tokens.len()
        ));
    }

    let mut board = Vec::with_capacity(tokens.len());
    for token in tokens {
        let card = Card::from_str(token)
            .map_err(|error| format!("не удалось разобрать карту борда '{token}': {:?}", error))?;
        if board.contains(&card) {
            return Err(format!("дублирующаяся карта на борде: {token}"));
        }
        board.push(card);
    }
    Ok(board)
}

pub fn parse_cards(input: &str, expected: usize) -> Result<Vec<Card>, String> {
    let tokens: Vec<&str> = input.split_whitespace().collect();
    if tokens.len() != expected {
        return Err(format!(
            "ожидается {expected} карт, получено {}",
            tokens.len()
        ));
    }

    let mut cards = Vec::with_capacity(expected);
    for token in tokens {
        let card = Card::from_str(token)
            .map_err(|error| format!("не удалось разобрать карту '{token}': {:?}", error))?;
        if cards.contains(&card) {
            return Err(format!("дублирующаяся карта: {token}"));
        }
        cards.push(card);
    }
    Ok(cards)
}

pub fn hands_overlap(h1: [Card; 2], h2: [Card; 2]) -> bool {
    h1[0] == h2[0] || h1[0] == h2[1] || h1[1] == h2[0] || h1[1] == h2[1]
}

pub fn format_duration(duration: std::time::Duration) -> String {
    let millis = duration.as_secs_f64() * 1000.0;
    if millis < 1000.0 {
        format!("{millis:.1} мс")
    } else {
        format!("{:.2} с", millis / 1000.0)
    }
}
