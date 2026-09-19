use regex::Regex;
use std::sync::LazyLock;

#[derive(Debug, Clone)]
pub struct ParsedHand {
    pub tournament_id: Option<String>,
    pub buy_in: Option<f64>,
    pub fee: Option<f64>,
    pub level: Option<String>,
    pub small_blind: f64,
    pub big_blind: f64,
    pub ante: f64,
    pub table_max: usize,
    pub button_seat: usize,
    pub players: Vec<ParsedPlayer>,
}

#[derive(Debug, Clone)]
pub struct ParsedPlayer {
    pub seat: usize,
    pub name: String,
    pub stack: f64,
    pub position: Position,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    BTN,
    SB,
    BB,
    UTG,
    UTG1,
    MP,
    MP1,
    HJ,
    CO,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    Empty,
    NotPokerStarsTournament,
    NoBlinds,
    NoPlayers,
    TooManyPlayers(usize),
    InvalidFormat(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::Empty => write!(f, "пустой ввод"),
            ParseError::NotPokerStarsTournament => {
                write!(f, "не распознан формат PokerStars Tournament")
            }
            ParseError::NoBlinds => write!(f, "не найдены блайнды"),
            ParseError::NoPlayers => write!(f, "не найдены игроки"),
            ParseError::TooManyPlayers(n) => {
                write!(
                    f,
                    "слишком много игроков ({n}), поддерживаются только 2 и 3"
                )
            }
            ParseError::InvalidFormat(msg) => write!(f, "некорректный формат: {msg}"),
        }
    }
}

impl std::error::Error for ParseError {}

static RE_TOURNAMENT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"Tournament #(\d+), \$([\d.]+)\+\$([\d.]+)").expect("valid regex")
});
static RE_LEVEL_BLINDS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"Level\s+(\S+)\s+\((\d+)/(\d+)(?:/(\d+))?\)").expect("valid regex")
});
static RE_BUTTON: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"Seat #(\d+) is the button").expect("valid regex"));
static RE_TABLE_MAX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\d+)-max").expect("valid regex"));
static RE_PLAYER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"Seat (\d+): (.+?) \((\d+) in chips\)").expect("valid regex"));
static RE_ANTE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r": posts the ante (\d+)").expect("valid regex"));

const POSITIONS_9MAX: [Position; 9] = [
    Position::BTN,
    Position::SB,
    Position::BB,
    Position::UTG,
    Position::UTG1,
    Position::MP,
    Position::MP1,
    Position::HJ,
    Position::CO,
];

pub fn parse_hand_history(text: &str) -> Result<ParsedHand, ParseError> {
    let text = text.trim();
    if text.is_empty() {
        return Err(ParseError::Empty);
    }

    if !text.contains("Tournament #") {
        return Err(ParseError::NotPokerStarsTournament);
    }

    let tournament_id = RE_TOURNAMENT.captures(text).map(|c| c[1].to_string());
    let buy_in = RE_TOURNAMENT.captures(text).and_then(|c| c[2].parse().ok());
    let fee = RE_TOURNAMENT.captures(text).and_then(|c| c[3].parse().ok());

    let blinds_cap = RE_LEVEL_BLINDS.captures(text).ok_or(ParseError::NoBlinds)?;
    let level = Some(blinds_cap[1].to_string());
    let small_blind = blinds_cap[2]
        .parse()
        .map_err(|_| ParseError::InvalidFormat("small blind".into()))?;
    let big_blind = blinds_cap[3]
        .parse()
        .map_err(|_| ParseError::InvalidFormat("big blind".into()))?;
    let header_ante = blinds_cap
        .get(4)
        .and_then(|m| m.as_str().parse().ok())
        .unwrap_or(0.0);

    let button_seat = RE_BUTTON
        .captures(text)
        .and_then(|c| c[1].parse().ok())
        .ok_or_else(|| ParseError::InvalidFormat("button seat".into()))?;

    let table_max = RE_TABLE_MAX
        .captures(text)
        .and_then(|c| c[1].parse().ok())
        .unwrap_or(9);

    let mut players = Vec::new();
    for cap in RE_PLAYER.captures_iter(text) {
        let seat = cap[1]
            .parse()
            .map_err(|_| ParseError::InvalidFormat("seat number".into()))?;
        let name = cap[2].to_string();
        let stack = cap[3]
            .parse()
            .map_err(|_| ParseError::InvalidFormat("stack".into()))?;
        players.push(ParsedPlayer {
            seat,
            name,
            stack,
            position: Position::Unknown,
        });
    }

    if players.is_empty() {
        return Err(ParseError::NoPlayers);
    }

    let player_count = players.len();
    if player_count > 3 {
        return Err(ParseError::TooManyPlayers(player_count));
    }
    if player_count < 2 {
        return Err(ParseError::InvalidFormat("нужно минимум 2 игрока".into()));
    }

    let ante = if header_ante > 0.0 {
        header_ante
    } else {
        parse_ante_from_posts(text)
    };

    assign_positions(&mut players, button_seat, table_max);

    Ok(ParsedHand {
        tournament_id,
        buy_in,
        fee,
        level,
        small_blind,
        big_blind,
        ante,
        table_max,
        button_seat,
        players,
    })
}

fn parse_ante_from_posts(text: &str) -> f64 {
    let mut ante_values: Vec<f64> = Vec::new();
    for cap in RE_ANTE.captures_iter(text) {
        if let Ok(value) = cap[1].parse::<f64>() {
            ante_values.push(value);
        }
    }
    if ante_values.is_empty() {
        return 0.0;
    }
    let first = ante_values[0];
    if ante_values.iter().all(|&v| (v - first).abs() < 1e-9) {
        first
    } else {
        0.0
    }
}

fn assign_positions(players: &mut [ParsedPlayer], button_seat: usize, table_max: usize) {
    let seats: Vec<usize> = players.iter().map(|p| p.seat).collect();
    let ordered = seats_clockwise_from_button(&seats, button_seat, table_max);
    let count = ordered.len();

    let positions: &[Position] = match count {
        2 => &[Position::BTN, Position::BB],
        3 => &[Position::BTN, Position::SB, Position::BB],
        _ => &POSITIONS_9MAX[..count.min(POSITIONS_9MAX.len())],
    };

    for player in players.iter_mut() {
        if let Some(idx) = ordered.iter().position(|&s| s == player.seat) {
            player.position = positions.get(idx).copied().unwrap_or(Position::Unknown);
        }
    }
}

fn seats_clockwise_from_button(
    seats: &[usize],
    button_seat: usize,
    table_max: usize,
) -> Vec<usize> {
    let max_seat = table_max;
    let mut ordered = Vec::with_capacity(seats.len());
    let mut current = button_seat;
    for _ in 0..max_seat {
        if seats.contains(&current) {
            ordered.push(current);
        }
        current = if current >= max_seat { 1 } else { current + 1 };
        if ordered.len() == seats.len() {
            break;
        }
    }
    ordered
}

/// Эвристика: раздача не является чистой push/fold ситуацией.
pub fn is_non_push_fold_situation(text: &str) -> bool {
    if text.contains("*** FLOP ***") {
        return true;
    }
    for line in text.lines() {
        if line.contains(": calls ") && !line.contains("all-in") {
            return true;
        }
    }
    false
}

/// Индекс позиции для солвера: 0=BTN (или SB в HU), 1=SB/BB, 2=BB.
pub fn solver_position_index(position: Position, num_players: usize) -> Option<usize> {
    match (num_players, position) {
        (2, Position::BTN) | (2, Position::SB) => Some(0),
        (2, Position::BB) => Some(1),
        (3, Position::BTN) => Some(0),
        (3, Position::SB) => Some(1),
        (3, Position::BB) => Some(2),
        _ => None,
    }
}

pub fn position_label(position: Position) -> &'static str {
    match position {
        Position::BTN => "BTN",
        Position::SB => "SB",
        Position::BB => "BB",
        Position::UTG => "UTG",
        Position::UTG1 => "UTG1",
        Position::MP => "MP",
        Position::MP1 => "MP1",
        Position::HJ => "HJ",
        Position::CO => "CO",
        Position::Unknown => "?",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASIC_HH: &str = r"PokerStars Hand #262106673632: Tournament #4032582650, $0.85+$0.15 USD Hold'em No Limit - Level VII (100/200)

2026/09/16 20:18:50 ET
Table '4032582650 1' 9-max Seat #4 is the button
Seat 2: bbcbcv3 (1807 in chips)
Seat 4: grande.h74 (8601 in chips)
Seat 9: SeanPL41 (3092 in chips)
bbcbcv3: posts the ante 25
grande.h74: posts the ante 25
SeanPL41: posts the ante 25
SeanPL41: posts small blind 100
bbcbcv3: posts big blind 200
*** HOLE CARDS ***
Dealt to grande.h74 [Ah Kd]
grande.h74: raises 400 to 600
SeanPL41: calls 500
bbcbcv3: folds
*** FLOP *** [2c 7h Ts]
";

    #[test]
    fn parse_basic_example() {
        let hand = parse_hand_history(BASIC_HH).expect("should parse");
        assert_eq!(hand.tournament_id.as_deref(), Some("4032582650"));
        assert_eq!(hand.buy_in, Some(0.85));
        assert_eq!(hand.fee, Some(0.15));
        assert_eq!(hand.small_blind, 100.0);
        assert_eq!(hand.big_blind, 200.0);
        assert_eq!(hand.ante, 25.0);
        assert_eq!(hand.table_max, 9);
        assert_eq!(hand.button_seat, 4);
        assert_eq!(hand.players.len(), 3);

        let stacks: Vec<f64> = hand.players.iter().map(|p| p.stack).collect();
        assert!(stacks.contains(&1807.0));
        assert!(stacks.contains(&8601.0));
        assert!(stacks.contains(&3092.0));
    }

    #[test]
    fn positions_3max() {
        let hand = parse_hand_history(BASIC_HH).expect("should parse");
        let btn = hand
            .players
            .iter()
            .find(|p| p.position == Position::BTN)
            .expect("BTN");
        assert_eq!(btn.seat, 4);
        assert_eq!(btn.name, "grande.h74");

        let sb = hand
            .players
            .iter()
            .find(|p| p.position == Position::SB)
            .expect("SB");
        assert_eq!(sb.seat, 9);
        assert_eq!(sb.name, "SeanPL41");

        let bb = hand
            .players
            .iter()
            .find(|p| p.position == Position::BB)
            .expect("BB");
        assert_eq!(bb.seat, 2);
        assert_eq!(bb.name, "bbcbcv3");
    }

    #[test]
    fn empty_input_returns_error() {
        assert!(matches!(parse_hand_history(""), Err(ParseError::Empty)));
    }

    #[test]
    fn not_tournament_returns_error() {
        let text = "PokerStars Hand #xxx: Hold'em No Limit";
        assert!(matches!(
            parse_hand_history(text),
            Err(ParseError::NotPokerStarsTournament)
        ));
    }

    #[test]
    fn missing_blinds_returns_error() {
        let text = "PokerStars Hand #1: Tournament #123, $1+$0.10 USD Hold'em No Limit\nSeat 1: player (1000 in chips)";
        assert!(matches!(
            parse_hand_history(text),
            Err(ParseError::NoBlinds)
        ));
    }

    #[test]
    fn hu_two_players() {
        let text = r"PokerStars Hand #1: Tournament #999, $1+$0.10 USD Hold'em No Limit - Level I (25/50)
Table '1' 9-max Seat #3 is the button
Seat 3: Hero (1000 in chips)
Seat 7: Villain (1500 in chips)
Hero: posts small blind 25
Villain: posts big blind 50
*** HOLE CARDS ***";
        let hand = parse_hand_history(text).expect("should parse HU");
        assert_eq!(hand.players.len(), 2);

        let btn = hand.players.iter().find(|p| p.seat == 3).expect("button");
        assert_eq!(btn.position, Position::BTN);
        assert_eq!(btn.name, "Hero");

        let bb = hand.players.iter().find(|p| p.seat == 7).expect("bb");
        assert_eq!(bb.position, Position::BB);
        assert_eq!(bb.name, "Villain");
    }

    #[test]
    fn ante_zero_when_not_present() {
        let text = r"PokerStars Hand #1: Tournament #123, $1+$0.10 USD Hold'em No Limit - Level I (25/50)
Table '1' 9-max Seat #1 is the button
Seat 1: A (1000 in chips)
Seat 2: B (1000 in chips)
Seat 3: C (1000 in chips)
A: posts small blind 25
B: posts big blind 50
*** HOLE CARDS ***";
        let hand = parse_hand_history(text).expect("should parse");
        assert_eq!(hand.ante, 0.0);
    }

    #[test]
    fn detects_non_push_fold() {
        assert!(is_non_push_fold_situation(BASIC_HH));
    }
}
