use regex::Regex;
use std::str::FromStr;
use std::sync::LazyLock;

use crate::card::Card;
use crate::equity_cache::{combo_index, combo_label};

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
    pub sitting_out: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeroActionKind {
    Fold,
    Call,
    Raise,
    Check,
}

impl HeroActionKind {
    pub fn code(self) -> &'static str {
        match self {
            Self::Fold => "F",
            Self::Call => "C",
            Self::Raise => "R",
            Self::Check => "X",
        }
    }

    pub fn is_fold(self) -> bool {
        matches!(self, Self::Fold | Self::Check)
    }

    pub fn is_aggressive(self) -> bool {
        matches!(self, Self::Raise | Self::Call)
    }
}

#[derive(Debug, Clone)]
pub struct ImportedAction {
    pub name: String,
    pub kind: HeroActionKind,
    pub all_in: bool,
}

#[derive(Debug, Clone)]
pub struct ImportedHand {
    pub hand_id: String,
    pub tournament_id: String,
    pub buy_in: f64,
    pub fee: f64,
    pub datetime: String,
    pub level_label: String,
    pub small_blind: f64,
    pub big_blind: f64,
    pub ante: f64,
    pub table_max: usize,
    pub button_seat: usize,
    pub players: Vec<ParsedPlayer>,
    pub hero_name: String,
    pub hero_cards: Option<[Card; 2]>,
    pub hero_combo: Option<u8>,
    pub hero_combo_label: Option<String>,
    pub hero_action: Option<HeroActionKind>,
    pub preflop_actions: Vec<ImportedAction>,
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
                    "слишком много игроков ({n}), поддерживаются 2–9"
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
static RE_HAND_ID: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"PokerStars Hand #(\d+)").expect("valid regex"));
static RE_DATETIME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(\d{4})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2}):(\d{2})").expect("valid regex")
});
static RE_DEALT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"Dealt to (.+?) \[(\S+) (\S+)\]").expect("valid regex"));

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

    let header = text.split("*** HOLE CARDS ***").next().unwrap_or(text);
    let mut players = Vec::new();
    for cap in RE_PLAYER.captures_iter(header) {
        let seat = cap[1]
            .parse()
            .map_err(|_| ParseError::InvalidFormat("seat number".into()))?;
        let name = cap[2].to_string();
        let stack = cap[3]
            .parse()
            .map_err(|_| ParseError::InvalidFormat("stack".into()))?;
        let match_end = cap.get(0).map(|m| m.end()).unwrap_or(0);
        let line_end = header[match_end..]
            .find('\n')
            .map(|i| match_end + i)
            .unwrap_or(header.len());
        let line_tail = &header[match_end..line_end];
        let sitting_out = line_tail.contains("is sitting out");
        if sitting_out {
            continue;
        }
        players.push(ParsedPlayer {
            seat,
            name,
            stack,
            position: Position::Unknown,
            sitting_out: false,
        });
    }

    if players.is_empty() {
        return Err(ParseError::NoPlayers);
    }

    let player_count = players.len();
    if player_count > 9 {
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
        4 => &[Position::BTN, Position::SB, Position::BB, Position::CO],
        5 => &[
            Position::BTN,
            Position::SB,
            Position::BB,
            Position::HJ,
            Position::CO,
        ],
        6 => &[
            Position::BTN,
            Position::SB,
            Position::BB,
            Position::UTG,
            Position::HJ,
            Position::CO,
        ],
        7 => &[
            Position::BTN,
            Position::SB,
            Position::BB,
            Position::UTG,
            Position::MP,
            Position::HJ,
            Position::CO,
        ],
        8 => &[
            Position::BTN,
            Position::SB,
            Position::BB,
            Position::UTG,
            Position::UTG1,
            Position::MP,
            Position::HJ,
            Position::CO,
        ],
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
        (4, Position::CO) => Some(0),
        (4, Position::BTN) => Some(1),
        (4, Position::SB) => Some(2),
        (4, Position::BB) => Some(3),
        (5, Position::HJ) => Some(0),
        (5, Position::CO) => Some(1),
        (5, Position::BTN) => Some(2),
        (5, Position::SB) => Some(3),
        (5, Position::BB) => Some(4),
        (6, Position::UTG) => Some(0),
        (6, Position::HJ) => Some(1),
        (6, Position::CO) => Some(2),
        (6, Position::BTN) => Some(3),
        (6, Position::SB) => Some(4),
        (6, Position::BB) => Some(5),
        (7, Position::UTG) => Some(0),
        (7, Position::MP) => Some(1),
        (7, Position::HJ) => Some(2),
        (7, Position::CO) => Some(3),
        (7, Position::BTN) => Some(4),
        (7, Position::SB) => Some(5),
        (7, Position::BB) => Some(6),
        (8, Position::UTG) => Some(0),
        (8, Position::UTG1) => Some(1),
        (8, Position::MP) => Some(2),
        (8, Position::HJ) => Some(3),
        (8, Position::CO) => Some(4),
        (8, Position::BTN) => Some(5),
        (8, Position::SB) => Some(6),
        (8, Position::BB) => Some(7),
        (9, Position::UTG) => Some(0),
        (9, Position::UTG1) => Some(1),
        (9, Position::MP) => Some(2),
        (9, Position::MP1) => Some(3),
        (9, Position::HJ) => Some(4),
        (9, Position::CO) => Some(5),
        (9, Position::BTN) => Some(6),
        (9, Position::SB) => Some(7),
        (9, Position::BB) => Some(8),
        _ => None,
    }
}

pub fn split_hand_histories(text: &str) -> Vec<&str> {
    let mut starts: Vec<usize> = text.match_indices("PokerStars Hand #").map(|(i, _)| i).collect();
    if starts.is_empty() {
        return Vec::new();
    }
    starts.push(text.len());
    starts
        .windows(2)
        .filter_map(|w| {
            let chunk = text[w[0]..w[1]].trim();
            (!chunk.is_empty()).then_some(chunk)
        })
        .collect()
}

pub fn parse_hand_histories(text: &str) -> Vec<ImportedHand> {
    split_hand_histories(text)
        .into_iter()
        .filter_map(|chunk| parse_imported_hand(chunk).ok())
        .collect()
}

pub fn parse_imported_hand(text: &str) -> Result<ImportedHand, ParseError> {
    let parsed = parse_hand_history(text)?;
    let tournament_id = parsed
        .tournament_id
        .clone()
        .ok_or(ParseError::NotPokerStarsTournament)?;

    let players = parsed.players;

    let hand_id = RE_HAND_ID
        .captures(text)
        .map(|c| c[1].to_string())
        .unwrap_or_default();

    let datetime = RE_DATETIME
        .captures(text)
        .map(|c| format!("{}-{}-{} {}:{}", &c[1], &c[2], &c[3], &c[4], &c[5]))
        .unwrap_or_default();

    let (hero_name, hero_cards) = match RE_DEALT.captures(text) {
        Some(cap) => {
            let name = cap[1].to_string();
            let cards = parse_hole_cards(&cap[2], &cap[3]);
            (name, cards)
        }
        None => (String::new(), None),
    };

    let hero_combo = hero_cards.map(combo_index);
    let hero_combo_label = hero_combo.map(combo_label);

    let preflop_end = text
        .find("*** FLOP ***")
        .or_else(|| text.find("*** SHOW DOWN ***"))
        .or_else(|| text.find("*** SUMMARY ***"))
        .unwrap_or(text.len());
    let preflop = text
        .find("*** HOLE CARDS ***")
        .map(|i| &text[i..preflop_end])
        .unwrap_or("");
    let preflop_actions = parse_preflop_actions(preflop);
    let hero_action = preflop_actions
        .iter()
        .find(|action| action.name == hero_name)
        .map(|action| action.kind);

    let level_label = if parsed.ante > 0.0 {
        format!(
            "{:.0}/{:.0}/{:.0}",
            parsed.small_blind, parsed.big_blind, parsed.ante
        )
    } else {
        format!("{:.0}/{:.0}", parsed.small_blind, parsed.big_blind)
    };

    Ok(ImportedHand {
        hand_id,
        tournament_id,
        buy_in: parsed.buy_in.unwrap_or(0.0),
        fee: parsed.fee.unwrap_or(0.0),
        datetime,
        level_label,
        small_blind: parsed.small_blind,
        big_blind: parsed.big_blind,
        ante: parsed.ante,
        table_max: parsed.table_max,
        button_seat: parsed.button_seat,
        players,
        hero_name,
        hero_cards,
        hero_combo,
        hero_combo_label,
        hero_action,
        preflop_actions,
    })
}

fn parse_hole_cards(first: &str, second: &str) -> Option<[Card; 2]> {
    let a = Card::from_str(first).ok()?;
    let b = Card::from_str(second).ok()?;
    Some([a, b])
}

fn parse_preflop_actions(preflop: &str) -> Vec<ImportedAction> {
    let mut actions = Vec::new();
    for line in preflop.lines() {
        let line = line.trim();
        let Some((name, rest)) = line.split_once(": ") else {
            continue;
        };
        let rest = rest.trim();
        let (kind, all_in) = if rest.starts_with("folds") {
            (HeroActionKind::Fold, false)
        } else if rest.starts_with("checks") {
            (HeroActionKind::Check, false)
        } else if rest.starts_with("calls") {
            (HeroActionKind::Call, rest.contains("all-in"))
        } else if rest.starts_with("raises") || rest.starts_with("bets") {
            (HeroActionKind::Raise, rest.contains("all-in"))
        } else {
            continue;
        };
        actions.push(ImportedAction {
            name: name.to_string(),
            kind,
            all_in,
        });
    }
    actions
}

impl ImportedHand {
    pub fn hero_player(&self) -> Option<&ParsedPlayer> {
        self.players.iter().find(|p| p.name == self.hero_name)
    }

    pub fn hero_stack(&self) -> Option<f64> {
        self.hero_player().map(|p| p.stack)
    }

    pub fn effective_bb(&self) -> f64 {
        let bb = self.big_blind.max(1e-9);
        let Some(hero) = self.hero_player() else {
            return 0.0;
        };
        let max_other = self
            .players
            .iter()
            .filter(|p| p.name != self.hero_name)
            .map(|p| p.stack)
            .fold(0.0_f64, f64::max);
        if max_other <= 0.0 {
            hero.stack / bb
        } else {
            hero.stack.min(max_other) / bb
        }
    }

    pub fn hero_stack_bb(&self) -> f64 {
        let bb = self.big_blind.max(1e-9);
        self.hero_stack().unwrap_or(0.0) / bb
    }

    pub fn player_action(&self, name: &str) -> Option<HeroActionKind> {
        self.preflop_actions
            .iter()
            .find(|action| action.name == name)
            .map(|action| action.kind)
    }

    pub fn solver_stacks(&self) -> Option<Vec<f64>> {
        let n = self.players.len();
        if !(2..=9).contains(&n) {
            return None;
        }
        let mut stacks = vec![0.0; n];
        for player in &self.players {
            let index = solver_position_index(player.position, n)?;
            stacks[index] = player.stack;
        }
        if stacks.iter().any(|&s| s <= 0.0) {
            return None;
        }
        Some(stacks)
    }
}

/// Метка позиции в порядке стеков солвера (0 = первый актёр, кроме HU/3-max BTN).
pub fn table_position_label(players: usize, index: usize) -> &'static str {
    match (players, index) {
        (2, 0) => "SB",
        (2, _) => "BB",
        (4, 0) => "CO",
        (4, 1) => "BTN",
        (4, 2) => "SB",
        (4, _) => "BB",
        (5, 0) => "HJ",
        (5, 1) => "CO",
        (5, 2) => "BTN",
        (5, 3) => "SB",
        (5, _) => "BB",
        (6, 0) => "UTG",
        (6, 1) => "HJ",
        (6, 2) => "CO",
        (6, 3) => "BTN",
        (6, 4) => "SB",
        (6, _) => "BB",
        (7, 0) => "UTG",
        (7, 1) => "MP",
        (7, 2) => "HJ",
        (7, 3) => "CO",
        (7, 4) => "BTN",
        (7, 5) => "SB",
        (7, _) => "BB",
        (8, 0) => "UTG",
        (8, 1) => "EP",
        (8, 2) => "MP",
        (8, 3) => "HJ",
        (8, 4) => "CO",
        (8, 5) => "BTN",
        (8, 6) => "SB",
        (8, _) => "BB",
        (9, 0) => "UTG",
        (9, 1) => "EP",
        (9, 2) => "MP1",
        (9, 3) => "MP2",
        (9, 4) => "HJ",
        (9, 5) => "CO",
        (9, 6) => "BTN",
        (9, 7) => "SB",
        (9, _) => "BB",
        (_, 0) => "BTN",
        (_, 1) => "SB",
        (_, _) => "BB",
    }
}

pub fn solver_seat_label(position: Position, num_players: usize) -> Option<&'static str> {
    let index = solver_position_index(position, num_players)?;
    Some(table_position_label(num_players, index))
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

    #[test]
    fn imported_hand_extracts_hero_cards_and_action() {
        let text = r"PokerStars Hand #262256461738: Tournament #4036170758, $0.42+$0.08 USD Hold'em No Limit - Level I (10/20) - 2026/09/30 18:40:06 ET
Table '4036170758 1' 9-max Seat #1 is the button
Seat 1: indymumma (1500 in chips)
Seat 2: PRRP2010 (1500 in chips)
Seat 3: Melo7Melo (1500 in chips)
Seat 4: bbcbcv3 (1500 in chips)
Seat 5: Jokasei (1500 in chips)
Seat 6: Vhferraz95 (1500 in chips)
Seat 7: camargo18 (1500 in chips)
Seat 8: cloudfarter (1500 in chips)
Seat 9: leosteinke (1500 in chips)
PRRP2010: posts small blind 10
Melo7Melo: posts big blind 20
*** HOLE CARDS ***
Dealt to bbcbcv3 [As Kc]
bbcbcv3: raises 20 to 40
Jokasei: folds
Vhferraz95: folds
camargo18: folds
cloudfarter: raises 40 to 80
leosteinke: folds
indymumma: folds
PRRP2010: calls 70
Melo7Melo: calls 60
bbcbcv3: raises 1420 to 1500 and is all-in
*** SUMMARY ***
Total pot 3160 | Rake 0
";
        let hand = parse_imported_hand(text).expect("imported");
        assert_eq!(hand.hand_id, "262256461738");
        assert_eq!(hand.tournament_id, "4036170758");
        assert_eq!(hand.datetime, "2026-09-30 18:40");
        assert_eq!(hand.hero_name, "bbcbcv3");
        assert_eq!(hand.hero_combo_label.as_deref(), Some("AKo"));
        assert_eq!(hand.hero_action, Some(HeroActionKind::Raise));
        assert_eq!(hand.players.len(), 9);
        let hero = hand.hero_player().expect("hero");
        assert_eq!(hero.position, Position::UTG);
        assert!((hand.hero_stack_bb() - 75.0).abs() < 1e-9);
    }

    #[test]
    fn split_multiple_hands() {
        let text = format!("{BASIC_HH}\n\n{BASIC_HH}");
        assert_eq!(split_hand_histories(&text).len(), 2);
        assert_eq!(parse_hand_histories(&text).len(), 2);
    }

    #[test]
    fn sitting_out_players_are_skipped() {
        let text = r"PokerStars Hand #1: Tournament #123, $1+$0.10 USD Hold'em No Limit - Level I (25/50)
Table '1' 9-max Seat #1 is the button
Seat 1: A (1000 in chips)
Seat 2: B (1000 in chips) is sitting out
Seat 3: C (1000 in chips)
A: posts small blind 25
C: posts big blind 50
*** HOLE CARDS ***
Dealt to A [Ah Kd]
A: raises 50 to 75
C: folds
*** SUMMARY ***
";
        let hand = parse_imported_hand(text).expect("imported");
        assert_eq!(hand.players.len(), 2);
        assert!(hand.players.iter().all(|p| p.name != "B"));
    }

    #[test]
    fn parses_export_hands_sample_file() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("export_hands")
            .join("File31.txt");
        let path = if path.exists() {
            path
        } else {
            std::path::PathBuf::from(r"C:\Users\Никита\Desktop\export_hands\File31.txt")
        };
        if !path.exists() {
            return;
        }
        let text = std::fs::read_to_string(&path).expect("read sample");
        let hands = parse_hand_histories(&text);
        assert!(!hands.is_empty());
        assert!(hands.iter().all(|h| h.tournament_id == "4036159319"));
        assert!(hands.iter().any(|h| h.hero_combo_label.is_some()));
    }
}
