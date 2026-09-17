use crate::card::{Card, RANK_CHARS};
use crate::hand_evaluator::{evaluate_hand, HandRank};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EquityResult {
    pub win: f64,
    pub tie: f64,
    pub lose: f64,
}

impl EquityResult {
    pub fn equity(&self) -> f64 {
        self.win + self.tie / 2.0
    }
}

/// Диапазон рук одного игрока — список возможных 2-карточных комбинаций.
pub type HandRange = Vec<[Card; 2]>;

/// Точный расчёт через полный перебор board.
pub fn equity_exact(ranges: &[HandRange], board: &[Card]) -> Vec<EquityResult> {
    validate_board(board);

    let mut totals = vec![EquityResult::default(); ranges.len()];
    let mut scenarios = 0_u64;
    let mut assignment = vec![[Card::new(2, 0); 2]; ranges.len()];

    enumerate_assignments(ranges, board, 0, &mut assignment, &mut |hands| {
        let mut dead = Vec::with_capacity(board.len() + hands.len() * 2);
        dead.extend_from_slice(board);
        for hand in hands {
            dead.push(hand[0]);
            dead.push(hand[1]);
        }

        let available = available_cards(&dead);
        let missing = 5 - board.len();
        if available.len() < missing {
            return;
        }

        let mut full_board = [Card::new(2, 0); 5];
        full_board[..board.len()].copy_from_slice(board);

        enumerate_board_completions(
            &available,
            missing,
            0,
            &mut full_board,
            &mut |board_cards| {
                let mut ranks = Vec::with_capacity(hands.len());
                for hand in hands {
                    let mut cards = [Card::new(2, 0); 7];
                    cards[0] = hand[0];
                    cards[1] = hand[1];
                    cards[2..7].copy_from_slice(board_cards);
                    ranks.push(evaluate_hand(&cards));
                }

                accumulate_showdown(&ranks, &mut totals);
                scenarios += 1;
            },
        );
    });

    normalize_results(&mut totals, scenarios);
    totals
}

/// Monte Carlo симуляция.
pub fn equity_monte_carlo(
    ranges: &[HandRange],
    board: &[Card],
    iterations: u64,
) -> Vec<EquityResult> {
    validate_board(board);

    if iterations == 0 || ranges.is_empty() {
        return vec![EquityResult::default(); ranges.len()];
    }

    let mut totals = vec![EquityResult::default(); ranges.len()];
    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
    let mut successful = 0_u64;

    for _ in 0..iterations {
        let Some(hands) = sample_assignment(ranges, board, &mut rng) else {
            continue;
        };

        let mut dead = Vec::with_capacity(board.len() + hands.len() * 2);
        dead.extend_from_slice(board);
        for hand in &hands {
            dead.push(hand[0]);
            dead.push(hand[1]);
        }

        let available = available_cards(&dead);
        let missing = 5 - board.len();
        if available.len() < missing {
            continue;
        }

        let mut full_board = [Card::new(2, 0); 5];
        full_board[..board.len()].copy_from_slice(board);
        sample_partial_shuffle(
            &available,
            missing,
            &mut rng,
            &mut full_board[board.len()..],
        );

        let mut ranks = Vec::with_capacity(hands.len());
        for hand in &hands {
            let mut cards = [Card::new(2, 0); 7];
            cards[0] = hand[0];
            cards[1] = hand[1];
            cards[2..7].copy_from_slice(&full_board);
            ranks.push(evaluate_hand(&cards));
        }

        accumulate_showdown(&ranks, &mut totals);
        successful += 1;
    }

    normalize_results(&mut totals, successful);
    totals
}

/// Парсинг диапазона из строки формата "AA,KK,AKs,AQo+,77+,A2s-A5s".
pub fn parse_range(s: &str) -> HandRange {
    let mut combos = HandRange::new();

    for token in s.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }

        if let Some((left, right)) = token.split_once('-') {
            expand_notation_range(left.trim(), right.trim(), &mut combos);
        } else {
            expand_notation(token, false, &mut combos);
        }
    }

    dedup_hands(&mut combos);
    combos
}

fn validate_board(board: &[Card]) {
    assert!(board.len() <= 5, "board cannot contain more than 5 cards");
    assert!(!has_duplicates(board), "board contains duplicate cards");
}

fn has_duplicates(cards: &[Card]) -> bool {
    for i in 0..cards.len() {
        for j in (i + 1)..cards.len() {
            if cards[i] == cards[j] {
                return true;
            }
        }
    }
    false
}

fn available_cards(dead: &[Card]) -> Vec<Card> {
    let mut deck = Vec::with_capacity(52);
    'outer: for suit in 0..4 {
        for rank in Card::MIN_RANK..=Card::MAX_RANK {
            let card = Card::new(rank, suit);
            for dead_card in dead {
                if card == *dead_card {
                    continue 'outer;
                }
            }
            deck.push(card);
        }
    }
    deck
}

fn enumerate_assignments<F>(
    ranges: &[HandRange],
    board: &[Card],
    player_index: usize,
    assignment: &mut [[Card; 2]],
    callback: &mut F,
) where
    F: FnMut(&[[Card; 2]]),
{
    if player_index == ranges.len() {
        callback(assignment);
        return;
    }

    let mut dead = Vec::with_capacity(board.len() + player_index * 2);
    dead.extend_from_slice(board);
    for hand in &assignment[..player_index] {
        dead.push(hand[0]);
        dead.push(hand[1]);
    }

    for hand in &ranges[player_index] {
        if dead.contains(&hand[0]) || dead.contains(&hand[1]) {
            continue;
        }

        assignment[player_index] = *hand;
        enumerate_assignments(ranges, board, player_index + 1, assignment, callback);
    }
}

fn enumerate_board_completions<F>(
    available: &[Card],
    missing: usize,
    start: usize,
    board: &mut [Card; 5],
    callback: &mut F,
) where
    F: FnMut(&[Card; 5]),
{
    if missing == 0 {
        callback(board);
        return;
    }

    if missing > available.len() {
        return;
    }

    for index in start..=available.len() - missing {
        board[5 - missing] = available[index];
        enumerate_board_completions(available, missing - 1, index + 1, board, callback);
    }
}

fn accumulate_showdown(ranks: &[HandRank], totals: &mut [EquityResult]) {
    let best = ranks.iter().copied().max().expect("at least one player");
    let winners = ranks.iter().filter(|rank| **rank == best).count();

    for (index, rank) in ranks.iter().enumerate() {
        if *rank == best {
            if winners == 1 {
                totals[index].win += 1.0;
            } else {
                totals[index].tie += 2.0 / winners as f64;
            }
        } else {
            totals[index].lose += 1.0;
        }
    }
}

fn normalize_results(results: &mut [EquityResult], scenarios: u64) {
    if scenarios == 0 {
        return;
    }

    let total = scenarios as f64;
    for result in results {
        result.win /= total;
        result.tie /= total;
        result.lose /= total;
    }
}

fn sample_assignment(
    ranges: &[HandRange],
    board: &[Card],
    rng: &mut Rng,
) -> Option<Vec<[Card; 2]>> {
    let mut hands = Vec::with_capacity(ranges.len());
    let mut dead = board.to_vec();

    for range in ranges {
        if range.is_empty() {
            return None;
        }

        let mut candidates = Vec::new();
        for hand in range {
            if !dead.contains(&hand[0]) && !dead.contains(&hand[1]) {
                candidates.push(*hand);
            }
        }

        if candidates.is_empty() {
            return None;
        }

        let hand = candidates[rng.gen_range(candidates.len())];
        dead.push(hand[0]);
        dead.push(hand[1]);
        hands.push(hand);
    }

    Some(hands)
}

fn sample_partial_shuffle(available: &[Card], count: usize, rng: &mut Rng, out: &mut [Card]) {
    let mut indices: Vec<usize> = (0..available.len()).collect();
    for (slot, out_card) in out.iter_mut().enumerate().take(count) {
        let remaining = indices.len() - slot;
        let pick = rng.gen_range(remaining);
        let chosen = indices.len() - slot - 1;
        indices.swap(pick, chosen);
        *out_card = available[indices[chosen]];
    }
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
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

fn expand_notation_range(left: &str, right: &str, out: &mut HandRange) {
    let (left_high, left_low, left_suited) = parse_hand_notation(left);
    let (right_high, right_low, right_suited) = parse_hand_notation(right);

    if left_high != right_high || left_suited != right_suited {
        return;
    }

    if left_low.is_none() && right_low.is_none() {
        if let (Some(start), Some(end)) = (left_high, right_high) {
            for rank in start.min(end)..=start.max(end) {
                expand_pair(rank, out);
            }
        }
        return;
    }

    if left_low == right_low {
        if let Some(rank) = left_high {
            expand_pair(rank, out);
        }
        return;
    }

    if let (Some(left), Some(right)) = (left_low, right_low) {
        let start = left.min(right);
        let end = left.max(right);

        for rank in start..=end {
            if Some(rank) == left_high {
                expand_pair(rank, out);
            } else if left_suited == Some(true) {
                expand_suited(left_high.unwrap(), rank, out);
            } else if left_suited == Some(false) {
                expand_offsuit(left_high.unwrap(), rank, out);
            }
        }
    }
}

fn expand_notation(token: &str, allow_plus: bool, out: &mut HandRange) {
    let plus = allow_plus || token.ends_with('+');
    let token = token.trim_end_matches('+');
    let (high, low, suited) = parse_hand_notation(token);

    if low.is_none() {
        if let Some(rank) = high {
            if plus {
                for pair_rank in rank..=Card::MAX_RANK {
                    expand_pair(pair_rank, out);
                }
            } else {
                expand_pair(rank, out);
            }
        }
        return;
    }

    let high_rank = high.unwrap();
    let low_rank = low.unwrap();

    if plus {
        if suited == Some(true) {
            for rank in low_rank..high_rank {
                expand_suited(high_rank, rank, out);
            }
        } else if suited == Some(false) {
            for rank in low_rank..high_rank {
                expand_offsuit(high_rank, rank, out);
            }
        }
        return;
    }

    match suited {
        Some(true) => expand_suited(high_rank, low_rank, out),
        Some(false) => expand_offsuit(high_rank, low_rank, out),
        None => expand_pair(high_rank, out),
    }
}

fn parse_hand_notation(token: &str) -> (Option<u8>, Option<u8>, Option<bool>) {
    let chars: Vec<char> = token.chars().collect();
    if chars.is_empty() {
        return (None, None, None);
    }

    if chars.len() == 2 {
        let first = parse_rank_char(chars[0]);
        let second = parse_rank_char(chars[1]);
        if first.is_some() && first == second {
            return (first, None, None);
        }
    }

    if chars.len() == 3 {
        let first = parse_rank_char(chars[0]);
        let second = parse_rank_char(chars[1]);
        let suffix = chars[2].to_ascii_lowercase();
        let suited = match suffix {
            's' => Some(true),
            'o' => Some(false),
            _ => None,
        };

        if let (Some(mut high), Some(mut low)) = (first, second) {
            if high < low {
                std::mem::swap(&mut high, &mut low);
            }
            return (Some(high), Some(low), suited);
        }
    }

    (None, None, None)
}

fn parse_rank_char(value: char) -> Option<u8> {
    RANK_CHARS
        .chars()
        .position(|rank| rank == value || rank == value.to_ascii_uppercase())
        .map(|index| index as u8 + Card::MIN_RANK)
}

fn expand_pair(rank: u8, out: &mut HandRange) {
    for suit_a in 0..4 {
        for suit_b in (suit_a + 1)..4 {
            out.push(normalize_hand([
                Card::new(rank, suit_a),
                Card::new(rank, suit_b),
            ]));
        }
    }
}

fn expand_suited(high: u8, low: u8, out: &mut HandRange) {
    for suit in 0..4 {
        out.push(normalize_hand([
            Card::new(high, suit),
            Card::new(low, suit),
        ]));
    }
}

fn expand_offsuit(high: u8, low: u8, out: &mut HandRange) {
    for suit_a in 0..4 {
        for suit_b in 0..4 {
            if suit_a == suit_b {
                continue;
            }
            out.push(normalize_hand([
                Card::new(high, suit_a),
                Card::new(low, suit_b),
            ]));
        }
    }
}

fn normalize_hand(hand: [Card; 2]) -> [Card; 2] {
    if hand[0].rank() > hand[1].rank()
        || (hand[0].rank() == hand[1].rank() && hand[0].suit() > hand[1].suit())
    {
        hand
    } else {
        [hand[1], hand[0]]
    }
}

fn dedup_hands(hands: &mut HandRange) {
    hands.sort_unstable();
    hands.dedup();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn card(s: &str) -> Card {
        Card::from_str(s).unwrap()
    }

    fn specific(hole: [&str; 2]) -> HandRange {
        vec![normalize_hand([card(hole[0]), card(hole[1])])]
    }

    #[test]
    fn parse_range_counts() {
        assert_eq!(parse_range("AA").len(), 6);
        assert_eq!(parse_range("AKs").len(), 4);
        assert_eq!(parse_range("AKo").len(), 12);
        assert_eq!(parse_range("77+").len(), 48);
        assert_eq!(parse_range("A2s-A5s").len(), 16);
    }

    #[test]
    #[ignore = "slow in debug, run with --ignored"]
    fn equity_sum_invariant_three_way() {
        let ranges = [parse_range("AA"), parse_range("KK"), parse_range("QQ")];
        let results = equity_exact(&ranges, &[]);
        let sum: f64 = results.iter().map(|r| r.equity()).sum();
        assert!((sum - 1.0).abs() < 0.0001);
    }

    #[test]
    #[ignore = "slow in debug, run with --ignored"]
    fn aks_vs_22_preflop_equity() {
        let ranges = [specific(["Ah", "Kh"]), specific(["2c", "2d"])];
        let exact = equity_exact(&ranges, &[]);
        assert!((exact[0].equity() - 0.50).abs() < 0.03);
    }

    #[test]
    fn parse_range_handles_common_tokens() {
        let range = parse_range("AA,KK,AKs,AQo+,77+,A2s-A5s");

        assert!(range.contains(&normalize_hand([card("Ah"), card("Ad")])));
        assert!(range.contains(&normalize_hand([card("Kh"), card("Kd")])));
        assert!(range.contains(&normalize_hand([card("Ah"), card("Kh")])));
        assert!(range.contains(&normalize_hand([card("Ah"), card("Qd")])));
        assert!(range.contains(&normalize_hand([card("Ah"), card("Kd")])));
        assert!(range.contains(&normalize_hand([card("7h"), card("7d")])));
        assert!(range.contains(&normalize_hand([card("Ah"), card("2h")])));
        assert!(range.contains(&normalize_hand([card("Ah"), card("5h")])));
        assert!(!range.contains(&normalize_hand([card("Ah"), card("6h")])));
    }

    #[test]
    #[ignore = "slow in debug, run with --ignored"]
    fn aa_vs_kk_preflop_equity_is_expected() {
        let ranges = [specific(["Ah", "Ad"]), specific(["Kh", "Kd"])];
        let exact = equity_exact(&ranges, &[]);
        let mc = equity_monte_carlo(&ranges, &[], 50_000);

        assert!((exact[0].equity() - 0.819).abs() < 0.01);
        assert!((exact[1].equity() - 0.181).abs() < 0.01);
        assert!((mc[0].equity() - exact[0].equity()).abs() < 0.02);
    }

    #[test]
    fn exact_and_monte_carlo_split_on_board_tie() {
        let ranges = [specific(["Ah", "Kd"]), specific(["Ac", "Kh"])];
        let board = [card("2s"), card("3s"), card("4s"), card("5s"), card("6s")];

        let exact = equity_exact(&ranges, &board);
        assert!((exact[0].win - 0.0).abs() < f64::EPSILON);
        assert!((exact[0].tie - 1.0).abs() < f64::EPSILON);
        assert!((exact[0].equity() - 0.5).abs() < f64::EPSILON);

        let mc = equity_monte_carlo(&ranges, &board, 10_000);
        assert!((mc[0].equity() - 0.5).abs() < 0.02);
    }

    #[test]
    fn three_way_tie_equity_uses_share_formula() {
        let ranges = [
            specific(["Ah", "Kd"]),
            specific(["Ac", "Qc"]),
            specific(["Ad", "Jd"]),
        ];
        let board = [card("2s"), card("3s"), card("4s"), card("5s"), card("6s")];

        let exact = equity_exact(&ranges, &board);
        for result in exact {
            assert!((result.equity() - 1.0 / 3.0).abs() < 0.001);
        }
    }
}
