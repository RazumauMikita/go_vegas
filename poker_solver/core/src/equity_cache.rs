//! Кэш префлоп-эквити для всех 169×169 типов рук.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use crate::card::Card;
use crate::hand_evaluator::evaluate_hand;

const COMBO_COUNT: usize = 169;
const TABLE_LEN: usize = COMBO_COUNT * COMBO_COUNT;
const CACHE_FILE_BYTES: usize = TABLE_LEN * 4;

/// Таблица префлоп-эквити hero против villain для всех 169×169 типов рук.
#[derive(Debug, Clone, PartialEq)]
pub struct EquityCache {
    values: Vec<f32>,
}

impl EquityCache {
    /// Сгенерировать кэш через Monte Carlo.
    ///
    /// `iterations` — число симуляций на каждую валидную пару конкретных комбинаций.
    pub fn generate(iterations: u64) -> Self {
        Self::generate_with_progress(iterations, |_, _| {})
    }

    /// Сгенерировать кэш и вызывать `progress(completed, started)` по мере готовности пар.
    pub fn generate_with_progress<F>(iterations: u64, progress: F) -> Self
    where
        F: Fn(u64, Instant) + Sync,
    {
        let completed = AtomicU64::new(0);
        let started = Instant::now();
        let report_every = (TABLE_LEN / 20).max(1) as u64;

        let values: Vec<f32> = (0..TABLE_LEN)
            .into_par_iter()
            .map(|flat| {
                let h1 = (flat / COMBO_COUNT) as u8;
                let h2 = (flat % COMBO_COUNT) as u8;
                let equity = pair_equity(h1, h2, iterations) as f32;

                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if done == TABLE_LEN as u64 || done.is_multiple_of(report_every) {
                    progress(done, started);
                }

                equity
            })
            .collect();

        Self { values }
    }

    /// Сгенерировать кэш только для указанных пар индексов.
    ///
    /// Используется в тестах, чтобы не считать весь 169×169.
    /// Незаполненные ячейки остаются равными `0.0`.
    pub fn generate_subset(pairs: &[(u8, u8)], iterations: u64) -> Self {
        let mut values = vec![0.0f32; TABLE_LEN];

        for &(h1, h2) in pairs {
            values[table_index(h1, h2)] = pair_equity(h1, h2, iterations) as f32;
        }

        Self { values }
    }

    /// Сохранить таблицу в бинарный файл little-endian без заголовка.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut file = File::create(path)?;
        for value in &self.values {
            file.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    }

    /// Загрузить таблицу из бинарного файла фиксированного размера.
    pub fn load(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.len() != CACHE_FILE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid cache size: expected {CACHE_FILE_BYTES} bytes, got {}",
                    bytes.len()
                ),
            ));
        }

        let mut values = Vec::with_capacity(TABLE_LEN);
        for chunk in bytes.as_chunks::<4>().0 {
            let value = f32::from_le_bytes(*chunk);
            values.push(value);
        }

        Ok(Self { values })
    }

    /// Эквити руки `h1` против руки `h2` (индексы 0..169).
    pub fn equity(&self, h1: u8, h2: u8) -> f64 {
        self.values[table_index(h1, h2)] as f64
    }

    /// Эквити конкретной пары карт против диапазона 169-индексов.
    ///
    /// Руки диапазона, пересекающиеся с `h1` по картам, исключаются из среднего.
    pub fn equity_vs_range(&self, h1: [Card; 2], range: &[u8]) -> f64 {
        let hero_index = combo_index(h1);
        let mut weighted_equity = 0.0;
        let mut total_weight = 0.0;

        for &range_index in range {
            let weight = expand_combo(range_index)
                .iter()
                .filter(|combo| !hands_overlap(h1, **combo))
                .count() as f64;

            if weight == 0.0 {
                continue;
            }

            weighted_equity += weight * self.equity(hero_index, range_index);
            total_weight += weight;
        }

        if total_weight == 0.0 {
            return 0.0;
        }

        weighted_equity / total_weight
    }
}

/// Преобразовать пару карт в индекс 0..168 стандартной матрицы 13×13.
pub fn combo_index(cards: [Card; 2]) -> u8 {
    let (high_rank, low_rank, suited) = normalized_ranks(cards);
    let row_high = rank_to_row(high_rank);
    let row_low = rank_to_row(low_rank);

    if high_rank == low_rank {
        return row_high;
    }

    if suited {
        suited_index(row_high, row_low)
    } else {
        offsuit_index(row_low, row_high)
    }
}

/// Обратный маппинг: `(rank_high, rank_low, is_suited)`.
pub fn index_to_ranks(idx: u8) -> (u8, u8, bool) {
    if idx < 13 {
        let rank = row_to_rank(idx);
        return (rank, rank, false);
    }

    if idx < 91 {
        let position = idx - 13;
        let (row, col) = suited_row_col(position);
        return (row_to_rank(row), row_to_rank(col), true);
    }

    let position = idx - 91;
    let (row, col) = offsuit_row_col(position);
    (row_to_rank(col), row_to_rank(row), false)
}

/// Развернуть 169-индекс в конкретные двухкарточные комбинации.
pub fn expand_combo(idx: u8) -> Vec<[Card; 2]> {
    let (high_rank, low_rank, suited) = index_to_ranks(idx);
    let mut combos = Vec::new();

    if high_rank == low_rank {
        expand_pair(high_rank, &mut combos);
    } else if suited {
        expand_suited(high_rank, low_rank, &mut combos);
    } else {
        expand_offsuit(high_rank, low_rank, &mut combos);
    }

    combos
}

fn table_index(h1: u8, h2: u8) -> usize {
    h1 as usize * COMBO_COUNT + h2 as usize
}

fn rank_to_row(rank: u8) -> u8 {
    Card::MAX_RANK - rank
}

fn row_to_rank(row: u8) -> u8 {
    Card::MAX_RANK - row
}

fn suited_index(row: u8, col: u8) -> u8 {
    debug_assert!(row < col);
    13 + suited_position(row, col)
}

fn offsuit_index(row: u8, col: u8) -> u8 {
    debug_assert!(row > col);
    91 + offsuit_position(row, col)
}

fn suited_position(row: u8, col: u8) -> u8 {
    let mut position = 0_u8;
    for current_row in 0..13 {
        for current_col in (current_row + 1)..13 {
            if current_row == row && current_col == col {
                return position;
            }
            position += 1;
        }
    }
    0
}

fn offsuit_position(row: u8, col: u8) -> u8 {
    let mut position = 0_u8;
    for current_row in 0..13 {
        for current_col in 0..current_row {
            if current_row == row && current_col == col {
                return position;
            }
            position += 1;
        }
    }
    0
}

fn suited_row_col(position: u8) -> (u8, u8) {
    let mut current = 0_u8;
    for row in 0..13 {
        for col in (row + 1)..13 {
            if current == position {
                return (row, col);
            }
            current += 1;
        }
    }
    (0, 1)
}

fn offsuit_row_col(position: u8) -> (u8, u8) {
    let mut current = 0_u8;
    for row in 0..13 {
        for col in 0..row {
            if current == position {
                return (row, col);
            }
            current += 1;
        }
    }
    (1, 0)
}

fn normalized_ranks(cards: [Card; 2]) -> (u8, u8, bool) {
    let mut first = cards[0];
    let mut second = cards[1];
    if first.rank() < second.rank() {
        std::mem::swap(&mut first, &mut second);
    }

    let suited = first.suit() == second.suit();
    (first.rank(), second.rank(), suited)
}

fn expand_pair(rank: u8, out: &mut Vec<[Card; 2]>) {
    for suit_a in 0..4 {
        for suit_b in (suit_a + 1)..4 {
            out.push(order_hand([
                Card::new(rank, suit_a),
                Card::new(rank, suit_b),
            ]));
        }
    }
}

fn expand_suited(high: u8, low: u8, out: &mut Vec<[Card; 2]>) {
    for suit in 0..4 {
        out.push(order_hand([Card::new(high, suit), Card::new(low, suit)]));
    }
}

fn expand_offsuit(high: u8, low: u8, out: &mut Vec<[Card; 2]>) {
    for suit_a in 0..4 {
        for suit_b in 0..4 {
            if suit_a == suit_b {
                continue;
            }
            out.push(order_hand([
                Card::new(high, suit_a),
                Card::new(low, suit_b),
            ]));
        }
    }
}

fn order_hand(hand: [Card; 2]) -> [Card; 2] {
    if hand[0].rank() > hand[1].rank()
        || (hand[0].rank() == hand[1].rank() && hand[0].suit() > hand[1].suit())
    {
        hand
    } else {
        [hand[1], hand[0]]
    }
}

fn hands_overlap(h1: [Card; 2], h2: [Card; 2]) -> bool {
    h1[0] == h2[0] || h1[0] == h2[1] || h1[1] == h2[0] || h1[1] == h2[1]
}

fn pair_equity(h1: u8, h2: u8, iterations: u64) -> f64 {
    let combos_h1 = expand_combo(h1);
    let combos_h2 = expand_combo(h2);
    let mut matchups = Vec::new();

    for combo_h1 in &combos_h1 {
        for combo_h2 in &combos_h2 {
            if hands_overlap(*combo_h1, *combo_h2) {
                continue;
            }
            matchups.push((*combo_h1, *combo_h2));
        }
    }

    if matchups.is_empty() {
        return 0.5;
    }

    let total: f64 = matchups
        .par_iter()
        .map(|(combo_h1, combo_h2)| monte_carlo_equity(*combo_h1, *combo_h2, iterations))
        .sum();

    total / matchups.len() as f64
}

fn monte_carlo_equity(hero: [Card; 2], villain: [Card; 2], iterations: u64) -> f64 {
    if iterations == 0 {
        return 0.5;
    }

    let dead = [hero[0], hero[1], villain[0], villain[1]];
    let available = available_cards(&dead);
    let mut rng = Rng::new(0xD1B5_4A32_D192_ED03 ^ hash_hand(hero, villain));
    let mut win = 0.0;
    let mut tie = 0.0;

    for _ in 0..iterations {
        let board = sample_board(&available, &mut rng);
        let hero_rank = evaluate_seven(hero, board);
        let villain_rank = evaluate_seven(villain, board);

        if hero_rank > villain_rank {
            win += 1.0;
        } else if hero_rank == villain_rank {
            tie += 1.0;
        }
    }

    let total = iterations as f64;
    win / total + tie / (2.0 * total)
}

fn evaluate_seven(hole: [Card; 2], board: [Card; 5]) -> crate::hand_evaluator::HandRank {
    let cards = [
        hole[0], hole[1], board[0], board[1], board[2], board[3], board[4],
    ];
    evaluate_hand(&cards)
}

fn available_cards(dead: &[Card]) -> Vec<Card> {
    let mut deck = Vec::with_capacity(48);
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

fn sample_board(available: &[Card], rng: &mut Rng) -> [Card; 5] {
    let mut indices: Vec<usize> = (0..available.len()).collect();
    let mut board = [Card::new(2, 0); 5];

    for (slot, board_card) in board.iter_mut().enumerate() {
        let remaining = indices.len() - slot;
        let pick = rng.gen_range(remaining);
        let chosen = indices.len() - slot - 1;
        indices.swap(pick, chosen);
        *board_card = available[indices[chosen]];
    }

    board
}

fn hash_hand(hero: [Card; 2], villain: [Card; 2]) -> u64 {
    let mut hash = 0_u64;
    for card in [hero[0], hero[1], villain[0], villain[1]] {
        hash = hash.wrapping_mul(53).wrapping_add(card.rank() as u64);
        hash = hash.wrapping_mul(53).wrapping_add(card.suit() as u64);
    }
    hash
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn card(text: &str) -> Card {
        Card::from_str(text).expect("valid test card")
    }

    #[test]
    fn combo_index_roundtrip() {
        for idx in 0..169_u8 {
            for hand in expand_combo(idx) {
                let roundtrip = combo_index(hand);
                assert_eq!(
                    roundtrip, idx,
                    "hand {:?} mapped to {roundtrip}, expected {idx}",
                    hand
                );
            }
        }
    }

    #[test]
    fn expand_combo_counts() {
        assert_eq!(expand_combo(0).len(), 6);
        assert_eq!(expand_combo(13).len(), 4);
        assert_eq!(expand_combo(91).len(), 12);
        assert_eq!(expand_combo(168).len(), 12);
    }

    #[test]
    fn cache_aa_vs_kk() {
        let cache = EquityCache::generate_subset(&[(0, 1)], 10_000);
        let aa_vs_kk = cache.equity(0, 1);
        assert!(
            (aa_vs_kk - 0.81).abs() < 0.05,
            "expected ~0.81, got {aa_vs_kk}"
        );
    }

    #[test]
    fn equity_vs_range_single() {
        let hero = [card("Ah"), card("Kh")];
        let villain = [card("Ac"), card("Kd")];
        let hero_index = combo_index(hero);
        let villain_index = combo_index(villain);
        let cache = EquityCache::generate_subset(&[(hero_index, villain_index)], 10_000);
        let equity = cache.equity_vs_range(hero, &[villain_index]);
        assert!((equity - 0.5).abs() < 0.05, "expected ~0.5, got {equity}");
    }

    #[test]
    fn save_load_roundtrip() {
        let cache = EquityCache::generate_subset(&[(0, 0), (0, 1)], 100);
        let path = std::env::temp_dir().join("poker_equity_cache_roundtrip.bin");
        cache.save(&path).expect("save cache");
        let loaded = EquityCache::load(&path).expect("load cache");
        assert_eq!(loaded.equity(0, 0), cache.equity(0, 0));
        assert_eq!(loaded.equity(0, 1), cache.equity(0, 1));
        let _ = fs::remove_file(path);
    }

    #[test]
    #[ignore = "full 169x169 generation, slow"]
    fn full_cache_generation() {
        let _cache = EquityCache::generate(100);
    }

    #[test]
    fn load_wrong_size_fails() {
        let path = std::env::temp_dir().join("poker_equity_cache_wrong_size.bin");
        fs::write(&path, vec![0_u8; 16]).expect("write invalid cache");
        let result = EquityCache::load(&path);
        assert!(result.is_err());
        let _ = fs::remove_file(path);
    }
}
