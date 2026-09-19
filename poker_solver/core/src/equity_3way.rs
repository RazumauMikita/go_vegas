use crate::card::Card;
use crate::hand_evaluator::{evaluate_hand, HandRank};
use crate::icm::{icm_equity, IcmCache};

const DECK_AFTER_SIX: usize = 46;

/// Chip-share игрока `hand1` в 3-way all-in: P(выигрыш) + 0.5·P(тай на двоих) + 1/3·P(тай на троих).
pub fn equity_3way(hand1: [Card; 2], hand2: [Card; 2], hand3: [Card; 2], iterations: u64) -> f64 {
    equity_3way_shares(hand1, hand2, hand3, iterations)[0]
}

/// Chip-share всех трёх игроков.
pub fn equity_3way_shares(
    hand1: [Card; 2],
    hand2: [Card; 2],
    hand3: [Card; 2],
    iterations: u64,
) -> [f64; 3] {
    if iterations == 0 {
        return [1.0 / 3.0; 3];
    }

    let dead = [hand1[0], hand1[1], hand2[0], hand2[1], hand3[0], hand3[1]];
    if has_duplicate(&dead) {
        return [1.0 / 3.0; 3];
    }

    let available = available_cards(dead);
    let mut rng = Rng::new(0xC3A5_9B17_F4D2_81E9 ^ hash_six(dead));
    let mut indices = [0_u8; DECK_AFTER_SIX];
    for (slot, index) in indices.iter_mut().enumerate() {
        *index = slot as u8;
    }

    let mut shares = [0.0; 3];
    for _ in 0..iterations {
        let board = sample_board(&available, &mut indices, &mut rng);
        let ranks = [
            evaluate_seven(hand1, board),
            evaluate_seven(hand2, board),
            evaluate_seven(hand3, board),
        ];
        let sample = chip_shares(ranks);
        shares[0] += sample[0];
        shares[1] += sample[1];
        shares[2] += sample[2];
    }

    let total = iterations as f64;
    [shares[0] / total, shares[1] / total, shares[2] / total]
}

/// $EV трёх игроков при равных стеках (места 1/2/3 по силе руки).
pub fn equity_3way_places(
    hand1: [Card; 2],
    hand2: [Card; 2],
    hand3: [Card; 2],
    payouts: &[f64],
    iterations: u64,
) -> [f64; 3] {
    if iterations == 0 {
        return split_all(payouts);
    }

    let dead = [hand1[0], hand1[1], hand2[0], hand2[1], hand3[0], hand3[1]];
    if has_duplicate(&dead) {
        return split_all(payouts);
    }

    let available = available_cards(dead);
    let mut rng = Rng::new(0x91D3_E70A_22BC_44F1 ^ hash_six(dead));
    let mut indices = [0_u8; DECK_AFTER_SIX];
    for (slot, index) in indices.iter_mut().enumerate() {
        *index = slot as u8;
    }

    let mut ev = [0.0; 3];
    for _ in 0..iterations {
        let board = sample_board(&available, &mut indices, &mut rng);
        let ranks = [
            evaluate_seven(hand1, board),
            evaluate_seven(hand2, board),
            evaluate_seven(hand3, board),
        ];
        let sample = place_payouts(ranks, payouts);
        ev[0] += sample[0];
        ev[1] += sample[1];
        ev[2] += sample[2];
    }

    let total = iterations as f64;
    [ev[0] / total, ev[1] / total, ev[2] / total]
}

/// Возвращает новые стеки после 3-way all-in showdown.
/// `stacks` — эффективные стеки (сколько каждый ставит в олл-ин).
/// Порядок: btn, sb, bb.
pub fn resolve_3way_showdown(stacks: [f64; 3], ranks: [HandRank; 3]) -> [f64; 3] {
    let mut levels = stacks;
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut unique = [0.0; 3];
    let mut n_unique = 0;
    for level in levels {
        if level <= 0.0 {
            continue;
        }
        if n_unique == 0 || unique[n_unique - 1] != level {
            unique[n_unique] = level;
            n_unique += 1;
        }
    }

    let mut won = [0.0; 3];
    let mut prev = 0.0;
    for level in unique.iter().copied().take(n_unique) {
        let mut n_in = 0_u32;
        let mut in_pot = [false; 3];
        for i in 0..3 {
            if stacks[i] >= level {
                in_pot[i] = true;
                n_in += 1;
            }
        }
        if n_in == 0 {
            prev = level;
            continue;
        }

        let pot = (level - prev) * f64::from(n_in);
        let mut best: Option<HandRank> = None;
        for i in 0..3 {
            if in_pot[i] {
                best = Some(match best {
                    None => ranks[i],
                    Some(current) => current.max(ranks[i]),
                });
            }
        }
        let best = best.expect("pot has a participant");
        let winners = in_pot
            .iter()
            .enumerate()
            .filter(|(i, is_in)| **is_in && ranks[*i] == best)
            .count() as f64;
        let share = pot / winners;
        for i in 0..3 {
            if in_pot[i] && ranks[i] == best {
                won[i] += share;
            }
        }
        prev = level;
    }
    won
}

/// Финальные стеки после 3-way all-in: side pots по contested-стекам + uncalled.
pub fn finalize_3way_stacks(
    contested: [f64; 3],
    uncalled: [f64; 3],
    ranks: [HandRank; 3],
) -> [f64; 3] {
    let mut stacks = resolve_3way_showdown(contested, ranks);
    for i in 0..3 {
        stacks[i] += uncalled[i];
    }
    stacks
}

/// $EV трёх игроков после 3-way all-in: side pots, затем ICM оставшихся стеков.
///
/// `contested` — эффективные олл-ин стеки (btn, sb, bb).
/// `uncalled` — фишки, возвращённые до расчёта ICM (например, overbet BTN).
pub fn equity_3way_icm(
    hand1: [Card; 2],
    hand2: [Card; 2],
    hand3: [Card; 2],
    contested: [f64; 3],
    uncalled: [f64; 3],
    payouts: &[f64],
    iterations: u64,
) -> [f64; 3] {
    equity_3way_icm_with_cache(
        hand1,
        hand2,
        hand3,
        contested,
        uncalled,
        payouts,
        iterations,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn equity_3way_icm_with_cache(
    hand1: [Card; 2],
    hand2: [Card; 2],
    hand3: [Card; 2],
    contested: [f64; 3],
    uncalled: [f64; 3],
    payouts: &[f64],
    iterations: u64,
    icm_cache: Option<&IcmCache>,
) -> [f64; 3] {
    if iterations == 0 {
        return split_all(payouts);
    }

    let dead = [hand1[0], hand1[1], hand2[0], hand2[1], hand3[0], hand3[1]];
    if has_duplicate(&dead) {
        return split_all(payouts);
    }

    let available = available_cards(dead);
    let mut rng = Rng::new(0x91D3_E70A_22BC_44F1 ^ hash_six(dead));
    let mut indices = [0_u8; DECK_AFTER_SIX];
    for (slot, index) in indices.iter_mut().enumerate() {
        *index = slot as u8;
    }

    let mut ev = [0.0; 3];
    for _ in 0..iterations {
        let board = sample_board(&available, &mut indices, &mut rng);
        let ranks = [
            evaluate_seven(hand1, board),
            evaluate_seven(hand2, board),
            evaluate_seven(hand3, board),
        ];
        let new_stacks = finalize_3way_stacks(contested, uncalled, ranks);
        let pre_showdown = [
            contested[0] + uncalled[0],
            contested[1] + uncalled[1],
            contested[2] + uncalled[2],
        ];
        let sample =
            icm_after_showdown_with_cache(new_stacks, pre_showdown, ranks, payouts, icm_cache);
        ev[0] += sample[0];
        ev[1] += sample[1];
        ev[2] += sample[2];
    }

    let total = iterations as f64;
    [ev[0] / total, ev[1] / total, ev[2] / total]
}

/// $EV после 3-way showdown (для диагностики).
pub fn icm_showdown_equity(
    new_stacks: [f64; 3],
    pre_showdown_stacks: [f64; 3],
    ranks: [HandRank; 3],
    payouts: &[f64],
) -> [f64; 3] {
    icm_after_showdown_with_cache(new_stacks, pre_showdown_stacks, ranks, payouts, None)
}

#[cfg(test)]
fn icm_after_showdown(
    new_stacks: [f64; 3],
    pre_showdown_stacks: [f64; 3],
    ranks: [HandRank; 3],
    payouts: &[f64],
) -> [f64; 3] {
    icm_after_showdown_with_cache(new_stacks, pre_showdown_stacks, ranks, payouts, None)
}

fn icm_lookup(stacks: &[f64], payouts: &[f64], cache: Option<&IcmCache>) -> Vec<f64> {
    match cache {
        Some(cache) => cache.lookup(stacks, payouts),
        None => icm_equity(stacks, payouts),
    }
}

fn icm_after_showdown_with_cache(
    new_stacks: [f64; 3],
    pre_showdown_stacks: [f64; 3],
    _ranks: [HandRank; 3],
    payouts: &[f64],
    icm_cache: Option<&IcmCache>,
) -> [f64; 3] {
    let alive: Vec<usize> = (0..3).filter(|&i| new_stacks[i] > 0.0).collect();
    let busted: Vec<usize> = (0..3).filter(|&i| new_stacks[i] <= 0.0).collect();
    let mut ev = [0.0; 3];

    if !busted.is_empty() {
        let first_place = alive.len();
        for &player in &busted {
            let better = busted
                .iter()
                .filter(|&&j| pre_showdown_stacks[j] > pre_showdown_stacks[player])
                .count();
            let tied = busted
                .iter()
                .filter(|&&j| pre_showdown_stacks[j] == pre_showdown_stacks[player])
                .count()
                .max(1);
            let mut prize = 0.0;
            for offset in 0..tied {
                prize += payouts
                    .get(first_place + better + offset)
                    .copied()
                    .unwrap_or(0.0);
            }
            ev[player] = prize / tied as f64;
        }
    }

    if alive.is_empty() {
        return ev;
    }

    if alive.len() == 3 {
        let values = icm_lookup(&new_stacks, payouts, icm_cache);
        return [values[0], values[1], values[2]];
    }

    if alive.len() == 1 {
        ev[alive[0]] = payouts.first().copied().unwrap_or(0.0);
        return ev;
    }

    let remaining: Vec<f64> = (0..alive.len())
        .map(|place| payouts.get(place).copied().unwrap_or(0.0))
        .collect();
    let remaining_sum: f64 = remaining.iter().sum();
    if remaining_sum <= 0.0 {
        return ev;
    }
    let alive_stacks: Vec<f64> = alive.iter().map(|&i| new_stacks[i]).collect();
    let normalized: Vec<f64> = remaining
        .iter()
        .map(|payout| payout / remaining_sum)
        .collect();
    let icm = icm_lookup(&alive_stacks, &normalized, icm_cache);
    for (offset, &index) in alive.iter().enumerate() {
        ev[index] = icm[offset] * remaining_sum;
    }
    ev
}

pub(crate) fn prefill_icm_cache(
    cache: &mut IcmCache,
    contested: [f64; 3],
    uncalled: [f64; 3],
    payouts: &[f64],
) {
    let hi = dummy_rank([
        "Ah", "Ad", "Ac", "As", "Kh", "2c", "3d",
    ]);
    let mid = dummy_rank([
        "Kh", "Kd", "Kc", "Qh", "Qd", "2s", "3s",
    ]);
    let lo = dummy_rank([
        "9h", "8d", "7c", "5s", "4h", "3c", "2d",
    ]);
    let patterns = [
        [hi, mid, lo],
        [hi, lo, mid],
        [mid, hi, lo],
        [mid, lo, hi],
        [lo, hi, mid],
        [lo, mid, hi],
        [hi, hi, lo],
        [hi, lo, hi],
        [lo, hi, hi],
        [hi, mid, mid],
        [mid, hi, mid],
        [mid, mid, hi],
        [hi, hi, hi],
    ];
    for ranks in patterns {
        cache.remember_showdown(finalize_3way_stacks(contested, uncalled, ranks), payouts);
    }
}

fn dummy_rank(cards: [&str; 7]) -> HandRank {
    use std::str::FromStr;
    let parsed: [Card; 7] = cards
        .iter()
        .map(|s| Card::from_str(s).expect("card"))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    evaluate_hand(&parsed)
}

fn split_all(payouts: &[f64]) -> [f64; 3] {
    let sum: f64 = (0..3).map(|i| payouts.get(i).copied().unwrap_or(0.0)).sum();
    [sum / 3.0, sum / 3.0, sum / 3.0]
}

fn chip_shares(ranks: [HandRank; 3]) -> [f64; 3] {
    let best = ranks[0].max(ranks[1]).max(ranks[2]);
    let winners = ranks.iter().filter(|rank| **rank == best).count() as f64;
    let mut shares = [0.0; 3];
    for (index, rank) in ranks.iter().enumerate() {
        if *rank == best {
            shares[index] = 1.0 / winners;
        }
    }
    shares
}

fn place_payouts(ranks: [HandRank; 3], payouts: &[f64]) -> [f64; 3] {
    let mut ev = [0.0; 3];
    for player in 0..3 {
        let better = ranks.iter().filter(|rank| **rank > ranks[player]).count();
        let tied = ranks.iter().filter(|rank| **rank == ranks[player]).count();
        let mut prize = 0.0;
        for offset in 0..tied {
            prize += payouts.get(better + offset).copied().unwrap_or(0.0);
        }
        ev[player] = prize / tied as f64;
    }
    ev
}

fn evaluate_seven(hole: [Card; 2], board: [Card; 5]) -> HandRank {
    evaluate_hand(&[
        hole[0], hole[1], board[0], board[1], board[2], board[3], board[4],
    ])
}

fn has_duplicate(cards: &[Card; 6]) -> bool {
    for i in 0..6 {
        for j in (i + 1)..6 {
            if cards[i] == cards[j] {
                return true;
            }
        }
    }
    false
}

fn available_cards(dead: [Card; 6]) -> [Card; DECK_AFTER_SIX] {
    let mut available = [Card::new(2, 0); DECK_AFTER_SIX];
    let mut slot = 0;
    for suit in 0..4 {
        for rank in Card::MIN_RANK..=Card::MAX_RANK {
            let card = Card::new(rank, suit);
            if dead.contains(&card) {
                continue;
            }
            available[slot] = card;
            slot += 1;
        }
    }
    available
}

fn sample_board(
    available: &[Card; DECK_AFTER_SIX],
    indices: &mut [u8; DECK_AFTER_SIX],
    rng: &mut Rng,
) -> [Card; 5] {
    for slot in 0..5 {
        let remaining = DECK_AFTER_SIX - slot;
        let pick = rng.gen_range(remaining) + slot;
        indices.swap(slot, pick);
    }
    [
        available[indices[0] as usize],
        available[indices[1] as usize],
        available[indices[2] as usize],
        available[indices[3] as usize],
        available[indices[4] as usize],
    ]
}

fn hash_six(cards: [Card; 6]) -> u64 {
    let mut hash = 0_u64;
    for card in cards {
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
        Self { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn gen_range(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_u64() as usize) % max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hand(a: &str, b: &str) -> [Card; 2] {
        [a.parse().expect("card"), b.parse().expect("card")]
    }

    #[test]
    fn aa_is_ahead_in_three_way() {
        let aa = hand("Ah", "Ad");
        let kk = hand("Kh", "Kd");
        let qq = hand("Qh", "Qd");
        let eq = equity_3way(aa, kk, qq, 4_000);
        assert!(eq > 0.45, "AA 3-way equity {eq}");
        assert!(eq < 0.75, "AA 3-way equity {eq}");
    }

    #[test]
    fn shares_sum_to_one() {
        let shares =
            equity_3way_shares(hand("Ah", "Kd"), hand("Qc", "Js"), hand("7h", "7d"), 3_000);
        let sum = shares[0] + shares[1] + shares[2];
        assert!((sum - 1.0).abs() < 1e-9, "sum {sum}");
    }

    fn rank(cards: [&str; 7]) -> HandRank {
        evaluate_hand(&cards.map(|card| card.parse().expect("card")))
    }

    fn btn_wins() -> HandRank {
        rank(["Ah", "Ad", "Ac", "As", "Kh", "2c", "3d"])
    }

    fn sb_mid() -> HandRank {
        rank(["Kh", "Kd", "Kc", "Qh", "Qd", "2s", "3s"])
    }

    fn bb_lose() -> HandRank {
        rank(["9h", "8d", "7c", "5s", "4h", "3c", "2d"])
    }

    fn bb_wins() -> HandRank {
        rank(["Ah", "Kh", "Qh", "Jh", "Th", "2c", "3d"])
    }

    fn btn_lose() -> HandRank {
        rank(["9h", "8d", "7c", "5s", "4h", "3c", "2d"])
    }

    fn assert_stacks(actual: [f64; 3], expected: [f64; 3]) {
        for i in 0..3 {
            assert!(
                (actual[i] - expected[i]).abs() < 1e-9,
                "stack[{i}]: expected {}, got {}",
                expected[i],
                actual[i]
            );
        }
    }

    #[test]
    fn finalize_effective_matches_full_overbet() {
        let ranks = [btn_wins(), sb_mid(), bb_lose()];
        let full = finalize_3way_stacks([2000.0, 500.0, 900.0], [0.0, 0.0, 0.0], ranks);
        let effective = finalize_3way_stacks([900.0, 500.0, 900.0], [1100.0, 0.0, 0.0], ranks);
        assert_stacks(full, effective);
    }

    #[test]
    fn resolve_all_equal_stacks() {
        let new_stacks =
            resolve_3way_showdown([1000.0, 1000.0, 1000.0], [btn_wins(), sb_mid(), bb_lose()]);
        assert_stacks(new_stacks, [3000.0, 0.0, 0.0]);
    }

    #[test]
    fn resolve_short_stack_wins() {
        let new_stacks =
            resolve_3way_showdown([500.0, 1000.0, 5000.0], [btn_wins(), sb_mid(), bb_lose()]);
        assert_stacks(new_stacks, [1500.0, 1000.0, 4000.0]);
    }

    #[test]
    fn resolve_bb_wins_all() {
        let new_stacks =
            resolve_3way_showdown([500.0, 1000.0, 5000.0], [btn_lose(), sb_mid(), bb_wins()]);
        assert_stacks(new_stacks, [0.0, 0.0, 6500.0]);
    }

    #[test]
    fn resolve_two_way_tie() {
        let tied = btn_wins();
        let new_stacks = resolve_3way_showdown([1000.0, 1000.0, 1000.0], [tied, tied, bb_lose()]);
        assert_stacks(new_stacks, [1500.0, 1500.0, 0.0]);
    }

    #[test]
    fn busted_tiebreak_by_pre_showdown_stack() {
        let payouts = [0.5, 0.3, 0.2];
        let pre = [2000.0, 500.0, 900.0];
        let ranks = [btn_wins(), sb_mid(), bb_lose()];
        let ev = icm_after_showdown([3400.0, 0.0, 0.0], pre, ranks, &payouts);
        assert!((ev[0] - 0.5).abs() < 1e-9, "BTN ICM {}", ev[0]);
        assert!((ev[1] - 0.2).abs() < 1e-9, "SB 3rd ICM {}", ev[1]);
        assert!((ev[2] - 0.3).abs() < 1e-9, "BB 2nd ICM {}", ev[2]);
    }

    #[test]
    fn bb_surviving_loss_keeps_high_icm() {
        let payouts = [0.5, 0.3, 0.2];
        let lose_ranks = [btn_wins(), sb_mid(), bb_lose()];
        let lose_stacks = resolve_3way_showdown([500.0, 1000.0, 5000.0], lose_ranks);
        let pre = [500.0, 1000.0, 5000.0];
        let lose_ev = icm_after_showdown(lose_stacks, pre, lose_ranks, &payouts);
        assert!(
            lose_ev[2] > 0.35,
            "BB leftover chips ICM {}, stacks {lose_stacks:?}",
            lose_ev[2]
        );

        let win_ranks = [btn_lose(), sb_mid(), bb_wins()];
        let win_stacks = resolve_3way_showdown([500.0, 1000.0, 5000.0], win_ranks);
        let win_ev = icm_after_showdown(win_stacks, pre, win_ranks, &payouts);
        assert!((win_ev[2] - 0.5).abs() < 1e-9, "BB win ICM {}", win_ev[2]);

        let aa = hand("Ah", "Ad");
        let kk = hand("Kh", "Kd");
        let qq = hand("Qh", "Qd");
        let as_bb = equity_3way_icm(
            aa,
            kk,
            qq,
            [1000.0, 500.0, 1000.0],
            [4000.0, 0.0, 0.0],
            &payouts,
            800,
        );
        assert!(as_bb[0] > 0.45, "AA as BB 3-way EV {}", as_bb[0]);
    }
}
