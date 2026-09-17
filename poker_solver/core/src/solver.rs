use std::collections::HashMap;

use crate::card::Card;
use crate::equity_cache::{expand_combo, hands_overlap, EquityCache};
use crate::icm::icm_equity;

const HAND_TYPES: usize = 169;
const INDIFFERENT_EPS: f64 = 1e-9;

/// Входные данные для солвера.
#[derive(Debug, Clone)]
pub struct SolverInput {
    /// Стеки всех игроков в фишках (в порядке позиций).
    pub stacks: Vec<f64>,
    /// Выплаты (например [0.5, 0.3, 0.2]).
    pub payouts: Vec<f64>,
    /// Малый блайнд.
    pub small_blind: f64,
    /// Большой блайнд.
    pub big_blind: f64,
    /// Анте (0 если нет).
    pub ante: f64,
    /// Индекс игрока на баттоне (0..N-1).
    pub button_index: usize,
    /// Количество итераций поиска равновесия.
    pub max_iterations: usize,
    /// Критерий сходимости (изменение диапазонов < tolerance).
    pub tolerance: f64,
}

/// Выходные данные солвера.
#[derive(Debug, Clone)]
pub struct SolverOutput {
    /// Для каждой позиции: диапазон пуша (169 частот, 0.0..1.0).
    pub push_ranges: Vec<[f64; HAND_TYPES]>,
    /// Для каждой позиции: диапазон колла (169 частот).
    pub call_ranges: Vec<[f64; HAND_TYPES]>,
    /// $EV каждого игрока при равновесии.
    pub equities: Vec<f64>,
    /// Сколько итераций потребовалось.
    pub iterations_used: usize,
    /// Достигнута ли сходимость.
    pub converged: bool,
}

/// Найти равновесие Нэша для push/fold.
///
/// Сейчас реализован только HU: SB (баттон) пушит или фолдит, BB коллит или фолдит.
pub fn solve(input: &SolverInput, cache: &EquityCache) -> SolverOutput {
    if input.stacks.len() != 2 {
        return empty_output(input.stacks.len());
    }
    if input.button_index >= input.stacks.len() {
        return empty_output(input.stacks.len());
    }
    if input.max_iterations == 0 {
        return empty_output(input.stacks.len());
    }

    let ctx = match HuContext::new(input) {
        Some(ctx) => ctx,
        None => return empty_output(input.stacks.len()),
    };

    let mut push_range = [1.0; HAND_TYPES];
    let mut call_range = [1.0; HAND_TYPES];
    let mut push_sum = [0.0; HAND_TYPES];
    let mut call_sum = [0.0; HAND_TYPES];
    let mut converged = false;
    let mut iterations_used = 0;

    for iteration in 1..=input.max_iterations {
        iterations_used = iteration;

        let mut br_push = [0.0; HAND_TYPES];
        let mut br_call = [0.0; HAND_TYPES];

        for hand_idx in 0..HAND_TYPES {
            let ev_push = ev_push_sb(hand_idx, &call_range, &ctx, cache);
            br_push[hand_idx] = best_response(ev_push, ctx.ev_sb_fold, push_range[hand_idx]);
        }

        for hand_idx in 0..HAND_TYPES {
            let ev_call = ev_call_bb(hand_idx, &push_range, &ctx, cache);
            br_call[hand_idx] = best_response(ev_call, ctx.ev_bb_fold, call_range[hand_idx]);
        }

        let mut change = 0.0;
        let t = iteration as f64;
        for hand_idx in 0..HAND_TYPES {
            push_sum[hand_idx] += br_push[hand_idx];
            call_sum[hand_idx] += br_call[hand_idx];
            let next_push = push_sum[hand_idx] / t;
            let next_call = call_sum[hand_idx] / t;
            change += (next_push - push_range[hand_idx]).abs();
            change += (next_call - call_range[hand_idx]).abs();
            push_range[hand_idx] = next_push;
            call_range[hand_idx] = next_call;
        }

        let mean_change = change / (2.0 * HAND_TYPES as f64);
        if change < input.tolerance || mean_change < input.tolerance {
            converged = true;
            break;
        }
    }

    let mut push_ranges = vec![[0.0; HAND_TYPES]; 2];
    let mut call_ranges = vec![[0.0; HAND_TYPES]; 2];
    push_ranges[ctx.sb] = push_range;
    call_ranges[ctx.bb] = call_range;

    let sb_equity = average_sb_equity(&push_range, &call_range, &ctx, cache);
    let mut equities = vec![0.0; 2];
    equities[ctx.sb] = sb_equity;
    equities[ctx.bb] = 1.0 - sb_equity;

    SolverOutput {
        push_ranges,
        call_ranges,
        equities,
        iterations_used,
        converged,
    }
}

fn empty_output(players: usize) -> SolverOutput {
    SolverOutput {
        push_ranges: vec![[0.0; HAND_TYPES]; players],
        call_ranges: vec![[0.0; HAND_TYPES]; players],
        equities: vec![0.0; players],
        iterations_used: 0,
        converged: false,
    }
}

struct HuContext {
    sb: usize,
    bb: usize,
    payouts: Vec<f64>,
    combos: Vec<Vec<[Card; 2]>>,
    unblocked: Vec<Vec<[u8; HAND_TYPES]>>,
    ev_sb_fold: f64,
    ev_sb_bb_folds: f64,
    ev_bb_fold: f64,
    ev_sb_showdown_win: f64,
    ev_sb_showdown_lose: f64,
    ev_bb_showdown_win: f64,
    ev_bb_showdown_lose: f64,
    icm_cache: HashMap<[u64; 2], Vec<f64>>,
}

impl HuContext {
    fn new(input: &SolverInput) -> Option<Self> {
        if input.stacks.len() != 2 {
            return None;
        }
        if input.small_blind < 0.0 || input.big_blind < 0.0 || input.ante < 0.0 {
            return None;
        }
        if input.stacks.iter().any(|&stack| stack < 0.0) {
            return None;
        }
        if !input.stacks.iter().any(|&stack| stack > 0.0) {
            return None;
        }

        let sb = input.button_index;
        let bb = 1 - sb;
        let payouts = normalize_payouts(&input.payouts);
        let combos: Vec<Vec<[Card; 2]>> = (0..HAND_TYPES as u8).map(expand_combo).collect();
        let unblocked = build_unblocked(&combos);

        let mut ctx = Self {
            sb,
            bb,
            payouts,
            combos,
            unblocked,
            ev_sb_fold: 0.0,
            ev_sb_bb_folds: 0.0,
            ev_bb_fold: 0.0,
            ev_sb_showdown_win: 0.0,
            ev_sb_showdown_lose: 0.0,
            ev_bb_showdown_win: 0.0,
            ev_bb_showdown_lose: 0.0,
            icm_cache: HashMap::new(),
        };

        let sb_fold_stacks = transfer(&input.stacks, sb, bb, input.small_blind + input.ante);
        let bb_fold_stacks = transfer(&input.stacks, bb, sb, input.big_blind + input.ante);
        let sb_wins = showdown_stacks(&input.stacks, sb, bb);
        let sb_loses = showdown_stacks(&input.stacks, bb, sb);

        ctx.ev_sb_fold = ctx.icm_for(&sb_fold_stacks, sb);
        ctx.ev_sb_bb_folds = ctx.icm_for(&bb_fold_stacks, sb);
        ctx.ev_bb_fold = ctx.icm_for(&bb_fold_stacks, bb);
        ctx.ev_sb_showdown_win = ctx.icm_for(&sb_wins, sb);
        ctx.ev_sb_showdown_lose = ctx.icm_for(&sb_loses, sb);
        ctx.ev_bb_showdown_win = ctx.icm_for(&sb_loses, bb);
        ctx.ev_bb_showdown_lose = ctx.icm_for(&sb_wins, bb);

        Some(ctx)
    }

    fn icm_for(&mut self, stacks: &[f64], player: usize) -> f64 {
        let key = [
            (stacks[0] * 1000.0).round() as u64,
            (stacks[1] * 1000.0).round() as u64,
        ];
        if let Some(equity) = self.icm_cache.get(&key) {
            return equity[player];
        }
        let equity = tournament_equity(stacks, &self.payouts);
        let value = equity[player];
        self.icm_cache.insert(key, equity);
        value
    }
}

fn tournament_equity(stacks: &[f64], payouts: &[f64]) -> Vec<f64> {
    if stacks.len() != 2 {
        return icm_equity(stacks, payouts);
    }

    let first = payouts.first().copied().unwrap_or(1.0);
    let second = payouts.get(1).copied().unwrap_or(0.0);

    if stacks[0] <= 0.0 && stacks[1] <= 0.0 {
        return vec![(first + second) / 2.0, (first + second) / 2.0];
    }
    if stacks[0] <= 0.0 {
        return vec![second, first];
    }
    if stacks[1] <= 0.0 {
        return vec![first, second];
    }

    icm_equity(stacks, payouts)
}

fn normalize_payouts(payouts: &[f64]) -> Vec<f64> {
    let sum: f64 = payouts.iter().copied().filter(|payout| *payout > 0.0).sum();
    if payouts.is_empty() || sum <= 0.0 {
        return vec![0.5, 0.5];
    }
    payouts
        .iter()
        .map(|&payout| (payout.max(0.0)) / sum)
        .collect()
}

fn transfer(stacks: &[f64], from: usize, to: usize, amount: f64) -> Vec<f64> {
    let amount = amount.max(0.0).min(stacks[from]);
    let mut next = stacks.to_vec();
    next[from] -= amount;
    next[to] += amount;
    next
}

fn showdown_stacks(stacks: &[f64], winner: usize, loser: usize) -> Vec<f64> {
    let contested = stacks[winner].min(stacks[loser]).max(0.0);
    transfer(stacks, loser, winner, contested)
}

fn build_unblocked(combos: &[Vec<[Card; 2]>]) -> Vec<Vec<[u8; HAND_TYPES]>> {
    combos
        .iter()
        .map(|hero_combos| {
            hero_combos
                .iter()
                .map(|hero| {
                    let mut counts = [0_u8; HAND_TYPES];
                    for (type_idx, villain_combos) in combos.iter().enumerate() {
                        counts[type_idx] = villain_combos
                            .iter()
                            .filter(|villain| !hands_overlap(*hero, **villain))
                            .count() as u8;
                    }
                    counts
                })
                .collect()
        })
        .collect()
}

fn best_response(ev_action: f64, ev_fold: f64, current: f64) -> f64 {
    let diff = ev_action - ev_fold;
    if diff > INDIFFERENT_EPS {
        1.0
    } else if diff < -INDIFFERENT_EPS {
        0.0
    } else {
        current
    }
}

fn ev_push_sb(
    hand_idx: usize,
    call_range: &[f64; HAND_TYPES],
    ctx: &HuContext,
    cache: &EquityCache,
) -> f64 {
    let combo_count = ctx.combos[hand_idx].len();
    if combo_count == 0 {
        return ctx.ev_sb_fold;
    }

    let mut total = 0.0;
    for combo_idx in 0..combo_count {
        total += ev_push_sb_combo(hand_idx, combo_idx, call_range, ctx, cache);
    }
    total / combo_count as f64
}

fn ev_push_sb_combo(
    hand_idx: usize,
    combo_idx: usize,
    call_range: &[f64; HAND_TYPES],
    ctx: &HuContext,
    cache: &EquityCache,
) -> f64 {
    let unblocked = &ctx.unblocked[hand_idx][combo_idx];
    let mut live_weight = 0.0;
    let mut call_weight = 0.0;
    let mut equity_weight = 0.0;
    let hero_idx = hand_idx as u8;

    for (villain_idx, &live) in unblocked.iter().enumerate() {
        if live == 0 {
            continue;
        }
        let live_f = live as f64;
        let call_freq = call_range[villain_idx].clamp(0.0, 1.0);
        live_weight += live_f;
        let weight = live_f * call_freq;
        call_weight += weight;
        if weight > 0.0 {
            equity_weight += weight * cache.equity(hero_idx, villain_idx as u8);
        }
    }

    if live_weight <= 0.0 {
        return ctx.ev_sb_bb_folds;
    }

    let p_call = call_weight / live_weight;
    let p_fold = 1.0 - p_call;
    let equity = if call_weight > 0.0 {
        equity_weight / call_weight
    } else {
        0.5
    };

    let ev_showdown = equity * ctx.ev_sb_showdown_win + (1.0 - equity) * ctx.ev_sb_showdown_lose;
    p_fold * ctx.ev_sb_bb_folds + p_call * ev_showdown
}

fn ev_call_bb(
    hand_idx: usize,
    push_range: &[f64; HAND_TYPES],
    ctx: &HuContext,
    cache: &EquityCache,
) -> f64 {
    let combo_count = ctx.combos[hand_idx].len();
    if combo_count == 0 {
        return ctx.ev_bb_fold;
    }

    let mut total = 0.0;
    for combo_idx in 0..combo_count {
        total += ev_call_bb_combo(hand_idx, combo_idx, push_range, ctx, cache);
    }
    total / combo_count as f64
}

fn ev_call_bb_combo(
    hand_idx: usize,
    combo_idx: usize,
    push_range: &[f64; HAND_TYPES],
    ctx: &HuContext,
    cache: &EquityCache,
) -> f64 {
    let unblocked = &ctx.unblocked[hand_idx][combo_idx];
    let mut push_weight = 0.0;
    let mut equity_weight = 0.0;
    let hero_idx = hand_idx as u8;

    for (villain_idx, &live) in unblocked.iter().enumerate() {
        if live == 0 {
            continue;
        }
        let weight = live as f64 * push_range[villain_idx].clamp(0.0, 1.0);
        if weight <= 0.0 {
            continue;
        }
        push_weight += weight;
        equity_weight += weight * cache.equity(hero_idx, villain_idx as u8);
    }

    if push_weight <= 0.0 {
        return ctx.ev_bb_fold;
    }

    let equity = equity_weight / push_weight;
    equity * ctx.ev_bb_showdown_win + (1.0 - equity) * ctx.ev_bb_showdown_lose
}

#[allow(dead_code)]
fn equity_vs_range_weighted(
    hand: [Card; 2],
    range: &[f64; HAND_TYPES],
    cache: &EquityCache,
) -> f64 {
    cache.equity_vs_weighted_range(hand, range)
}

#[allow(dead_code)]
fn expand_hand_to_combos(hand_idx: u8) -> Vec<[Card; 2]> {
    expand_combo(hand_idx)
}

fn average_sb_equity(
    push_range: &[f64; HAND_TYPES],
    call_range: &[f64; HAND_TYPES],
    ctx: &HuContext,
    cache: &EquityCache,
) -> f64 {
    let mut weighted = 0.0;
    let mut total = 0.0;

    for (hand_idx, &freq) in push_range.iter().enumerate() {
        let weight = ctx.combos[hand_idx].len() as f64;
        if weight == 0.0 {
            continue;
        }
        let ev_push = ev_push_sb(hand_idx, call_range, ctx, cache);
        let freq = freq.clamp(0.0, 1.0);
        weighted += weight * (freq * ev_push + (1.0 - freq) * ctx.ev_sb_fold);
        total += weight;
    }

    if total == 0.0 {
        return 0.5;
    }
    weighted / total
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::OnceLock;

    fn test_cache() -> &'static EquityCache {
        static CACHE: OnceLock<EquityCache> = OnceLock::new();
        CACHE.get_or_init(|| {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("equity_cache.bin");
            EquityCache::load(&path).expect("equity_cache.bin must exist next to the crate")
        })
    }

    fn test_input(stacks: [f64; 2]) -> SolverInput {
        SolverInput {
            stacks: stacks.to_vec(),
            payouts: vec![0.5, 0.3],
            small_blind: 50.0,
            big_blind: 100.0,
            ante: 0.0,
            button_index: 0,
            max_iterations: 50,
            tolerance: 0.001,
        }
    }

    fn range_count(range: &[f64; HAND_TYPES]) -> usize {
        range.iter().filter(|&&freq| freq > 0.5).count()
    }

    fn range_combo_share(range: &[f64; HAND_TYPES]) -> f64 {
        let mut weighted = 0.0;
        let mut total = 0.0;
        for (idx, &freq) in range.iter().enumerate() {
            let combos = expand_combo(idx as u8).len() as f64;
            weighted += freq.clamp(0.0, 1.0) * combos;
            total += combos;
        }
        if total == 0.0 {
            0.0
        } else {
            weighted / total
        }
    }

    #[test]
    fn hu_equal_stacks_converges() {
        let output = solve(&test_input([1000.0, 1000.0]), test_cache());
        assert!(
            output.converged,
            "expected convergence, iterations={}",
            output.iterations_used
        );
        assert!(output.iterations_used <= 50);
        assert!((output.equities[0] + output.equities[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sb_pushes_wider_than_bb_calls() {
        let output = solve(&test_input([1000.0, 1000.0]), test_cache());
        let sb_push = range_count(&output.push_ranges[0]);
        let bb_call = range_count(&output.call_ranges[1]);
        assert!(
            sb_push > bb_call,
            "SB push {sb_push} hands, BB call {bb_call} hands"
        );
    }

    #[test]
    fn short_stack_pushes_wider() {
        let output = solve(&test_input([500.0, 2000.0]), test_cache());
        let share = range_combo_share(&output.push_ranges[0]);
        assert!(
            share > 0.60,
            "short SB expected to push > 60% of combos, got {:.1}%",
            share * 100.0
        );
    }

    #[test]
    fn huge_stack_folds_more() {
        let short = solve(&test_input([500.0, 2000.0]), test_cache());
        let huge = solve(&test_input([5000.0, 500.0]), test_cache());
        let short_width = range_combo_share(&short.push_ranges[0]);
        let huge_width = range_combo_share(&huge.push_ranges[0]);
        assert!(
            huge_width <= short_width + 1e-9,
            "big SB should not push wider than short SB: huge={huge_width:.3} short={short_width:.3}"
        );
    }
}
