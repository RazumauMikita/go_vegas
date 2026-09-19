use std::collections::HashMap;

use crate::card::Card;
use crate::equity_cache::{expand_combo, hands_overlap, EquityCache};
use crate::icm::icm_equity;

pub(crate) const HAND_TYPES: usize = 169;
pub(crate) const INDIFFERENT_EPS: f64 = 1e-9;

#[path = "three_max.rs"]
mod three_max;

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
    /// Число игроков (2 = HU, 3 = 3-max). Если 0 — берётся `stacks.len()`.
    pub num_players: usize,
    /// Печать push_change / call_change по итерациям (3-max FP).
    pub verbose_convergence: bool,
    /// Профилирование горячих EV-функций 3-max (одна последовательная проходка).
    pub profile: bool,
}

/// Диапазоны 3-max push/fold.
#[derive(Debug, Clone)]
pub struct ThreeMaxRanges {
    pub btn_push: [f64; HAND_TYPES],
    pub sb_call_vs_btn: [f64; HAND_TYPES],
    pub bb_call_vs_btn: [f64; HAND_TYPES],
    pub bb_call_vs_btn_and_sb: [f64; HAND_TYPES],
    pub sb_push: [f64; HAND_TYPES],
    pub bb_call_vs_sb: [f64; HAND_TYPES],
}

/// EV-дифференциал (action − fold) × 100 для каждой руки в 3-max.
#[derive(Debug, Clone)]
pub struct ThreeMaxHandEvs {
    pub btn_push: [f64; HAND_TYPES],
    pub sb_call_vs_btn: [f64; HAND_TYPES],
    pub bb_call_vs_btn: [f64; HAND_TYPES],
    pub bb_call_vs_btn_and_sb: [f64; HAND_TYPES],
    pub sb_push: [f64; HAND_TYPES],
    pub bb_call_vs_sb: [f64; HAND_TYPES],
}

pub(crate) const HAND_EV_SCALE: f64 = 100.0;

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
    /// Диапазоны 3-max (если солвер запускался на трёх игроках).
    pub three_max: Option<ThreeMaxRanges>,
    /// EV-дифференциал основного действия каждой позиции (push/call − fold) × 100.
    pub hand_evs: Vec<[f64; HAND_TYPES]>,
    /// EV-дифференциалы для всех 3-max диапазонов.
    pub three_max_hand_evs: Option<ThreeMaxHandEvs>,
}

/// Диагностика EV колла BB (без FP, фиксированные диапазоны по combo-share).
pub fn debug_bb_report(
    input: &SolverInput,
    cache: &EquityCache,
    hand_label: &str,
    btn_combo_share: f64,
    sb_combo_share: f64,
) -> Result<String, String> {
    three_max::debug_bb_report(input, cache, hand_label, btn_combo_share, sb_combo_share)
}

/// Найти равновесие Нэша для push/fold.
///
/// HU (2 игрока) или 3-max (BTN/SB/BB).
pub fn solve(input: &SolverInput, cache: &EquityCache) -> SolverOutput {
    let players = if input.num_players == 2 || input.num_players == 3 {
        input.num_players
    } else {
        input.stacks.len()
    };
    if players != input.stacks.len() {
        return empty_output(input.stacks.len());
    }
    if input.button_index >= input.stacks.len() {
        return empty_output(input.stacks.len());
    }
    if input.max_iterations == 0 {
        return empty_output(input.stacks.len());
    }

    match players {
        2 => solve_hu(input, cache),
        3 => three_max::solve_3max(input, cache),
        _ => empty_output(input.stacks.len()),
    }
}

fn solve_hu(input: &SolverInput, cache: &EquityCache) -> SolverOutput {
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

    let mut hand_evs = vec![[0.0; HAND_TYPES]; 2];
    for hand_idx in 0..HAND_TYPES {
        let ev_push = ev_push_sb(hand_idx, &call_range, &ctx, cache);
        hand_evs[ctx.sb][hand_idx] = (ev_push - ctx.ev_sb_fold) * HAND_EV_SCALE;
        let ev_call = ev_call_bb(hand_idx, &push_range, &ctx, cache);
        hand_evs[ctx.bb][hand_idx] = (ev_call - ctx.ev_bb_fold) * HAND_EV_SCALE;
    }

    SolverOutput {
        push_ranges,
        call_ranges,
        equities,
        iterations_used,
        converged,
        three_max: None,
        hand_evs,
        three_max_hand_evs: None,
    }
}

pub(crate) fn empty_output(players: usize) -> SolverOutput {
    SolverOutput {
        push_ranges: vec![[0.0; HAND_TYPES]; players],
        call_ranges: vec![[0.0; HAND_TYPES]; players],
        equities: vec![0.0; players],
        iterations_used: 0,
        converged: false,
        three_max: None,
        hand_evs: vec![[0.0; HAND_TYPES]; players],
        three_max_hand_evs: None,
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

pub(crate) fn tournament_equity(stacks: &[f64], payouts: &[f64]) -> Vec<f64> {
    let n = stacks.len();
    if n == 0 {
        return Vec::new();
    }

    let alive: Vec<usize> = (0..n).filter(|&index| stacks[index] > 0.0).collect();
    let busted: Vec<usize> = (0..n).filter(|&index| stacks[index] <= 0.0).collect();
    let mut result = vec![0.0; n];

    if !busted.is_empty() {
        let prize: f64 = (alive.len()..n)
            .map(|place| payouts.get(place).copied().unwrap_or(0.0))
            .sum();
        let share = prize / busted.len() as f64;
        for &index in &busted {
            result[index] = share;
        }
    }

    if alive.is_empty() {
        return result;
    }

    let remaining: Vec<f64> = (0..alive.len())
        .map(|place| payouts.get(place).copied().unwrap_or(0.0))
        .collect();
    let remaining_sum: f64 = remaining.iter().sum();
    if remaining_sum <= 0.0 {
        return result;
    }

    if alive.len() == 1 {
        result[alive[0]] = remaining_sum;
        return result;
    }

    let alive_stacks: Vec<f64> = alive.iter().map(|&index| stacks[index]).collect();
    let normalized: Vec<f64> = remaining
        .iter()
        .map(|payout| payout / remaining_sum)
        .collect();
    let icm = icm_equity(&alive_stacks, &normalized);
    for (offset, &index) in alive.iter().enumerate() {
        result[index] = icm[offset] * remaining_sum;
    }
    result
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

pub(crate) fn transfer(stacks: &[f64], from: usize, to: usize, amount: f64) -> Vec<f64> {
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

pub(crate) fn build_unblocked(combos: &[Vec<[Card; 2]>]) -> Vec<Vec<[u8; HAND_TYPES]>> {
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

pub(crate) fn best_response(ev_action: f64, ev_fold: f64, current: f64) -> f64 {
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
    use crate::equity_cache::{combo_index, combo_label};
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
            num_players: 2,
            verbose_convergence: false,
            profile: false,
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

    fn three_max_input() -> SolverInput {
        SolverInput {
            stacks: vec![1000.0, 1000.0, 1000.0],
            payouts: vec![0.5, 0.3, 0.2],
            small_blind: 50.0,
            big_blind: 100.0,
            ante: 0.0,
            button_index: 0,
            max_iterations: 50,
            tolerance: 0.001,
            num_players: 3,
            verbose_convergence: false,
            profile: false,
        }
    }

    fn three_max_output() -> &'static SolverOutput {
        static OUTPUT: std::sync::OnceLock<SolverOutput> = std::sync::OnceLock::new();
        OUTPUT.get_or_init(|| solve(&three_max_input(), test_cache()))
    }

    fn three_max_ranges() -> &'static ThreeMaxRanges {
        three_max_output().three_max.as_ref().expect("3-max ranges")
    }

    #[test]
    fn three_max_converges() {
        let output = three_max_output();
        assert!(output.converged, "3-max should converge");
        assert!(
            output.iterations_used < 100,
            "expected < 100 iterations, got {}",
            output.iterations_used
        );
    }

    #[test]
    fn btn_pushes_wider_than_sb() {
        let ranges = three_max_ranges();
        let btn = range_combo_share(&ranges.btn_push);
        let sb_call = range_combo_share(&ranges.sb_call_vs_btn);
        assert!(
            btn > sb_call,
            "BTN push {btn:.3} should be wider than SB call vs BTN {sb_call:.3}"
        );
    }

    #[test]
    fn bb_calls_tighter_after_sb_call() {
        let ranges = three_max_ranges();
        let vs_both = range_combo_share(&ranges.bb_call_vs_btn_and_sb);
        let vs_btn = range_combo_share(&ranges.bb_call_vs_btn);
        assert!(
            vs_both < vs_btn,
            "BB vs BTN+SB {vs_both:.3} should be tighter than BB vs BTN {vs_btn:.3}"
        );
    }

    #[test]
    fn bb_calls_wider_vs_sb_than_vs_btn() {
        let ranges = three_max_ranges();
        let vs_sb = range_combo_share(&ranges.bb_call_vs_sb);
        let vs_btn = range_combo_share(&ranges.bb_call_vs_btn);
        assert!(
            vs_sb > vs_btn,
            "BB vs SB {vs_sb:.3} should be wider than BB vs BTN {vs_btn:.3}"
        );
    }

    #[test]
    fn three_max_with_cache_still_converges() {
        let output = three_max_output();
        assert!(output.converged, "3-max should converge");
        assert!(
            output.iterations_used < 50,
            "expected < 50 iterations, got {}",
            output.iterations_used
        );

        let ranges = three_max_ranges();
        let expected = [
            ("BTN push", range_combo_share(&ranges.btn_push), 25.5),
            (
                "SB call vs BTN",
                range_combo_share(&ranges.sb_call_vs_btn),
                6.9,
            ),
            (
                "BB call vs BTN",
                range_combo_share(&ranges.bb_call_vs_btn),
                9.1,
            ),
            (
                "BB call vs both",
                range_combo_share(&ranges.bb_call_vs_btn_and_sb),
                1.8,
            ),
            ("SB push", range_combo_share(&ranges.sb_push), 59.9),
            (
                "BB call vs SB",
                range_combo_share(&ranges.bb_call_vs_sb),
                23.4,
            ),
        ];
        for (label, share, expected_pct) in expected {
            let got_pct = share * 100.0;
            assert!(
                (got_pct - expected_pct).abs() <= 0.5,
                "{label}: expected {expected_pct:.1}% ± 0.5, got {got_pct:.1}%"
            );
        }
    }

    #[test]
    fn hrc_ev_diagnostics() {
        let cache = test_cache();
        let ako = [Card::new(14, 0), Card::new(13, 1)];
        let ako_idx = combo_index(ako) as usize;
        assert_eq!(combo_label(ako_idx as u8), "AKo");

        let payouts = [0.625, 0.375];
        let input = SolverInput {
            stacks: vec![1000.0, 1000.0],
            payouts: payouts.to_vec(),
            small_blind: 50.0,
            big_blind: 100.0,
            ante: 0.0,
            button_index: 0,
            max_iterations: 50,
            tolerance: 0.001,
            num_players: 2,
            verbose_convergence: false,
            profile: false,
        };
        let ctx = HuContext::new(&input).expect("HU context");

        let push_58 = top_combo_range(cache, 0.583);
        let call_37 = top_combo_range(cache, 0.374);

        println!("\n========== TEST 1: equity_vs_range AKo vs SB top 58% ==========");
        let equity_list = cache.equity_vs_range(
            ako,
            &push_58
                .iter()
                .enumerate()
                .filter(|(_, &freq)| freq > 0.0)
                .map(|(idx, _)| idx as u8)
                .collect::<Vec<_>>(),
        );
        let equity_weighted = cache.equity_vs_weighted_range(ako, &push_58);
        println!("AKo index: {ako_idx}");
        println!(
            "cache.equity(AKo, AA)  = {:.4}  (expect ~0.12)",
            cache.equity(ako_idx as u8, 0)
        );
        println!(
            "cache.equity(AKo, KK)  = {:.4}  (expect ~0.34)",
            cache.equity(ako_idx as u8, 1)
        );
        println!(
            "cache.equity(AA, AKo)  = {:.4}  (expect ~0.88)",
            cache.equity(0, ako_idx as u8)
        );
        println!(
            "cache.equity(AKo, 32o) = {:.4}  (expect ~0.68)",
            cache.equity(ako_idx as u8, 168)
        );
        println!("equity_vs_range (type list):     {equity_list:.4}");
        println!("equity_vs_weighted_range:        {equity_weighted:.4}");
        println!(
            "SB top 58% type share:          {:.1}%",
            type_share(&push_58) * 100.0
        );
        println!(
            "SB top 58% combo share:         {:.1}%",
            range_combo_share(&push_58) * 100.0
        );

        println!("\n========== TEST 2: EV_call BB AKo vs SB top 58% ==========");
        let (p_sb_push, equity_call, ev_call) = call_stats(ako_idx, &push_58, &ctx, cache);
        let user_fold = icm_equity(&[1050.0, 950.0], &payouts);
        println!("equity (AKo vs SB push 58%):     {equity_call:.4}");
        println!("P(SB push | BB has AKo):         {p_sb_push:.4}");
        println!(
            "ICM showdown BB win  [0,2000]:  {:.4}",
            ctx.ev_bb_showdown_win
        );
        println!(
            "ICM showdown BB lose [2000,0]:  {:.4}",
            ctx.ev_bb_showdown_lose
        );
        println!("EV_call = eq*win + (1-eq)*lose: {ev_call:.6}");
        println!(
            "solver EV_fold BB (BB folds to shove, [1100,900]): {:.6}",
            ctx.ev_bb_fold
        );
        println!(
            "user   EV_fold BB icm([1050,950])[1]:               {:.6}",
            user_fold[1]
        );
        println!(
            "solver EV_call - EV_fold:          {:+.6}",
            ev_call - ctx.ev_bb_fold
        );
        println!(
            "user   EV_call - EV_fold:          {:+.6}",
            ev_call - user_fold[1]
        );
        println!(
            "needed equity vs [1100,900] ICM: {:.4}",
            needed_equity(
                ctx.ev_bb_fold,
                ctx.ev_bb_showdown_win,
                ctx.ev_bb_showdown_lose
            )
        );
        println!(
            "needed equity vs [1050,950] ICM: {:.4}",
            needed_equity(
                user_fold[1],
                ctx.ev_bb_showdown_win,
                ctx.ev_bb_showdown_lose
            )
        );

        println!("\n========== TEST 3: EV_push SB AKo vs BB top 37% ==========");
        let (p_call, p_fold, equity_push, ev_push) = push_stats(ako_idx, &call_37, &ctx, cache);
        let user_sb_fold = icm_equity(&[950.0, 1050.0], &payouts);
        println!("P(BB call | SB has AKo):          {p_call:.4}");
        println!("P(BB fold | SB has AKo):          {p_fold:.4}");
        println!("equity (AKo vs BB call 37%):      {equity_push:.4}");
        println!(
            "ICM SB wins shove [2000,0]:      {:.4}",
            ctx.ev_sb_showdown_win
        );
        println!(
            "ICM SB loses shove [0,2000]:     {:.4}",
            ctx.ev_sb_showdown_lose
        );
        println!("ICM BB folds [1100,900] SB:      {:.4}", ctx.ev_sb_bb_folds);
        println!("EV_push (solver formula):         {ev_push:.6}");
        println!(
            "solver EV_fold SB (SB folds, [950,1050]): {:.6}",
            ctx.ev_sb_fold
        );
        println!(
            "user   EV_fold SB icm([950,1050])[0]:     {:.6}",
            user_sb_fold[0]
        );
        println!(
            "solver EV_push - EV_fold:          {:+.6}",
            ev_push - ctx.ev_sb_fold
        );

        println!("\n========== TEST 4: actual solver ranges ==========");
        let output = solve(&input, cache);
        let ako_push = output.push_ranges[0][ako_idx];
        let ako_call = output.call_ranges[1][ako_idx];
        println!(
            "iterations: {}  converged: {}",
            output.iterations_used, output.converged
        );
        println!("AKo push_ranges[SB]: {ako_push:.4}");
        println!("AKo call_ranges[BB]: {ako_call:.4}");
        let equity_vs_solver_push = cache.equity_vs_weighted_range(ako, &output.push_ranges[0]);
        println!("AKo equity vs solver SB push:  {equity_vs_solver_push:.4}");
        println!(
            "push type share (sum/169):      {:.1}%  (HRC ~58.3%)",
            type_share(&output.push_ranges[0]) * 100.0
        );
        println!(
            "call type share (sum/169):      {:.1}%  (HRC ~37.4%)",
            type_share(&output.call_ranges[1]) * 100.0
        );
        println!(
            "push combo share:               {:.1}%",
            range_combo_share(&output.push_ranges[0]) * 100.0
        );
        println!(
            "call combo share:               {:.1}%",
            range_combo_share(&output.call_ranges[1]) * 100.0
        );
        println!(
            "solver ICM: fold_sb={:.4} fold_bb={:.4} sb_bb_folds={:.4}",
            ctx.ev_sb_fold, ctx.ev_bb_fold, ctx.ev_sb_bb_folds
        );
        println!(
            "solver ICM showdown: sb_win={:.4} sb_lose={:.4} bb_win={:.4} bb_lose={:.4}",
            ctx.ev_sb_showdown_win,
            ctx.ev_sb_showdown_lose,
            ctx.ev_bb_showdown_win,
            ctx.ev_bb_showdown_lose
        );

        let ev_call_vs_solver_push = ev_call_bb(ako_idx, &output.push_ranges[0], &ctx, cache);
        let ev_push_vs_solver_call = ev_push_sb(ako_idx, &output.call_ranges[1], &ctx, cache);
        println!(
            "AKo EV_call vs solver push: {:.6} vs fold {:.6} diff {:+.6}",
            ev_call_vs_solver_push,
            ctx.ev_bb_fold,
            ev_call_vs_solver_push - ctx.ev_bb_fold
        );
        println!(
            "AKo EV_push vs solver call: {:.6} vs fold {:.6} diff {:+.6}",
            ev_push_vs_solver_call,
            ctx.ev_sb_fold,
            ev_push_vs_solver_call - ctx.ev_sb_fold
        );
    }

    fn type_share(range: &[f64; HAND_TYPES]) -> f64 {
        range.iter().sum::<f64>() / HAND_TYPES as f64
    }

    fn top_combo_range(cache: &EquityCache, target_combo_share: f64) -> [f64; HAND_TYPES] {
        let mut ranked: Vec<(usize, f64, f64)> = (0..HAND_TYPES)
            .map(|idx| {
                let combos = expand_combo(idx as u8).len() as f64;
                let vs_random = equity_vs_random(cache, idx as u8);
                (idx, vs_random, combos)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let total_combos: f64 = ranked.iter().map(|item| item.2).sum();
        let mut range = [0.0; HAND_TYPES];
        let mut used = 0.0;
        for &(idx, _, combos) in &ranked {
            if used / total_combos >= target_combo_share {
                break;
            }
            range[idx] = 1.0;
            used += combos;
        }
        range
    }

    fn equity_vs_random(cache: &EquityCache, hero: u8) -> f64 {
        let mut weighted = 0.0;
        let mut total = 0.0;
        for villain in 0..HAND_TYPES as u8 {
            let weight = expand_combo(villain).len() as f64;
            weighted += weight * cache.equity(hero, villain);
            total += weight;
        }
        weighted / total
    }

    fn needed_equity(ev_fold: f64, ev_win: f64, ev_lose: f64) -> f64 {
        if (ev_win - ev_lose).abs() < 1e-12 {
            return 1.0;
        }
        (ev_fold - ev_lose) / (ev_win - ev_lose)
    }

    fn call_stats(
        hand_idx: usize,
        push_range: &[f64; HAND_TYPES],
        ctx: &HuContext,
        cache: &EquityCache,
    ) -> (f64, f64, f64) {
        let combo_count = ctx.combos[hand_idx].len() as f64;
        let mut p_push = 0.0;
        let mut equity = 0.0;
        for combo_idx in 0..ctx.combos[hand_idx].len() {
            let unblocked = &ctx.unblocked[hand_idx][combo_idx];
            let mut live = 0.0;
            let mut push_w = 0.0;
            let mut eq_w = 0.0;
            for (villain_idx, &count) in unblocked.iter().enumerate() {
                let live_f = count as f64;
                live += live_f;
                let weight = live_f * push_range[villain_idx];
                if weight > 0.0 {
                    push_w += weight;
                    eq_w += weight * cache.equity(hand_idx as u8, villain_idx as u8);
                }
            }
            p_push += if live > 0.0 { push_w / live } else { 0.0 };
            equity += if push_w > 0.0 { eq_w / push_w } else { 0.5 };
        }
        p_push /= combo_count;
        equity /= combo_count;
        let ev_call = equity * ctx.ev_bb_showdown_win + (1.0 - equity) * ctx.ev_bb_showdown_lose;
        (p_push, equity, ev_call)
    }

    fn push_stats(
        hand_idx: usize,
        call_range: &[f64; HAND_TYPES],
        ctx: &HuContext,
        cache: &EquityCache,
    ) -> (f64, f64, f64, f64) {
        let combo_count = ctx.combos[hand_idx].len() as f64;
        let mut p_call = 0.0;
        let mut equity = 0.0;
        for combo_idx in 0..ctx.combos[hand_idx].len() {
            let unblocked = &ctx.unblocked[hand_idx][combo_idx];
            let mut live = 0.0;
            let mut call_w = 0.0;
            let mut eq_w = 0.0;
            for (villain_idx, &count) in unblocked.iter().enumerate() {
                let live_f = count as f64;
                live += live_f;
                let weight = live_f * call_range[villain_idx];
                if weight > 0.0 {
                    call_w += weight;
                    eq_w += weight * cache.equity(hand_idx as u8, villain_idx as u8);
                }
            }
            p_call += if live > 0.0 { call_w / live } else { 0.0 };
            equity += if call_w > 0.0 { eq_w / call_w } else { 0.5 };
        }
        p_call /= combo_count;
        equity /= combo_count;
        let p_fold = 1.0 - p_call;
        let ev_showdown =
            equity * ctx.ev_sb_showdown_win + (1.0 - equity) * ctx.ev_sb_showdown_lose;
        let ev_push = p_fold * ctx.ev_sb_bb_folds + p_call * ev_showdown;
        (p_call, p_fold, equity, ev_push)
    }
}
