use std::time::Instant;

use rayon::prelude::*;

use crate::algorithm::cfr::regret_match;
use crate::algorithm::{run_loop, SolverAlgorithm};
use crate::equity_cache::EquityCache;
use crate::solver::three_max::{
    build_three_max_output, ev_bb_call_vs_both, ev_bb_call_vs_btn, ev_bb_call_vs_sb,
    ev_bb_fold_vs_both, ev_btn_fold, ev_btn_push, ev_sb_call_vs_btn, ev_sb_fold_vs_btn,
    ev_sb_push_after_btn_fold, load_three_way_cache, p_action, persist_three_way_cache,
    print_cache_stats, ThreeMaxContext,
};
use crate::solver::{empty_output, SolverInput, SolverOutput, HAND_TYPES};

const DEFAULT_TOLERANCE: f64 = 0.005;
const DCFR_ALPHA: f64 = 1.5;
const DCFR_BETA: f64 = 0.5;
const DCFR_GAMMA: f64 = 1.0;

pub struct Cfr3Max<'a> {
    input: SolverInput,
    cache: &'a EquityCache,
    ctx: ThreeMaxContext,
    regret_btn: [[f64; 2]; HAND_TYPES],
    regret_sb_vs_push: [[f64; 2]; HAND_TYPES],
    regret_sb_after_fold: [[f64; 2]; HAND_TYPES],
    regret_bb_vs_both: [[f64; 2]; HAND_TYPES],
    regret_bb_vs_btn: [[f64; 2]; HAND_TYPES],
    regret_bb_vs_sb: [[f64; 2]; HAND_TYPES],
    strategy_btn: [[f64; 2]; HAND_TYPES],
    strategy_sb_vs_push: [[f64; 2]; HAND_TYPES],
    strategy_sb_after_fold: [[f64; 2]; HAND_TYPES],
    strategy_bb_vs_both: [[f64; 2]; HAND_TYPES],
    strategy_bb_vs_btn: [[f64; 2]; HAND_TYPES],
    strategy_bb_vs_sb: [[f64; 2]; HAND_TYPES],
    sum: [[f64; HAND_TYPES]; 6],
    weight_sum: f64,
    prev: [[f64; HAND_TYPES]; 6],
    prev_avg: [[f64; HAND_TYPES]; 6],
    iterations_done: usize,
    has_converged: bool,
    solve_started: Instant,
}

impl<'a> Cfr3Max<'a> {
    pub fn init(input: &SolverInput, cache: &'a EquityCache) -> Option<Self> {
        let ctx = ThreeMaxContext::new(input, load_three_way_cache())?;
        Some(Self {
            input: input.clone(),
            cache,
            ctx,
            regret_btn: [[0.0; 2]; HAND_TYPES],
            regret_sb_vs_push: [[0.0; 2]; HAND_TYPES],
            regret_sb_after_fold: [[0.0; 2]; HAND_TYPES],
            regret_bb_vs_both: [[0.0; 2]; HAND_TYPES],
            regret_bb_vs_btn: [[0.0; 2]; HAND_TYPES],
            regret_bb_vs_sb: [[0.0; 2]; HAND_TYPES],
            strategy_btn: [[0.5; 2]; HAND_TYPES],
            strategy_sb_vs_push: [[0.5; 2]; HAND_TYPES],
            strategy_sb_after_fold: [[0.5; 2]; HAND_TYPES],
            strategy_bb_vs_both: [[0.5; 2]; HAND_TYPES],
            strategy_bb_vs_btn: [[0.5; 2]; HAND_TYPES],
            strategy_bb_vs_sb: [[0.5; 2]; HAND_TYPES],
            sum: [[0.0; HAND_TYPES]; 6],
            weight_sum: 0.0,
            prev: [[0.0; HAND_TYPES]; 6],
            prev_avg: [[0.0; HAND_TYPES]; 6],
            iterations_done: 0,
            has_converged: false,
            solve_started: Instant::now(),
        })
    }

    pub fn run(input: &SolverInput, cache: &'a EquityCache) -> SolverOutput {
        match Self::init(input, cache) {
            Some(solver) => run_loop(solver, input.max_iterations),
            None => empty_output(input.stacks.len()),
        }
    }

    fn recompute_strategies(&mut self) {
        for h in 0..HAND_TYPES {
            regret_match(&self.regret_btn[h], &mut self.strategy_btn[h]);
            regret_match(&self.regret_sb_vs_push[h], &mut self.strategy_sb_vs_push[h]);
            regret_match(
                &self.regret_sb_after_fold[h],
                &mut self.strategy_sb_after_fold[h],
            );
            regret_match(&self.regret_bb_vs_both[h], &mut self.strategy_bb_vs_both[h]);
            regret_match(&self.regret_bb_vs_btn[h], &mut self.strategy_bb_vs_btn[h]);
            regret_match(&self.regret_bb_vs_sb[h], &mut self.strategy_bb_vs_sb[h]);
        }
    }

    fn current_ranges(&self) -> [[f64; HAND_TYPES]; 6] {
        let mut ranges = [[0.0; HAND_TYPES]; 6];
        for h in 0..HAND_TYPES {
            ranges[0][h] = self.strategy_btn[h][0];
            ranges[1][h] = self.strategy_sb_vs_push[h][0];
            ranges[2][h] = self.strategy_bb_vs_btn[h][0];
            ranges[3][h] = self.strategy_bb_vs_both[h][0];
            ranges[4][h] = self.strategy_sb_after_fold[h][0];
            ranges[5][h] = self.strategy_bb_vs_sb[h][0];
        }
        ranges
    }

    fn average_ranges(&self) -> [[f64; HAND_TYPES]; 6] {
        if self.weight_sum <= 0.0 {
            return self.current_ranges();
        }
        let mut ranges = [[0.0; HAND_TYPES]; 6];
        for kind in 0..6 {
            for h in 0..HAND_TYPES {
                ranges[kind][h] = (self.sum[kind][h] / self.weight_sum).clamp(0.0, 1.0);
            }
        }
        ranges
    }

    fn accumulate(&mut self, ranges: &[[f64; HAND_TYPES]; 6], t: f64) {
        let tg = t.powf(DCFR_GAMMA);
        let beta = tg / (tg + 1.0);
        self.weight_sum = beta * self.weight_sum + 1.0;
        for kind in 0..6 {
            for h in 0..HAND_TYPES {
                self.sum[kind][h] = beta * self.sum[kind][h] + ranges[kind][h];
            }
        }
    }

    fn compute_evs(&self, ranges: &[[f64; HAND_TYPES]; 6]) -> [Vec<(f64, f64)>; 6] {
        let n = HAND_TYPES * 6;
        let ctx = &self.ctx;
        let cache = self.cache;
        let raw: Vec<(f64, f64)> = if ctx.parallel_hands {
            (0..n)
                .into_par_iter()
                .map(|job| action_evs(job / HAND_TYPES, job % HAND_TYPES, ranges, ctx, cache))
                .collect()
        } else {
            (0..n)
                .map(|job| action_evs(job / HAND_TYPES, job % HAND_TYPES, ranges, ctx, cache))
                .collect()
        };
        let mut out = std::array::from_fn(|_| vec![(0.0, 0.0); HAND_TYPES]);
        for kind in 0..6 {
            let start = kind * HAND_TYPES;
            out[kind].copy_from_slice(&raw[start..start + HAND_TYPES]);
        }
        out
    }

    fn update_regrets(
        &mut self,
        ranges: &[[f64; HAND_TYPES]; 6],
        evs: &[Vec<(f64, f64)>; 6],
        t: f64,
    ) {
        for hand in 0..HAND_TYPES {
            let reach = [
                1.0,
                opponent_reach(1, hand, ranges, &self.ctx),
                opponent_reach(2, hand, ranges, &self.ctx),
                opponent_reach(3, hand, ranges, &self.ctx),
                opponent_reach(4, hand, ranges, &self.ctx),
                opponent_reach(5, hand, ranges, &self.ctx),
            ];
            dcfr(
                &mut self.regret_btn[hand],
                evs[0][hand],
                self.strategy_btn[hand],
                reach[0],
                t,
            );
            dcfr(
                &mut self.regret_sb_vs_push[hand],
                evs[1][hand],
                self.strategy_sb_vs_push[hand],
                reach[1],
                t,
            );
            dcfr(
                &mut self.regret_bb_vs_btn[hand],
                evs[2][hand],
                self.strategy_bb_vs_btn[hand],
                reach[2],
                t,
            );
            dcfr(
                &mut self.regret_bb_vs_both[hand],
                evs[3][hand],
                self.strategy_bb_vs_both[hand],
                reach[3],
                t,
            );
            dcfr(
                &mut self.regret_sb_after_fold[hand],
                evs[4][hand],
                self.strategy_sb_after_fold[hand],
                reach[4],
                t,
            );
            dcfr(
                &mut self.regret_bb_vs_sb[hand],
                evs[5][hand],
                self.strategy_bb_vs_sb[hand],
                reach[5],
                t,
            );
        }
    }

    fn compute_change(&self, ranges: &[[f64; HAND_TYPES]; 6]) -> f64 {
        let mut change = 0.0;
        for kind in 0..6 {
            for h in 0..HAND_TYPES {
                change += (ranges[kind][h] - self.prev[kind][h]).abs();
            }
        }
        change
    }

    fn tolerance(&self) -> f64 {
        self.input.tolerance.max(DEFAULT_TOLERANCE)
    }
}

impl SolverAlgorithm for Cfr3Max<'_> {
    fn iterate(&mut self) {
        self.recompute_strategies();
        let ranges = self.current_ranges();

        let iter_started = Instant::now();
        let hits_before = self.ctx.cache_hits();
        let misses_before = self.ctx.cache_misses();
        let evs = self.compute_evs(&ranges);
        let t = (self.iterations_done + 1) as f64;
        self.update_regrets(&ranges, &evs, t);
        self.accumulate(&ranges, t);

        self.iterations_done += 1;
        let last_change = self.compute_change(&ranges);
        let averaged = self.average_ranges();
        let avg_change = {
            let mut change = 0.0;
            for kind in 0..6 {
                for h in 0..HAND_TYPES {
                    change += (averaged[kind][h] - self.prev_avg[kind][h]).abs();
                }
            }
            change
        };
        self.prev = ranges;
        self.prev_avg = averaged;

        let hits = self.ctx.cache_hits() - hits_before;
        let misses = self.ctx.cache_misses() - misses_before;
        let looked = hits + misses;
        let hit_pct = if looked == 0 {
            100.0
        } else {
            100.0 * hits as f64 / looked as f64
        };
        if self.input.verbose_convergence {
            eprintln!(
                "Iter {}: last_change={last_change:.6} avg_change={avg_change:.6}",
                self.iterations_done
            );
        } else {
            eprintln!(
                "Iter {}: {:.2}s, hits: {hit_pct:.1}%",
                self.iterations_done,
                iter_started.elapsed().as_secs_f64()
            );
        }

        let n = 6.0 * HAND_TYPES as f64;
        let tolerance = self.tolerance();
        let last_mean = last_change / n;
        let avg_mean = avg_change / n;
        if self.iterations_done >= 30
            && (last_change < tolerance
                || last_mean < tolerance
                || avg_change < tolerance
                || avg_mean < tolerance)
        {
            self.has_converged = true;
        }
    }

    fn converged(&self) -> bool {
        self.has_converged
    }

    fn result(&self) -> SolverOutput {
        print_cache_stats(&self.ctx, self.solve_started);
        persist_three_way_cache(&self.ctx);

        let ranges = self.average_ranges();
        build_three_max_output(
            &self.ctx,
            self.cache,
            &ranges[0],
            &ranges[1],
            &ranges[2],
            &ranges[3],
            &ranges[4],
            &ranges[5],
            self.iterations_done,
            self.has_converged,
        )
    }
}

fn action_evs(
    kind: usize,
    hand: usize,
    ranges: &[[f64; HAND_TYPES]; 6],
    ctx: &ThreeMaxContext,
    cache: &EquityCache,
) -> (f64, f64) {
    match kind {
        0 => (
            ev_btn_push(hand, &ranges[1], &ranges[2], &ranges[3], ctx, cache),
            ev_btn_fold(hand, &ranges[4], &ranges[5], ctx, cache),
        ),
        1 => (
            ev_sb_call_vs_btn(hand, &ranges[0], &ranges[3], ctx, cache),
            ev_sb_fold_vs_btn(hand, &ranges[0], &ranges[2], ctx, cache),
        ),
        2 => (
            ev_bb_call_vs_btn(hand, &ranges[0], &ranges[1], ctx, cache),
            ctx.icm_btn_takes_blinds[ctx.bb],
        ),
        3 => (
            ev_bb_call_vs_both(hand, &ranges[0], &ranges[1], ctx, cache),
            ev_bb_fold_vs_both(hand, &ranges[0], &ranges[1], ctx, cache),
        ),
        4 => (
            ev_sb_push_after_btn_fold(hand, &ranges[5], ctx, cache),
            ctx.icm_bb_walks[ctx.sb],
        ),
        _ => (
            ev_bb_call_vs_sb(hand, &ranges[4], ctx, cache),
            ctx.icm_sb_walks[ctx.bb],
        ),
    }
}

fn opponent_reach(
    kind: usize,
    hand: usize,
    ranges: &[[f64; HAND_TYPES]; 6],
    ctx: &ThreeMaxContext,
) -> f64 {
    if ctx.combos[hand].is_empty() {
        return 0.0;
    }
    let unblocked = &ctx.unblocked[hand][0];
    let p_btn_push = p_action(unblocked, &ranges[0]);
    let p_sb_call = p_action(unblocked, &ranges[1]);
    let p_sb_push = p_action(unblocked, &ranges[4]);
    match kind {
        0 => 1.0,
        1 => p_btn_push,
        2 => p_btn_push * (1.0 - p_sb_call),
        3 => p_btn_push * p_sb_call,
        4 => 1.0 - p_btn_push,
        _ => (1.0 - p_btn_push) * p_sb_push,
    }
}

fn dcfr(regret: &mut [f64; 2], ev: (f64, f64), sigma: [f64; 2], reach: f64, t: f64) {
    let expected = sigma[0] * ev.0 + sigma[1] * ev.1;
    let pos = t.powf(DCFR_ALPHA);
    let neg = t.powf(DCFR_BETA);
    let pos_coef = pos / (pos + 1.0);
    let neg_coef = neg / (neg + 1.0);
    for r in regret.iter_mut() {
        *r *= if *r >= 0.0 { pos_coef } else { neg_coef };
    }
    regret[0] += reach * (ev.0 - expected);
    regret[1] += reach * (ev.1 - expected);
}
