use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

use rayon::prelude::*;

use crate::algorithm::cfr::regret_match;
use crate::algorithm::{run_loop, SolverAlgorithm};
use crate::bucketing::{Bucketing, NUM_BUCKETS};
use crate::card::Card;
use crate::equity_3way::{apply_permutation, RANK_PERMUTATIONS as PERM3};
use crate::equity_cache::{expand_combo, EquityCache};
use crate::four_way_rank_cache::FourWayRankCache;
use crate::solver::{
    empty_output, tournament_equity, FourMaxRanges, SolverInput, SolverOutput, HAND_EV_SCALE,
    HAND_TYPES,
};
use crate::three_way_rank_cache::ThreeWayRankCache;

const DEFAULT_TOLERANCE: f64 = 0.005;
const MIN_CONVERGENCE_ITERS: usize = 200;
const DCFR_ALPHA: f64 = 1.5;
const DCFR_BETA: f64 = 0.5;
const DCFR_GAMMA: f64 = 1.0;
const BUCKETS: usize = NUM_BUCKETS;
const QUADS: usize = BUCKETS * BUCKETS * BUCKETS * BUCKETS;

const N_UTG: usize = 0;
const N_BTN_CALL: usize = 1;
const N_BTN_PUSH: usize = 2;
const N_SB_PC: usize = 3;
const N_SB_PF: usize = 4;
const N_SB_FP: usize = 5;
const N_SB_FF: usize = 6;
const N_BB_4: usize = 7;
const N_BB_PCF: usize = 8;
const N_BB_PFC: usize = 9;
const N_BB_PFF: usize = 10;
const N_BB_FPC: usize = 11;
const N_BB_FPF: usize = 12;
const N_BB_FFP: usize = 13;
const NODES: usize = 14;

pub(crate) const PERM4: [[usize; 4]; 24] = [
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [1, 2, 0, 3],
    [1, 2, 3, 0],
    [1, 3, 0, 2],
    [1, 3, 2, 0],
    [2, 0, 1, 3],
    [2, 0, 3, 1],
    [2, 1, 0, 3],
    [2, 1, 3, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
    [3, 0, 1, 2],
    [3, 0, 2, 1],
    [3, 1, 0, 2],
    [3, 1, 2, 0],
    [3, 2, 0, 1],
    [3, 2, 1, 0],
];

pub struct Cfr4Max {
    input: SolverInput,
    model: Model,
    regret_utg: [[f64; 2]; HAND_TYPES],
    regret_btn_vs_push: [[f64; 2]; HAND_TYPES],
    regret_btn_after_fold: [[f64; 2]; HAND_TYPES],
    regret_sb_after_push_call: [[f64; 2]; HAND_TYPES],
    regret_sb_after_push_fold: [[f64; 2]; HAND_TYPES],
    regret_sb_after_fold_push: [[f64; 2]; HAND_TYPES],
    regret_sb_after_fold_fold: [[f64; 2]; HAND_TYPES],
    regret_bb_4way: [[f64; 2]; HAND_TYPES],
    regret_bb_after_push_call_fold: [[f64; 2]; HAND_TYPES],
    regret_bb_after_push_fold_call: [[f64; 2]; HAND_TYPES],
    regret_bb_after_push_fold_fold: [[f64; 2]; HAND_TYPES],
    regret_bb_after_fold_push_call: [[f64; 2]; HAND_TYPES],
    regret_bb_after_fold_push_fold: [[f64; 2]; HAND_TYPES],
    regret_bb_after_fold_fold_push: [[f64; 2]; HAND_TYPES],
    strategy_utg: [[f64; 2]; HAND_TYPES],
    strategy_btn_vs_push: [[f64; 2]; HAND_TYPES],
    strategy_btn_after_fold: [[f64; 2]; HAND_TYPES],
    strategy_sb_after_push_call: [[f64; 2]; HAND_TYPES],
    strategy_sb_after_push_fold: [[f64; 2]; HAND_TYPES],
    strategy_sb_after_fold_push: [[f64; 2]; HAND_TYPES],
    strategy_sb_after_fold_fold: [[f64; 2]; HAND_TYPES],
    strategy_bb_4way: [[f64; 2]; HAND_TYPES],
    strategy_bb_after_push_call_fold: [[f64; 2]; HAND_TYPES],
    strategy_bb_after_push_fold_call: [[f64; 2]; HAND_TYPES],
    strategy_bb_after_push_fold_fold: [[f64; 2]; HAND_TYPES],
    strategy_bb_after_fold_push_call: [[f64; 2]; HAND_TYPES],
    strategy_bb_after_fold_push_fold: [[f64; 2]; HAND_TYPES],
    strategy_bb_after_fold_fold_push: [[f64; 2]; HAND_TYPES],
    sum: [[f64; HAND_TYPES]; NODES],
    weight_sum: f64,
    prev: [[f64; HAND_TYPES]; NODES],
    prev_avg: [[f64; HAND_TYPES]; NODES],
    iterations_done: usize,
    has_converged: bool,
    solve_started: Instant,
}

struct Model {
    map: [usize; 4],
    walk: [[f64; 4]; 4],
    hu_win: [[[f64; 4]; 2]; 6],
    three_pay: [[[f64; 6]; 4]; 4],
    three_slots: [[usize; 3]; 4],
    three_folder: [usize; 4],
    four_pay: Vec<[f32; 4]>,
}

struct Terminals {
    walk: [[f64; 4]; 4],
    hu: [[[f64; HAND_TYPES]; 4]; 6],
    three: [[[f64; HAND_TYPES]; 4]; 4],
    four: [[f64; HAND_TYPES]; 4],
}

struct Shared {
    equity: Vec<f32>,
    #[cfg_attr(not(test), allow(dead_code))]
    masks: [u64; HAND_TYPES],
    #[cfg_attr(not(test), allow(dead_code))]
    cards: [[Card; 2]; HAND_TYPES],
    combos: [f64; HAND_TYPES],
    unblock: Vec<Vec<[u8; HAND_TYPES]>>,
    block: [[bool; HAND_TYPES]; HAND_TYPES],
    bucket: [u8; HAND_TYPES],
    bucket_mask: [u64; BUCKETS],
    #[cfg_attr(not(test), allow(dead_code))]
    bucket_cards: [[Card; 2]; BUCKETS],
    three: Vec<[f32; 6]>,
}

pub fn apply_permutation_4way(perm: usize, contested: [f64; 4], uncalled: [f64; 4]) -> [f64; 4] {
    let order = PERM4[perm.min(23)];
    let mut score = [0_u8; 4];
    for (place, &player) in order.iter().enumerate() {
        score[player] = (3 - place) as u8;
    }

    let mut levels = contested;
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut unique = [0.0; 4];
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

    let mut won = [0.0; 4];
    let mut prev = 0.0;
    for level in unique.into_iter().take(n_unique) {
        let mut best = 0_u8;
        let mut n_in = 0_u32;
        let mut in_pot = [false; 4];
        for i in 0..4 {
            if contested[i] >= level {
                in_pot[i] = true;
                n_in += 1;
                best = best.max(score[i]);
            }
        }
        if n_in == 0 {
            prev = level;
            continue;
        }
        let pot = (level - prev) * f64::from(n_in);
        for i in 0..4 {
            if in_pot[i] && score[i] == best {
                won[i] += pot;
                break;
            }
        }
        prev = level;
    }
    for i in 0..4 {
        won[i] += uncalled[i];
    }
    won
}

impl Cfr4Max {
    pub fn init(input: &SolverInput, cache: &EquityCache) -> Option<Self> {
        let solve_started = Instant::now();
        let model = Model::build(input, cache)?;
        let half = || [[0.5, 0.5]; HAND_TYPES];
        let zero = || [[0.0, 0.0]; HAND_TYPES];
        Some(Self {
            input: input.clone(),
            model,
            regret_utg: zero(),
            regret_btn_vs_push: zero(),
            regret_btn_after_fold: zero(),
            regret_sb_after_push_call: zero(),
            regret_sb_after_push_fold: zero(),
            regret_sb_after_fold_push: zero(),
            regret_sb_after_fold_fold: zero(),
            regret_bb_4way: zero(),
            regret_bb_after_push_call_fold: zero(),
            regret_bb_after_push_fold_call: zero(),
            regret_bb_after_push_fold_fold: zero(),
            regret_bb_after_fold_push_call: zero(),
            regret_bb_after_fold_push_fold: zero(),
            regret_bb_after_fold_fold_push: zero(),
            strategy_utg: half(),
            strategy_btn_vs_push: half(),
            strategy_btn_after_fold: half(),
            strategy_sb_after_push_call: half(),
            strategy_sb_after_push_fold: half(),
            strategy_sb_after_fold_push: half(),
            strategy_sb_after_fold_fold: half(),
            strategy_bb_4way: half(),
            strategy_bb_after_push_call_fold: half(),
            strategy_bb_after_push_fold_call: half(),
            strategy_bb_after_push_fold_fold: half(),
            strategy_bb_after_fold_push_call: half(),
            strategy_bb_after_fold_push_fold: half(),
            strategy_bb_after_fold_fold_push: half(),
            sum: [[0.0; HAND_TYPES]; NODES],
            weight_sum: 0.0,
            prev: [[0.5; HAND_TYPES]; NODES],
            prev_avg: [[0.5; HAND_TYPES]; NODES],
            iterations_done: 0,
            has_converged: false,
            solve_started,
        })
    }

    pub fn run(input: &SolverInput, cache: &EquityCache) -> SolverOutput {
        match Self::init(input, cache) {
            Some(solver) => run_loop(solver, input.max_iterations),
            None => empty_output(input.stacks.len()),
        }
    }

    fn recompute_strategies(&mut self) {
        for h in 0..HAND_TYPES {
            regret_match(&self.regret_utg[h], &mut self.strategy_utg[h]);
            regret_match(
                &self.regret_btn_vs_push[h],
                &mut self.strategy_btn_vs_push[h],
            );
            regret_match(
                &self.regret_btn_after_fold[h],
                &mut self.strategy_btn_after_fold[h],
            );
            regret_match(
                &self.regret_sb_after_push_call[h],
                &mut self.strategy_sb_after_push_call[h],
            );
            regret_match(
                &self.regret_sb_after_push_fold[h],
                &mut self.strategy_sb_after_push_fold[h],
            );
            regret_match(
                &self.regret_sb_after_fold_push[h],
                &mut self.strategy_sb_after_fold_push[h],
            );
            regret_match(
                &self.regret_sb_after_fold_fold[h],
                &mut self.strategy_sb_after_fold_fold[h],
            );
            regret_match(&self.regret_bb_4way[h], &mut self.strategy_bb_4way[h]);
            regret_match(
                &self.regret_bb_after_push_call_fold[h],
                &mut self.strategy_bb_after_push_call_fold[h],
            );
            regret_match(
                &self.regret_bb_after_push_fold_call[h],
                &mut self.strategy_bb_after_push_fold_call[h],
            );
            regret_match(
                &self.regret_bb_after_push_fold_fold[h],
                &mut self.strategy_bb_after_push_fold_fold[h],
            );
            regret_match(
                &self.regret_bb_after_fold_push_call[h],
                &mut self.strategy_bb_after_fold_push_call[h],
            );
            regret_match(
                &self.regret_bb_after_fold_push_fold[h],
                &mut self.strategy_bb_after_fold_push_fold[h],
            );
            regret_match(
                &self.regret_bb_after_fold_fold_push[h],
                &mut self.strategy_bb_after_fold_fold_push[h],
            );
        }
    }

    fn frequencies(&self) -> [[f64; HAND_TYPES]; NODES] {
        let mut freq = [[0.0; HAND_TYPES]; NODES];
        for h in 0..HAND_TYPES {
            freq[N_UTG][h] = self.strategy_utg[h][0];
            freq[N_BTN_CALL][h] = self.strategy_btn_vs_push[h][0];
            freq[N_BTN_PUSH][h] = self.strategy_btn_after_fold[h][0];
            freq[N_SB_PC][h] = self.strategy_sb_after_push_call[h][0];
            freq[N_SB_PF][h] = self.strategy_sb_after_push_fold[h][0];
            freq[N_SB_FP][h] = self.strategy_sb_after_fold_push[h][0];
            freq[N_SB_FF][h] = self.strategy_sb_after_fold_fold[h][0];
            freq[N_BB_4][h] = self.strategy_bb_4way[h][0];
            freq[N_BB_PCF][h] = self.strategy_bb_after_push_call_fold[h][0];
            freq[N_BB_PFC][h] = self.strategy_bb_after_push_fold_call[h][0];
            freq[N_BB_PFF][h] = self.strategy_bb_after_push_fold_fold[h][0];
            freq[N_BB_FPC][h] = self.strategy_bb_after_fold_push_call[h][0];
            freq[N_BB_FPF][h] = self.strategy_bb_after_fold_push_fold[h][0];
            freq[N_BB_FFP][h] = self.strategy_bb_after_fold_fold_push[h][0];
        }
        freq
    }

    fn strategy_of(&self, node: usize, hand: usize) -> [f64; 2] {
        match node {
            N_UTG => self.strategy_utg[hand],
            N_BTN_CALL => self.strategy_btn_vs_push[hand],
            N_BTN_PUSH => self.strategy_btn_after_fold[hand],
            N_SB_PC => self.strategy_sb_after_push_call[hand],
            N_SB_PF => self.strategy_sb_after_push_fold[hand],
            N_SB_FP => self.strategy_sb_after_fold_push[hand],
            N_SB_FF => self.strategy_sb_after_fold_fold[hand],
            N_BB_4 => self.strategy_bb_4way[hand],
            N_BB_PCF => self.strategy_bb_after_push_call_fold[hand],
            N_BB_PFC => self.strategy_bb_after_push_fold_call[hand],
            N_BB_PFF => self.strategy_bb_after_push_fold_fold[hand],
            N_BB_FPC => self.strategy_bb_after_fold_push_call[hand],
            N_BB_FPF => self.strategy_bb_after_fold_push_fold[hand],
            _ => self.strategy_bb_after_fold_fold_push[hand],
        }
    }

    fn regret_mut(&mut self, node: usize) -> &mut [[f64; 2]; HAND_TYPES] {
        match node {
            N_UTG => &mut self.regret_utg,
            N_BTN_CALL => &mut self.regret_btn_vs_push,
            N_BTN_PUSH => &mut self.regret_btn_after_fold,
            N_SB_PC => &mut self.regret_sb_after_push_call,
            N_SB_PF => &mut self.regret_sb_after_push_fold,
            N_SB_FP => &mut self.regret_sb_after_fold_push,
            N_SB_FF => &mut self.regret_sb_after_fold_fold,
            N_BB_4 => &mut self.regret_bb_4way,
            N_BB_PCF => &mut self.regret_bb_after_push_call_fold,
            N_BB_PFC => &mut self.regret_bb_after_push_fold_call,
            N_BB_PFF => &mut self.regret_bb_after_push_fold_fold,
            N_BB_FPC => &mut self.regret_bb_after_fold_push_call,
            N_BB_FPF => &mut self.regret_bb_after_fold_push_fold,
            _ => &mut self.regret_bb_after_fold_fold_push,
        }
    }

    fn accumulate(&mut self, freq: &[[f64; HAND_TYPES]; NODES], t: f64) {
        let tg = t.powf(DCFR_GAMMA);
        let beta = tg / (tg + 1.0);
        self.weight_sum = beta * self.weight_sum + 1.0;
        for node in 0..NODES {
            for h in 0..HAND_TYPES {
                self.sum[node][h] = beta * self.sum[node][h] + freq[node][h];
            }
        }
    }

    fn average_freq(&self) -> [[f64; HAND_TYPES]; NODES] {
        if self.weight_sum <= 0.0 {
            return self.frequencies();
        }
        let mut freq = [[0.0; HAND_TYPES]; NODES];
        for node in 0..NODES {
            for h in 0..HAND_TYPES {
                freq[node][h] = (self.sum[node][h] / self.weight_sum).clamp(0.0, 1.0);
            }
        }
        freq
    }

    fn tolerance(&self) -> f64 {
        self.input.tolerance.max(DEFAULT_TOLERANCE)
    }
}

impl SolverAlgorithm for Cfr4Max {
    fn iterate(&mut self) {
        let iter_started = Instant::now();
        self.recompute_strategies();
        let freq = self.frequencies();
        let opp = opponent_reach(&freq);
        let terminals = self.model.terminals(&freq);
        let utg_fold = fold_freq(&freq[N_UTG]);
        let t = (self.iterations_done + 1) as f64;

        for node in 0..NODES {
            for hand in 0..HAND_TYPES {
                let ev = action_ev(node, hand, &opp, &freq, &terminals, &self.model, &utg_fold);
                let sigma = self.strategy_of(node, hand);
                let reach = reach_of(node, hand, &opp);
                dcfr(self.regret_mut(node), hand, ev, sigma, reach, t);
            }
        }

        if t > 1.0 {
            self.accumulate(&freq, t);
        }
        self.iterations_done += 1;

        let averaged = self.average_freq();
        let mut last_change = 0.0;
        let mut avg_change = 0.0;
        for node in 0..NODES {
            for h in 0..HAND_TYPES {
                last_change += (freq[node][h] - self.prev[node][h]).abs();
                avg_change += (averaged[node][h] - self.prev_avg[node][h]).abs();
            }
        }
        self.prev = freq;
        self.prev_avg = averaged;

        let n = (NODES * HAND_TYPES) as f64;
        let tol = self.tolerance();
        if self.iterations_done >= MIN_CONVERGENCE_ITERS
            && (last_change / n < tol || avg_change / n < tol)
        {
            self.has_converged = true;
        }

        eprintln!(
            "Iter {}: {:.2}s last_mean={:.6} avg_mean={:.6}",
            self.iterations_done,
            iter_started.elapsed().as_secs_f64(),
            last_change / n,
            avg_change / n
        );
    }

    fn converged(&self) -> bool {
        self.has_converged
    }

    fn result(&self) -> SolverOutput {
        let freq = self.average_freq();
        let opp = opponent_reach(&freq);
        let terminals = self.model.terminals(&freq);
        let utg_fold = fold_freq(&freq[N_UTG]);
        let mut evs = [[(0.0, 0.0); HAND_TYPES]; NODES];
        for node in 0..NODES {
            for hand in 0..HAND_TYPES {
                evs[node][hand] =
                    action_ev(node, hand, &opp, &freq, &terminals, &self.model, &utg_fold);
            }
        }

        let shared = shared_cards();
        let mut equities = vec![0.0; 4];
        for canon in 0..4 {
            let mut num = 0.0;
            let mut den = 0.0;
            for hand in 0..HAND_TYPES {
                let w = shared.combos[hand];
                num += w * game_ev(canon, hand, &freq, &opp, &evs, &self.model);
                den += w;
            }
            let real = self.model.map[canon];
            equities[real] = if den > 0.0 { num / den } else { 0.25 };
        }

        let mut push_ranges = vec![[0.0; HAND_TYPES]; 4];
        let mut call_ranges = vec![[0.0; HAND_TYPES]; 4];
        let mut hand_evs = vec![[0.0; HAND_TYPES]; 4];
        let seats = self.model.map;
        push_ranges[seats[0]] = freq[N_UTG];
        push_ranges[seats[1]] = freq[N_BTN_PUSH];
        call_ranges[seats[1]] = freq[N_BTN_CALL];
        push_ranges[seats[2]] = freq[N_SB_FF];
        call_ranges[seats[2]] = freq[N_SB_PF];
        call_ranges[seats[3]] = freq[N_BB_FFP];
        for hand in 0..HAND_TYPES {
            hand_evs[seats[0]][hand] = diff(evs[N_UTG][hand]);
            hand_evs[seats[1]][hand] = diff(evs[N_BTN_PUSH][hand]);
            hand_evs[seats[2]][hand] = diff(evs[N_SB_FF][hand]);
            hand_evs[seats[3]][hand] = diff(evs[N_BB_4][hand]);
        }

        let last = self.frequencies();
        let elapsed = self.solve_started.elapsed().as_secs_f64();
        eprintln!(
            "4-max CFR: iterations={} converged={} time={elapsed:.2}s",
            self.iterations_done, self.has_converged
        );
        eprintln!(
            "4-max ranges last/avg: CO {:.1}/{:.1} BTNpush {:.1}/{:.1} SBpush {:.1}/{:.1} BBvsSB {:.1}/{:.1}",
            combo_share(&last[N_UTG]) * 100.0,
            combo_share(&freq[N_UTG]) * 100.0,
            combo_share(&last[N_BTN_PUSH]) * 100.0,
            combo_share(&freq[N_BTN_PUSH]) * 100.0,
            combo_share(&last[N_SB_FF]) * 100.0,
            combo_share(&freq[N_SB_FF]) * 100.0,
            combo_share(&last[N_BB_FFP]) * 100.0,
            combo_share(&freq[N_BB_FFP]) * 100.0,
        );

        SolverOutput {
            push_ranges,
            call_ranges,
            equities,
            iterations_used: self.iterations_done,
            converged: self.has_converged,
            three_max: None,
            hand_evs,
            three_max_hand_evs: None,
            four_max: Some(FourMaxRanges {
                utg_push: freq[N_UTG],
                btn_call_vs_push: freq[N_BTN_CALL],
                btn_push: freq[N_BTN_PUSH],
                sb_call_vs_utg_btn: freq[N_SB_PC],
                sb_call_vs_utg: freq[N_SB_PF],
                sb_call_vs_btn: freq[N_SB_FP],
                sb_push: freq[N_SB_FF],
                bb_call_4way: freq[N_BB_4],
                bb_call_vs_utg_btn: freq[N_BB_PCF],
                bb_call_vs_utg_sb: freq[N_BB_PFC],
                bb_call_vs_utg: freq[N_BB_PFF],
                bb_call_vs_btn_sb: freq[N_BB_FPC],
                bb_call_vs_btn: freq[N_BB_FPF],
                bb_call_vs_sb: freq[N_BB_FFP],
            }),
            five_max: None,
        }
    }
}

fn diff(ev: (f64, f64)) -> f64 {
    (ev.0 - ev.1) * HAND_EV_SCALE
}

fn combo_share(range: &[f64; HAND_TYPES]) -> f64 {
    let shared = shared_cards();
    let mut weighted = 0.0;
    let mut total = 0.0;
    for h in 0..HAND_TYPES {
        total += shared.combos[h];
        weighted += shared.combos[h] * range[h].clamp(0.0, 1.0);
    }
    if total > 0.0 {
        weighted / total
    } else {
        0.0
    }
}

fn dcfr(
    table: &mut [[f64; 2]; HAND_TYPES],
    hand: usize,
    ev: (f64, f64),
    sigma: [f64; 2],
    reach: f64,
    t: f64,
) {
    let regret = &mut table[hand];
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

fn opponent_reach(freq: &[[f64; HAND_TYPES]; NODES]) -> [[f64; HAND_TYPES]; NODES] {
    let shared = shared_cards();
    let mut opp = [[0.0; HAND_TYPES]; NODES];
    for node in 0..NODES {
        let column: Vec<f64> = (0..HAND_TYPES)
            .into_par_iter()
            .map(|hand| p_action(hand, &freq[node], shared))
            .collect();
        opp[node].copy_from_slice(&column);
    }
    opp
}

fn p_action(hero: usize, freq: &[f64; HAND_TYPES], shared: &Shared) -> f64 {
    let mut live = 0.0;
    let mut acted = 0.0;
    for unblocked in &shared.unblock[hero] {
        for (opp, &count) in unblocked.iter().enumerate() {
            let w = f64::from(count);
            if w == 0.0 {
                continue;
            }
            live += w;
            acted += w * freq[opp].clamp(0.0, 1.0);
        }
    }
    if live <= 0.0 {
        0.0
    } else {
        acted / live
    }
}

fn types_block(a: usize, b: usize, shared: &Shared) -> bool {
    shared.block[a][b]
}

fn reach_of(node: usize, hand: usize, opp: &[[f64; HAND_TYPES]; NODES]) -> f64 {
    let p = |n: usize| opp[n][hand];
    match node {
        N_UTG => 1.0,
        N_BTN_CALL => p(N_UTG),
        N_BTN_PUSH => 1.0 - p(N_UTG),
        N_SB_PC => p(N_UTG) * p(N_BTN_CALL),
        N_SB_PF => p(N_UTG) * (1.0 - p(N_BTN_CALL)),
        N_SB_FP => (1.0 - p(N_UTG)) * p(N_BTN_PUSH),
        N_SB_FF => (1.0 - p(N_UTG)) * (1.0 - p(N_BTN_PUSH)),
        N_BB_4 => p(N_UTG) * p(N_BTN_CALL) * p(N_SB_PC),
        N_BB_PCF => p(N_UTG) * p(N_BTN_CALL) * (1.0 - p(N_SB_PC)),
        N_BB_PFC => p(N_UTG) * (1.0 - p(N_BTN_CALL)) * p(N_SB_PF),
        N_BB_PFF => p(N_UTG) * (1.0 - p(N_BTN_CALL)) * (1.0 - p(N_SB_PF)),
        N_BB_FPC => (1.0 - p(N_UTG)) * p(N_BTN_PUSH) * p(N_SB_FP),
        N_BB_FPF => (1.0 - p(N_UTG)) * p(N_BTN_PUSH) * (1.0 - p(N_SB_FP)),
        _ => (1.0 - p(N_UTG)) * (1.0 - p(N_BTN_PUSH)) * p(N_SB_FF),
    }
}

fn action_ev(
    node: usize,
    h: usize,
    opp: &[[f64; HAND_TYPES]; NODES],
    freq: &[[f64; HAND_TYPES]; NODES],
    t: &Terminals,
    model: &Model,
    utg_fold: &[f64; HAND_TYPES],
) -> (f64, f64) {
    let p = |n: usize| opp[n][h];
    let mix = |prob: f64, a: f64, b: f64| prob * a + (1.0 - prob) * b;
    match node {
        N_UTG => {
            let pc = p(N_BTN_CALL);
            let pf = 1.0 - pc;
            let push = pc * p(N_SB_PC) * mix(p(N_BB_4), t.four[0][h], t.three[0][0][h])
                + pc * (1.0 - p(N_SB_PC)) * mix(p(N_BB_PCF), t.three[1][0][h], t.hu[5][0][h])
                + pf * p(N_SB_PF) * mix(p(N_BB_PFC), t.three[2][0][h], t.hu[4][0][h])
                + pf * (1.0 - p(N_SB_PF)) * mix(p(N_BB_PFF), t.hu[3][0][h], t.walk[0][0]);
            let bp = p(N_BTN_PUSH);
            let fold = bp * p(N_SB_FP) * mix(p(N_BB_FPC), t.three[3][0][h], t.hu[2][0][h])
                + bp * (1.0 - p(N_SB_FP)) * mix(p(N_BB_FPF), t.hu[1][0][h], t.walk[1][0])
                + (1.0 - bp) * p(N_SB_FF) * mix(p(N_BB_FFP), t.hu[0][0][h], t.walk[2][0])
                + (1.0 - bp) * (1.0 - p(N_SB_FF)) * t.walk[3][0];
            (push, fold)
        }
        N_BTN_CALL => {
            let call = p(N_SB_PC) * mix(p(N_BB_4), t.four[1][h], t.three[0][1][h])
                + (1.0 - p(N_SB_PC)) * mix(p(N_BB_PCF), t.three[1][1][h], t.hu[5][1][h]);
            let fold = p(N_SB_PF) * mix(p(N_BB_PFC), t.three[2][1][h], t.hu[4][1][h])
                + (1.0 - p(N_SB_PF)) * mix(p(N_BB_PFF), t.hu[3][1][h], t.walk[0][1]);
            (call, fold)
        }
        N_BTN_PUSH => btn_push_ev(h, freq, t, model, utg_fold),
        N_SB_PC => (
            mix(p(N_BB_4), t.four[2][h], t.three[0][2][h]),
            mix(p(N_BB_PCF), t.three[1][2][h], t.hu[5][2][h]),
        ),
        N_SB_PF => (
            mix(p(N_BB_PFC), t.three[2][2][h], t.hu[4][2][h]),
            mix(p(N_BB_PFF), t.hu[3][2][h], t.walk[0][2]),
        ),
        N_SB_FP => (
            mix(p(N_BB_FPC), t.three[3][2][h], t.hu[2][2][h]),
            mix(p(N_BB_FPF), t.hu[1][2][h], t.walk[1][2]),
        ),
        N_SB_FF => (mix(p(N_BB_FFP), t.hu[0][2][h], t.walk[2][2]), t.walk[3][2]),
        N_BB_4 => (t.four[3][h], t.three[0][3][h]),
        N_BB_PCF => (t.three[1][3][h], t.hu[5][3][h]),
        N_BB_PFC => (t.three[2][3][h], t.hu[4][3][h]),
        N_BB_PFF => (t.hu[3][3][h], t.walk[0][3]),
        N_BB_FPC => (t.three[3][3][h], t.hu[2][3][h]),
        N_BB_FPF => (t.hu[1][3][h], t.walk[1][3]),
        _ => (t.hu[0][3][h], t.walk[2][3]),
    }
}

fn fold_freq(push: &[f64; HAND_TYPES]) -> [f64; HAND_TYPES] {
    let mut fold = [0.0; HAND_TYPES];
    for h in 0..HAND_TYPES {
        fold[h] = 1.0 - push[h].clamp(0.0, 1.0);
    }
    fold
}

fn avg_remaining() -> &'static Vec<[f64; HAND_TYPES]> {
    static AVG: OnceLock<Vec<[f64; HAND_TYPES]>> = OnceLock::new();
    AVG.get_or_init(|| {
        let shared = shared_cards();
        let mut avg = vec![[0.0; HAND_TYPES]; HAND_TYPES];
        for u in 0..HAND_TYPES {
            let rows = &shared.unblock[u];
            let n = rows.len().max(1) as f64;
            for row in rows {
                for s in 0..HAND_TYPES {
                    avg[u][s] += f64::from(row[s]);
                }
            }
            for s in 0..HAND_TYPES {
                avg[u][s] /= n;
            }
        }
        avg
    })
}

fn reweight_after_fold(
    row: &[f64; HAND_TYPES],
    fold: &[f64; HAND_TYPES],
    avg_live: &[[f64; HAND_TYPES]],
    combos: &[f64; HAND_TYPES],
) -> [f64; HAND_TYPES] {
    let mut p_u = [0.0; HAND_TYPES];
    let mut z = 0.0;
    for u in 0..HAND_TYPES {
        let w = row[u] * fold[u];
        p_u[u] = w;
        z += w;
    }
    if z <= 0.0 {
        return *row;
    }
    let mut out = [0.0; HAND_TYPES];
    for s in 0..HAND_TYPES {
        let base = row[s];
        if base <= 0.0 || combos[s] <= 0.0 {
            continue;
        }
        let mut factor = 0.0;
        for (u, &pu) in p_u.iter().enumerate() {
            if pu == 0.0 {
                continue;
            }
            factor += pu * (avg_live[u][s] / combos[s]);
        }
        out[s] = base * (factor / z);
    }
    out
}

fn btn_push_ev(
    h: usize,
    freq: &[[f64; HAND_TYPES]; NODES],
    t: &Terminals,
    model: &Model,
    utg_fold: &[f64; HAND_TYPES],
) -> (f64, f64) {
    let shared = shared_cards();
    let rows = &shared.unblock[h];
    if rows.is_empty() {
        return (t.walk[1][1], t.walk[2][1]);
    }

    let steal = t.walk[1][1];
    let sb_takes = t.walk[2][1];
    let bb_walk = t.walk[3][1];
    let three = t.three[3][1][h];
    let spec = t.hu[0][1][h];
    let btn_win_vs_bb = model.hu_win[1][0][1];
    let btn_lose_vs_bb = model.hu_win[1][1][1];
    let btn_win_vs_sb = model.hu_win[2][0][1];
    let btn_lose_vs_sb = model.hu_win[2][1][1];
    let avg_live = avg_remaining();

    let mut mean = [0.0; HAND_TYPES];
    for row in rows {
        for (s, slot) in mean.iter_mut().enumerate() {
            *slot += f64::from(row[s]);
        }
    }
    let n = rows.len() as f64;
    for slot in mean.iter_mut() {
        *slot /= n;
    }
    let live = reweight_after_fold(&mean, utg_fold, avg_live, &shared.combos);
    let p_sb_call = row_p_w(&live, &freq[N_SB_FP]);
    let p_bb_both = row_p_w(&live, &freq[N_BB_FPC]);
    let p_bb_vs_btn = row_p_w(&live, &freq[N_BB_FPF]);
    let p_sb_shove = row_p_w(&live, &freq[N_SB_FF]);
    let p_bb_vs_sb = row_p_w(&live, &freq[N_BB_FFP]);
    let hu_bb = row_hu_w(
        h,
        &live,
        &freq[N_BB_FPF],
        btn_win_vs_bb,
        btn_lose_vs_bb,
        shared,
    );
    let hu_sb = row_hu_w(
        h,
        &live,
        &freq[N_SB_FP],
        btn_win_vs_sb,
        btn_lose_vs_sb,
        shared,
    );
    let mix = |prob: f64, a: f64, b: f64| prob * a + (1.0 - prob) * b;
    let push = p_sb_call * mix(p_bb_both, three, hu_sb)
        + (1.0 - p_sb_call) * mix(p_bb_vs_btn, hu_bb, steal);
    let fold = p_sb_shove * mix(p_bb_vs_sb, spec, sb_takes) + (1.0 - p_sb_shove) * bb_walk;
    (push, fold)
}

fn row_p_w(row: &[f64; HAND_TYPES], freq: &[f64; HAND_TYPES]) -> f64 {
    let mut live = 0.0;
    let mut acted = 0.0;
    for (opp, &w) in row.iter().enumerate() {
        if w <= 0.0 {
            continue;
        }
        live += w;
        acted += w * freq[opp].clamp(0.0, 1.0);
    }
    if live <= 0.0 {
        0.0
    } else {
        acted / live
    }
}

fn row_hu_w(
    hero: usize,
    row: &[f64; HAND_TYPES],
    villain: &[f64; HAND_TYPES],
    icm_win: f64,
    icm_lose: f64,
    shared: &Shared,
) -> f64 {
    let mut weight = 0.0;
    let mut equity = 0.0;
    for v in 0..HAND_TYPES {
        let live = row[v] * villain[v].clamp(0.0, 1.0);
        if live <= 0.0 {
            continue;
        }
        weight += live;
        equity += live * f64::from(shared.equity[hero * HAND_TYPES + v]);
    }
    let p = if weight > 0.0 { equity / weight } else { 0.5 };
    p * icm_win + (1.0 - p) * icm_lose
}

fn game_ev(
    seat: usize,
    h: usize,
    freq: &[[f64; HAND_TYPES]; NODES],
    opp: &[[f64; HAND_TYPES]; NODES],
    evs: &[[(f64, f64); HAND_TYPES]; NODES],
    model: &Model,
) -> f64 {
    let mix = |node: usize| {
        let (a, f) = evs[node][h];
        let p = freq[node][h];
        p * a + (1.0 - p) * f
    };
    let p = |n: usize| opp[n][h];
    match seat {
        0 => mix(N_UTG),
        1 => {
            let after_push = mix(N_BTN_CALL);
            let after_fold = mix(N_BTN_PUSH);
            p(N_UTG) * after_push + (1.0 - p(N_UTG)) * after_fold
        }
        2 => {
            p(N_UTG) * (p(N_BTN_CALL) * mix(N_SB_PC) + (1.0 - p(N_BTN_CALL)) * mix(N_SB_PF))
                + (1.0 - p(N_UTG))
                    * (p(N_BTN_PUSH) * mix(N_SB_FP) + (1.0 - p(N_BTN_PUSH)) * mix(N_SB_FF))
        }
        _ => {
            let idle = model.walk[3][3];
            p(N_UTG)
                * p(N_BTN_CALL)
                * (p(N_SB_PC) * mix(N_BB_4) + (1.0 - p(N_SB_PC)) * mix(N_BB_PCF))
                + p(N_UTG)
                    * (1.0 - p(N_BTN_CALL))
                    * (p(N_SB_PF) * mix(N_BB_PFC) + (1.0 - p(N_SB_PF)) * mix(N_BB_PFF))
                + (1.0 - p(N_UTG))
                    * p(N_BTN_PUSH)
                    * (p(N_SB_FP) * mix(N_BB_FPC) + (1.0 - p(N_SB_FP)) * mix(N_BB_FPF))
                + (1.0 - p(N_UTG))
                    * (1.0 - p(N_BTN_PUSH))
                    * (p(N_SB_FF) * mix(N_BB_FFP) + (1.0 - p(N_SB_FF)) * idle)
        }
    }
}

impl Model {
    fn build(input: &SolverInput, _cache: &EquityCache) -> Option<Self> {
        if input.stacks.len() != 4 || input.button_index >= 4 {
            return None;
        }
        if input.small_blind < 0.0 || input.big_blind < 0.0 || input.ante < 0.0 {
            return None;
        }
        if input.stacks.iter().any(|stack| *stack < 0.0) || input.payouts.is_empty() {
            return None;
        }

        let btn = input.button_index;
        let sb = (btn + 1) % 4;
        let bb = (btn + 2) % 4;
        let utg = (btn + 3) % 4;
        let map = [utg, btn, sb, bb];
        let base = [
            input.stacks[utg],
            input.stacks[btn],
            input.stacks[sb],
            input.stacks[bb],
        ];
        let dead = [
            input.ante.min(base[0]).max(0.0),
            input.ante.min(base[1]).max(0.0),
            (input.small_blind + input.ante).min(base[2]).max(0.0),
            (input.big_blind + input.ante).min(base[3]).max(0.0),
        ];
        let payouts = &input.payouts;

        let walk = [
            icm4(&award(base, 0, &[1, 2, 3], dead), payouts),
            icm4(&award(base, 1, &[0, 2, 3], dead), payouts),
            icm4(&award(base, 2, &[0, 1, 3], dead), payouts),
            icm4(&award(base, 3, &[0, 1, 2], dead), payouts),
        ];

        let hu_seats = [(2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1)];
        let mut hu_win = [[[0.0; 4]; 2]; 6];
        for (i, (a, b)) in hu_seats.into_iter().enumerate() {
            let folders = folders_of(a, b);
            hu_win[i][0] = icm4(&hu_end(base, a, b, folders, dead, true), payouts);
            hu_win[i][1] = icm4(&hu_end(base, a, b, folders, dead, false), payouts);
        }

        let three_slots = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
        let three_folder = [3, 2, 1, 0];
        let mut three_pay = [[[0.0; 6]; 4]; 4];
        for cfg in 0..4 {
            for perm in 0..6 {
                let stacks = three_end(base, dead, three_slots[cfg], three_folder[cfg], perm);
                let eq = icm4(&stacks, payouts);
                for seat in 0..4 {
                    three_pay[cfg][seat][perm] = eq[seat];
                }
            }
        }

        let (contested, uncalled) = effective4(base);
        let mut four_icm = [[0.0; 4]; 24];
        for perm in 0..24 {
            four_icm[perm] = icm4(&apply_permutation_4way(perm, contested, uncalled), payouts);
        }

        cards()?;
        eprintln!("4-max: payoff table ({QUADS} bucket quads)");
        let four_cache = load_four_way()?;
        let four_pay: Vec<[f32; 4]> = (0..QUADS)
            .map(|idx| {
                let (ug, bg, sg, bbg) = decode_quad(idx);
                let dist = four_cache.lookup(ug, bg, sg, bbg);
                let mut pay = [0.0_f32; 4];
                for perm in 0..24 {
                    let p = dist[perm] as f32;
                    for seat in 0..4 {
                        pay[seat] += p * four_icm[perm][seat] as f32;
                    }
                }
                pay
            })
            .collect();

        Some(Self {
            map,
            walk,
            hu_win,
            three_pay,
            three_slots,
            three_folder,
            four_pay,
        })
    }

    fn terminals(&self, freq: &[[f64; HAND_TYPES]; NODES]) -> Terminals {
        let mut hu = [[[0.0; HAND_TYPES]; 4]; 6];
        let hu_ranges = [
            (&freq[N_SB_FF], &freq[N_BB_FFP], 2, 3),
            (&freq[N_BTN_PUSH], &freq[N_BB_FPF], 1, 3),
            (&freq[N_BTN_PUSH], &freq[N_SB_FP], 1, 2),
            (&freq[N_UTG], &freq[N_BB_PFF], 0, 3),
            (&freq[N_UTG], &freq[N_SB_PF], 0, 2),
            (&freq[N_UTG], &freq[N_BTN_CALL], 0, 1),
        ];
        for (i, (range_a, range_b, seat_a, seat_b)) in hu_ranges.into_iter().enumerate() {
            fill_hu(
                &mut hu[i],
                seat_a,
                seat_b,
                range_a,
                range_b,
                self.hu_win[i][0],
                self.hu_win[i][1],
            );
        }

        let mut three = [[[0.0; HAND_TYPES]; 4]; 4];
        let three_ranges: [[&[f64; HAND_TYPES]; 3]; 4] = [
            [&freq[N_UTG], &freq[N_BTN_CALL], &freq[N_SB_PC]],
            [&freq[N_UTG], &freq[N_BTN_CALL], &freq[N_BB_PCF]],
            [&freq[N_UTG], &freq[N_SB_PF], &freq[N_BB_PFC]],
            [&freq[N_BTN_PUSH], &freq[N_SB_FP], &freq[N_BB_FPC]],
        ];
        for cfg in 0..4 {
            fill_three(
                &three_ranges[cfg],
                &self.three_pay[cfg],
                self.three_slots[cfg],
                self.three_folder[cfg],
                &mut three[cfg],
            );
        }

        let four = fill_four(
            [
                &freq[N_UTG],
                &freq[N_BTN_CALL],
                &freq[N_SB_PC],
                &freq[N_BB_4],
            ],
            &self.four_pay,
        );

        Terminals {
            walk: self.walk,
            hu,
            three,
            four,
        }
    }
}

fn fill_hu(
    out: &mut [[f64; HAND_TYPES]; 4],
    seat_a: usize,
    seat_b: usize,
    range_a: &[f64; HAND_TYPES],
    range_b: &[f64; HAND_TYPES],
    icm_a_wins: [f64; 4],
    icm_b_wins: [f64; 4],
) {
    let shared = shared_cards();
    for h in 0..HAND_TYPES {
        out[seat_a][h] = hu_vs(h, range_b, icm_a_wins[seat_a], icm_b_wins[seat_a], shared);
        out[seat_b][h] = hu_vs(h, range_a, icm_b_wins[seat_b], icm_a_wins[seat_b], shared);
    }
    let p_a = hu_uncond(range_a, range_b, shared);
    for seat in 0..4 {
        if seat == seat_a || seat == seat_b {
            continue;
        }
        let ev = p_a * icm_a_wins[seat] + (1.0 - p_a) * icm_b_wins[seat];
        out[seat] = [ev; HAND_TYPES];
    }
}

fn hu_vs(
    hero: usize,
    villain: &[f64; HAND_TYPES],
    icm_win: f64,
    icm_lose: f64,
    shared: &Shared,
) -> f64 {
    let mut weight = 0.0;
    let mut equity = 0.0;
    for unblock in &shared.unblock[hero] {
        for v in 0..HAND_TYPES {
            let live = f64::from(unblock[v]) * villain[v].clamp(0.0, 1.0);
            if live <= 0.0 {
                continue;
            }
            weight += live;
            equity += live * f64::from(shared.equity[hero * HAND_TYPES + v]);
        }
    }
    let p = if weight > 0.0 { equity / weight } else { 0.5 };
    p * icm_win + (1.0 - p) * icm_lose
}

fn hu_uncond(range_a: &[f64; HAND_TYPES], range_b: &[f64; HAND_TYPES], shared: &Shared) -> f64 {
    let mut weight = 0.0;
    let mut equity = 0.0;
    for a in 0..HAND_TYPES {
        let wa = shared.combos[a] * range_a[a].clamp(0.0, 1.0);
        if wa <= 0.0 {
            continue;
        }
        for b in 0..HAND_TYPES {
            if types_block(a, b, shared) {
                continue;
            }
            let wb = shared.combos[b] * range_b[b].clamp(0.0, 1.0);
            if wb <= 0.0 {
                continue;
            }
            let w = wa * wb;
            weight += w;
            equity += w * f64::from(shared.equity[a * HAND_TYPES + b]);
        }
    }
    if weight > 0.0 {
        equity / weight
    } else {
        0.5
    }
}

struct ThreeAccum {
    num: [[f64; HAND_TYPES]; 3],
    den: [[f64; HAND_TYPES]; 3],
    folder_num: f64,
    folder_den: f64,
}

impl ThreeAccum {
    fn new() -> Self {
        Self {
            num: [[0.0; HAND_TYPES]; 3],
            den: [[0.0; HAND_TYPES]; 3],
            folder_num: 0.0,
            folder_den: 0.0,
        }
    }

    fn merge(mut self, other: Self) -> Self {
        for slot in 0..3 {
            for h in 0..HAND_TYPES {
                self.num[slot][h] += other.num[slot][h];
                self.den[slot][h] += other.den[slot][h];
            }
        }
        self.folder_num += other.folder_num;
        self.folder_den += other.folder_den;
        self
    }
}

fn fill_three(
    ranges: &[&[f64; HAND_TYPES]; 3],
    seat_pay: &[[f64; 6]; 4],
    slots: [usize; 3],
    folder: usize,
    out: &mut [[f64; HAND_TYPES]; 4],
) {
    let shared = shared_cards();
    let mut weight = [[0.0; HAND_TYPES]; 3];
    for slot in 0..3 {
        for h in 0..HAND_TYPES {
            weight[slot][h] = shared.combos[h] * ranges[slot][h].clamp(0.0, 1.0);
        }
    }
    let slot_pay = [seat_pay[slots[0]], seat_pay[slots[1]], seat_pay[slots[2]]];
    let folder_pay = seat_pay[folder];

    let acc = (0..HAND_TYPES)
        .into_par_iter()
        .fold(ThreeAccum::new, |mut acc, h0| {
            let w0 = weight[0][h0];
            for h1 in 0..HAND_TYPES {
                if types_block(h0, h1, shared) {
                    continue;
                }
                let w1 = weight[1][h1];
                if w0 == 0.0 && w1 == 0.0 {
                    continue;
                }
                let row = &shared.three[(h0 * HAND_TYPES + h1) * HAND_TYPES
                    ..(h0 * HAND_TYPES + h1) * HAND_TYPES + HAND_TYPES];
                for h2 in 0..HAND_TYPES {
                    let w2 = weight[2][h2];
                    if w2 == 0.0 && w0 * w1 == 0.0 {
                        continue;
                    }
                    if types_block(h0, h2, shared) || types_block(h1, h2, shared) {
                        continue;
                    }
                    let probs = row[h2];
                    if w1 > 0.0 && w2 > 0.0 {
                        let w = w1 * w2;
                        let pay = dot6(probs, &slot_pay[0]);
                        acc.num[0][h0] += w * pay;
                        acc.den[0][h0] += w;
                    }
                    if w0 > 0.0 && w2 > 0.0 {
                        let w = w0 * w2;
                        let pay = dot6(probs, &slot_pay[1]);
                        acc.num[1][h1] += w * pay;
                        acc.den[1][h1] += w;
                    }
                    if w0 > 0.0 && w1 > 0.0 {
                        let w = w0 * w1;
                        let pay = dot6(probs, &slot_pay[2]);
                        acc.num[2][h2] += w * pay;
                        acc.den[2][h2] += w;
                    }
                    if w0 > 0.0 && w1 > 0.0 && w2 > 0.0 {
                        let w = w0 * w1 * w2;
                        acc.folder_num += w * dot6(probs, &folder_pay);
                        acc.folder_den += w;
                    }
                }
            }
            acc
        })
        .reduce(ThreeAccum::new, ThreeAccum::merge);

    let fallback = [
        mean6(&slot_pay[0]),
        mean6(&slot_pay[1]),
        mean6(&slot_pay[2]),
    ];
    for slot in 0..3 {
        let seat = slots[slot];
        for h in 0..HAND_TYPES {
            out[seat][h] = if acc.den[slot][h] > 0.0 {
                acc.num[slot][h] / acc.den[slot][h]
            } else {
                fallback[slot]
            };
        }
    }
    let folder_ev = if acc.folder_den > 0.0 {
        acc.folder_num / acc.folder_den
    } else {
        mean6(&folder_pay)
    };
    out[folder] = [folder_ev; HAND_TYPES];
}

fn fill_four(ranges: [&[f64; HAND_TYPES]; 4], pay: &[[f32; 4]]) -> [[f64; HAND_TYPES]; 4] {
    let shared = shared_cards();
    let mut blocked = vec![0.0; 4 * BUCKETS * 4 * BUCKETS];
    for hero_seat in 0..4 {
        for hero_bucket in 0..BUCKETS {
            for opp_seat in 0..4 {
                if opp_seat == hero_seat {
                    continue;
                }
                for hand in 0..HAND_TYPES {
                    let bucket = shared.bucket[hand] as usize;
                    if buckets_conflict(hero_bucket, bucket, shared) {
                        continue;
                    }
                    let idx =
                        (((hero_seat * BUCKETS + hero_bucket) * 4 + opp_seat) * BUCKETS) + bucket;
                    blocked[idx] += shared.combos[hand] * ranges[opp_seat][hand].clamp(0.0, 1.0);
                }
            }
        }
    }

    let mass = |hero_seat: usize, hero_bucket: usize, opp_seat: usize, opp_bucket: usize| -> f64 {
        blocked[(((hero_seat * BUCKETS + hero_bucket) * 4 + opp_seat) * BUCKETS) + opp_bucket]
    };

    let acc = (0..BUCKETS)
        .into_par_iter()
        .fold(
            || ([[0.0; BUCKETS]; 4], [[0.0; BUCKETS]; 4]),
            |mut acc, ug| {
                for bg in 0..BUCKETS {
                    if buckets_conflict(ug, bg, shared) {
                        continue;
                    }
                    for sg in 0..BUCKETS {
                        if buckets_conflict(ug, sg, shared) || buckets_conflict(bg, sg, shared) {
                            continue;
                        }
                        for bbg in 0..BUCKETS {
                            if buckets_conflict(ug, bbg, shared)
                                || buckets_conflict(bg, bbg, shared)
                                || buckets_conflict(sg, bbg, shared)
                            {
                                continue;
                            }
                            let w_utg =
                                mass(0, ug, 1, bg) * mass(0, ug, 2, sg) * mass(0, ug, 3, bbg);
                            let w_btn =
                                mass(1, bg, 0, ug) * mass(1, bg, 2, sg) * mass(1, bg, 3, bbg);
                            let w_sb =
                                mass(2, sg, 0, ug) * mass(2, sg, 1, bg) * mass(2, sg, 3, bbg);
                            let w_bb =
                                mass(3, bbg, 0, ug) * mass(3, bbg, 1, bg) * mass(3, bbg, 2, sg);
                            if w_utg == 0.0 && w_btn == 0.0 && w_sb == 0.0 && w_bb == 0.0 {
                                continue;
                            }
                            let cell = pay[((ug * BUCKETS + bg) * BUCKETS + sg) * BUCKETS + bbg];
                            let weights = [w_utg, w_btn, w_sb, w_bb];
                            let buckets = [ug, bg, sg, bbg];
                            for seat in 0..4 {
                                let w = weights[seat];
                                if w == 0.0 {
                                    continue;
                                }
                                let b = buckets[seat];
                                acc.0[seat][b] += w * f64::from(cell[seat]);
                                acc.1[seat][b] += w;
                            }
                        }
                    }
                }
                acc
            },
        )
        .reduce(
            || ([[0.0; BUCKETS]; 4], [[0.0; BUCKETS]; 4]),
            |mut a, b| {
                for seat in 0..4 {
                    for bucket in 0..BUCKETS {
                        a.0[seat][bucket] += b.0[seat][bucket];
                        a.1[seat][bucket] += b.1[seat][bucket];
                    }
                }
                a
            },
        );

    let mut out = [[0.0; HAND_TYPES]; 4];
    for seat in 0..4 {
        let mut by_bucket = [0.0; BUCKETS];
        for bucket in 0..BUCKETS {
            by_bucket[bucket] = if acc.1[seat][bucket] > 0.0 {
                acc.0[seat][bucket] / acc.1[seat][bucket]
            } else {
                0.25
            };
        }
        for hand in 0..HAND_TYPES {
            out[seat][hand] = by_bucket[shared.bucket[hand] as usize];
        }
    }
    out
}

fn buckets_conflict(a: usize, b: usize, shared: &Shared) -> bool {
    let left = shared.bucket_mask[a];
    let right = shared.bucket_mask[b];
    left == 0 || right == 0 || left & right != 0
}

fn dot6(probs: [f32; 6], pay: &[f64; 6]) -> f64 {
    let mut sum = 0.0;
    for i in 0..6 {
        sum += f64::from(probs[i]) * pay[i];
    }
    sum
}

fn mean6(pay: &[f64; 6]) -> f64 {
    pay.iter().sum::<f64>() / 6.0
}

fn icm4(stacks: &[f64; 4], payouts: &[f64]) -> [f64; 4] {
    let eq = tournament_equity(stacks, payouts);
    [eq[0], eq[1], eq[2], eq[3]]
}

fn award(base: [f64; 4], winner: usize, folders: &[usize], dead: [f64; 4]) -> [f64; 4] {
    let mut stacks = base;
    for &folder in folders {
        let take = dead[folder].min(stacks[folder]).max(0.0);
        stacks[folder] -= take;
        stacks[winner] += take;
    }
    stacks
}

fn folders_of(a: usize, b: usize) -> [usize; 2] {
    let mut folders = [0; 2];
    let mut n = 0;
    for seat in 0..4 {
        if seat != a && seat != b {
            folders[n] = seat;
            n += 1;
        }
    }
    folders
}

fn hu_end(
    base: [f64; 4],
    a: usize,
    b: usize,
    folders: [usize; 2],
    dead: [f64; 4],
    a_wins: bool,
) -> [f64; 4] {
    let mut stacks = base;
    let mut pot = 0.0;
    for folder in folders {
        let take = dead[folder].min(stacks[folder]).max(0.0);
        stacks[folder] -= take;
        pot += take;
    }
    let contested = stacks[a].min(stacks[b]).max(0.0);
    let winner = if a_wins { a } else { b };
    let loser = if a_wins { b } else { a };
    stacks[loser] -= contested;
    stacks[winner] += contested + pot;
    stacks
}

fn three_end(
    base: [f64; 4],
    dead: [f64; 4],
    slots: [usize; 3],
    folder: usize,
    perm: usize,
) -> [f64; 4] {
    let active = [base[slots[0]], base[slots[1]], base[slots[2]]];
    let (contested, uncalled) = effective3(active);
    let won = apply_permutation(perm, contested, uncalled);
    let mut stacks = [0.0; 4];
    for slot in 0..3 {
        stacks[slots[slot]] = won[slot];
    }
    let take = dead[folder].min(base[folder]).max(0.0);
    stacks[folder] = base[folder] - take;
    let best = PERM3[perm][0];
    stacks[slots[best]] += take;
    stacks
}

fn effective3(stacks: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let max_other = [
        stacks[1].max(stacks[2]),
        stacks[0].max(stacks[2]),
        stacks[0].max(stacks[1]),
    ];
    let contested = [
        stacks[0].min(max_other[0]).max(0.0),
        stacks[1].min(max_other[1]).max(0.0),
        stacks[2].min(max_other[2]).max(0.0),
    ];
    let uncalled = [
        (stacks[0] - contested[0]).max(0.0),
        (stacks[1] - contested[1]).max(0.0),
        (stacks[2] - contested[2]).max(0.0),
    ];
    (contested, uncalled)
}

fn effective4(stacks: [f64; 4]) -> ([f64; 4], [f64; 4]) {
    let mut contested = [0.0; 4];
    let mut uncalled = [0.0; 4];
    for i in 0..4 {
        let mut max_other = 0.0_f64;
        for j in 0..4 {
            if i != j {
                max_other = max_other.max(stacks[j]);
            }
        }
        contested[i] = stacks[i].min(max_other).max(0.0);
        uncalled[i] = (stacks[i] - contested[i]).max(0.0);
    }
    (contested, uncalled)
}

fn decode_quad(idx: usize) -> (u8, u8, u8, u8) {
    let bbg = idx % BUCKETS;
    let rem = idx / BUCKETS;
    let sg = rem % BUCKETS;
    let rem = rem / BUCKETS;
    let bg = rem % BUCKETS;
    let ug = rem / BUCKETS;
    (ug as u8, bg as u8, sg as u8, bbg as u8)
}

fn cards() -> Option<&'static Shared> {
    static SHARED: OnceLock<Option<Shared>> = OnceLock::new();
    SHARED
        .get_or_init(|| match Shared::load() {
            Ok(shared) => Some(shared),
            Err(error) => {
                eprintln!("4-max caches: {error}");
                None
            }
        })
        .as_ref()
}

fn shared_cards() -> &'static Shared {
    cards().expect("4-max card tables")
}

impl Shared {
    fn load() -> Result<Self, String> {
        let equity_path = find_file("equity_cache.bin")
            .ok_or_else(|| "equity_cache.bin not found".to_string())?;
        let bucket_path =
            find_file("bucketing.bin").ok_or_else(|| "bucketing.bin not found".to_string())?;
        let three_path = find_file("three_way_rank_cache.bin")
            .ok_or_else(|| "three_way_rank_cache.bin not found".to_string())?;

        let equity_cache = EquityCache::load(&equity_path).map_err(|error| error.to_string())?;
        let bucketing = Bucketing::load(&bucket_path).map_err(|error| error.to_string())?;
        let three = ThreeWayRankCache::load(&three_path).map_err(|error| error.to_string())?;

        let mut equity = vec![0.0_f32; HAND_TYPES * HAND_TYPES];
        let mut masks = [0_u64; HAND_TYPES];
        let mut cards = [[Card::new(2, 0); 2]; HAND_TYPES];
        let mut combos = [0.0; HAND_TYPES];
        for hand in 0..HAND_TYPES {
            let list = expand_combo(hand as u8);
            combos[hand] = list.len() as f64;
            let rep = list[0];
            cards[hand] = rep;
            masks[hand] = card_mask(rep);
            for other in 0..HAND_TYPES {
                equity[hand * HAND_TYPES + other] =
                    equity_cache.equity(hand as u8, other as u8) as f32;
            }
        }

        let combos_all: Vec<Vec<[Card; 2]>> = (0..HAND_TYPES as u8).map(expand_combo).collect();
        let unblock = crate::solver::build_unblocked(&combos_all);
        let mut block = [[false; HAND_TYPES]; HAND_TYPES];
        for a in 0..HAND_TYPES {
            for b in 0..HAND_TYPES {
                block[a][b] = unblock[a].is_empty() || unblock[a].iter().all(|row| row[b] == 0);
            }
        }

        let mut bucket = [0_u8; HAND_TYPES];
        let mut bucket_mask = [0_u64; BUCKETS];
        let mut bucket_cards = [[Card::new(2, 0); 2]; BUCKETS];
        let mut seen = [false; BUCKETS];
        for hand in 0..HAND_TYPES {
            let b = bucketing.bucket_of(hand as u8);
            bucket[hand] = b;
            let idx = b as usize;
            if !seen[idx] {
                seen[idx] = true;
                bucket_mask[idx] = masks[hand];
                bucket_cards[idx] = cards[hand];
            }
        }

        eprintln!("4-max: materialize 3-way rank cache");
        let mut three_probs = vec![[1.0_f32 / 6.0; 6]; HAND_TYPES * HAND_TYPES * HAND_TYPES];
        for h0 in 0..HAND_TYPES {
            for h1 in 0..HAND_TYPES {
                for h2 in 0..HAND_TYPES {
                    if let Some(dist) = three.get(h0 as u8, h1 as u8, h2 as u8) {
                        let idx = (h0 * HAND_TYPES + h1) * HAND_TYPES + h2;
                        for perm in 0..6 {
                            three_probs[idx][perm] = dist[perm] as f32;
                        }
                    }
                }
            }
        }

        Ok(Self {
            equity,
            masks,
            cards,
            combos,
            unblock,
            block,
            bucket,
            bucket_mask,
            bucket_cards,
            three: three_probs,
        })
    }
}

fn load_four_way() -> Option<FourWayRankCache> {
    let path = find_file("four_way_rank_cache.bin")?;
    match FourWayRankCache::load(&path) {
        Ok(cache) => Some(cache),
        Err(error) => {
            eprintln!("four_way_rank_cache.bin: {error}");
            None
        }
    }
}

fn find_file(name: &str) -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        PathBuf::from(name),
        manifest.join(name),
        manifest.join("..").join(name),
        manifest
            .join("..")
            .join("poker_ui")
            .join("assets")
            .join(name),
    ];
    candidates.into_iter().find(|path| path.is_file())
}

fn card_mask(hand: [Card; 2]) -> u64 {
    let bit = |card: Card| 1_u64 << ((u32::from(card.rank() - 2) * 4) + u32::from(card.suit()));
    bit(hand[0]) | bit(hand[1])
}

#[cfg(test)]
fn cards_share(hands: &[[Card; 2]]) -> bool {
    let mut seen = [false; 52];
    for hand in hands {
        for card in hand {
            let idx = ((card.rank() - 2) * 4 + card.suit()) as usize;
            if seen[idx] {
                return true;
            }
            seen[idx] = true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity_cache::combo_index;
    use crate::solver::{solve, SolverInput};
    use crate::Algorithm;

    fn assert_stacks(got: [f64; 4], expected: [f64; 4]) {
        for i in 0..4 {
            assert!(
                (got[i] - expected[i]).abs() < 1e-6,
                "seat {i}: got {got:?} expected {expected:?}"
            );
        }
    }

    #[test]
    fn apply_permutation_4way_correct() {
        let uncalled = [0.0; 4];
        assert_stacks(
            apply_permutation_4way(0, [100.0; 4], uncalled),
            [400.0, 0.0, 0.0, 0.0],
        );

        let contested = [100.0, 200.0, 300.0, 300.0];
        let uncalled = [0.0, 0.0, 0.0, 100.0];
        assert_stacks(
            apply_permutation_4way(0, contested, uncalled),
            [400.0, 300.0, 200.0, 100.0],
        );
        assert_stacks(
            apply_permutation_4way(23, contested, uncalled),
            [0.0, 0.0, 0.0, 1000.0],
        );

        let best_is_sb = PERM4.iter().position(|order| order[0] == 2).unwrap();
        let stacks = apply_permutation_4way(best_is_sb, contested, uncalled);
        assert!((stacks.iter().sum::<f64>() - 1000.0).abs() < 1e-6);
        assert!(stacks[2] >= 400.0, "SB wins main pot, got {stacks:?}");

        let wide = [500.0, 500.0, 1000.0, 1000.0];
        assert_stacks(
            apply_permutation_4way(0, wide, [0.0; 4]),
            [2000.0, 0.0, 1000.0, 0.0],
        );

        for perm in 0..24 {
            let stacks = apply_permutation_4way(perm, contested, uncalled);
            assert!(
                (stacks.iter().sum::<f64>() - 1000.0).abs() < 1e-6,
                "perm {perm} chips {stacks:?}"
            );
            assert!(
                (stacks[3] - 100.0).abs() < 1e-6 || stacks[3] > 100.0,
                "perm {perm} keeps uncalled: {stacks:?}"
            );
        }
    }

    #[test]
    fn card_removal_no_conflicts() {
        let shared = cards().expect("caches");
        let mut accepted = 0_u64;
        let mut skipped = 0_u64;
        for a in 0..HAND_TYPES {
            for b in 0..HAND_TYPES {
                let conflict = shared.masks[a] & shared.masks[b] != 0;
                let share = cards_share(&[shared.cards[a], shared.cards[b]]);
                assert_eq!(conflict, share, "hands {a},{b}");
                if conflict {
                    skipped += 1;
                } else {
                    accepted += 1;
                    assert!(!share);
                }
            }
        }
        assert!(accepted > 0);
        assert!(skipped > 0);

        let mut bucket_ok = 0_u64;
        let mut bucket_skip = 0_u64;
        for ug in 0..BUCKETS {
            for bg in 0..BUCKETS {
                if buckets_conflict(ug, bg, shared) {
                    bucket_skip += 1;
                    continue;
                }
                for sg in 0..BUCKETS {
                    if buckets_conflict(ug, sg, shared) || buckets_conflict(bg, sg, shared) {
                        bucket_skip += 1;
                        continue;
                    }
                    for bbg in 0..BUCKETS {
                        if buckets_conflict(ug, bbg, shared)
                            || buckets_conflict(bg, bbg, shared)
                            || buckets_conflict(sg, bbg, shared)
                        {
                            bucket_skip += 1;
                            continue;
                        }
                        let hands = [
                            shared.bucket_cards[ug],
                            shared.bucket_cards[bg],
                            shared.bucket_cards[sg],
                            shared.bucket_cards[bbg],
                        ];
                        assert!(
                            !cards_share(&hands),
                            "buckets {ug},{bg},{sg},{bbg} cards overlap"
                        );
                        bucket_ok += 1;
                    }
                }
            }
        }
        assert!(bucket_ok > 0, "no legal 4-way bucket matchups");
        assert!(bucket_skip > 0, "card removal never fired");
    }

    fn input(stacks: [f64; 4], iterations: usize, tolerance: f64) -> SolverInput {
        SolverInput {
            stacks: stacks.to_vec(),
            payouts: vec![0.4, 0.3, 0.2, 0.1],
            small_blind: 50.0,
            big_blind: 100.0,
            ante: 0.0,
            button_index: 0,
            max_iterations: iterations,
            tolerance,
            num_players: 4,
            verbose_convergence: false,
            profile: false,
            algorithm: Algorithm::Cfr4Max,
            rank_cache_strict: false,
        }
    }

    #[test]
    fn cfr_4max_equal_stacks_converges() {
        let cache = EquityCache::load(&find_file("equity_cache.bin").expect("equity_cache.bin"))
            .expect("equity cache");
        let output = solve(&input([1000.0; 4], 200, 0.005), &cache);
        assert!(
            output.converged,
            "expected convergence, iterations {}",
            output.iterations_used
        );
        assert!(
            output.iterations_used <= 200,
            "iterations {}",
            output.iterations_used
        );
        assert!(output.iterations_used >= MIN_CONVERGENCE_ITERS);
        let sum: f64 = output.equities.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.05,
            "equity sum {sum} {:?}",
            output.equities
        );
        let ranges = output.four_max.expect("4-max ranges");
        let aa = combo_index([Card::new(14, 0), Card::new(14, 1)]) as usize;
        let trash = combo_index([Card::new(7, 0), Card::new(2, 1)]) as usize;
        assert!(
            ranges.utg_push[aa] > ranges.utg_push[trash],
            "AA {:.3} should push more than 72o {:.3}",
            ranges.utg_push[aa],
            ranges.utg_push[trash]
        );
    }

    #[test]
    fn cfr_4max_unequal_stacks() {
        let cache = EquityCache::load(&find_file("equity_cache.bin").expect("equity_cache.bin"))
            .expect("equity cache");
        let output = solve(&input([2000.0, 500.0, 1000.0, 1500.0], 200, 0.01), &cache);
        assert!(output.iterations_used > 0);
        assert_eq!(output.equities.len(), 4);
        for eq in &output.equities {
            assert!(eq.is_finite() && *eq >= -1e-6 && *eq <= 1.0 + 1e-6);
        }
        let sum: f64 = output.equities.iter().sum();
        assert!((sum - 1.0).abs() < 0.05, "equity sum {sum}");
        let ranges = output.four_max.expect("4-max ranges");
        for freq in ranges.utg_push {
            assert!((0.0..=1.0).contains(&freq));
        }
    }
}
