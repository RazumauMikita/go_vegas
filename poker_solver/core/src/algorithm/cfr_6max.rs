use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

use rayon::prelude::*;

use crate::algorithm::cfr::regret_match;
use crate::algorithm::cfr_4max::{apply_permutation_4way, PERM4};
use crate::algorithm::{run_loop, SolverAlgorithm};
use crate::bucketing::{Bucketing, NUM_BUCKETS};
use crate::card::Card;
use crate::equity_3way::{apply_permutation, RANK_PERMUTATIONS as PERM3};
use crate::equity_cache::{expand_combo, EquityCache};
use crate::four_way_rank_cache::FourWayRankCache;
use crate::solver::{
    empty_output, six_node, tournament_equity, SixMaxRanges, SolverInput, SolverOutput,
    HAND_EV_SCALE, HAND_TYPES, SIX_MAX_NODES,
};
use crate::three_way_rank_cache::ThreeWayRankCache;

const DEFAULT_TOLERANCE: f64 = 0.005;
const MIN_CONVERGENCE_ITERS: usize = 200;
const DCFR_ALPHA: f64 = 1.5;
const DCFR_BETA: f64 = 0.5;
const DCFR_GAMMA: f64 = 1.0;
const BUCKETS: usize = NUM_BUCKETS;
const NODES: usize = SIX_MAX_NODES;
const LIVE: f64 = 1e-12;
const PLAYERS: usize = 6;

pub struct Cfr6Max {
    input: SolverInput,
    model: Model,
    regret: Vec<[[f64; 2]; HAND_TYPES]>,
    strategy: Vec<[[f64; 2]; HAND_TYPES]>,
    sum: Vec<[f64; HAND_TYPES]>,
    weight_sum: f64,
    prev: Vec<[f64; HAND_TYPES]>,
    prev_avg: Vec<[f64; HAND_TYPES]>,
    iterations_done: usize,
    has_converged: bool,
    solve_started: Instant,
}

struct Model {
    map: [usize; PLAYERS],
    walk: [[f64; PLAYERS]; 64],
    hus: Vec<HuTerm>,
    threes: Vec<ThreeTerm>,
    fours: Vec<FourTerm>,
    fives: Vec<FiveTerm>,
    six_icm: Vec<[f64; PLAYERS]>,
    four_cache: FourWayRankCache,
    bucket_eq: [[f64; BUCKETS]; BUCKETS],
}

struct HuTerm {
    mask: u32,
    a: usize,
    b: usize,
    icm_a: [f64; PLAYERS],
    icm_b: [f64; PLAYERS],
}

struct ThreeTerm {
    mask: u32,
    seats: [usize; 3],
    icm: [[f64; PLAYERS]; 6],
}

struct FourTerm {
    mask: u32,
    seats: [usize; 4],
    icm: [[f64; PLAYERS]; 24],
}

struct FiveTerm {
    mask: u32,
    seats: [usize; 5],
    folder: usize,
    icm: Vec<[f64; PLAYERS]>,
}

struct Terminals {
    by_mask: Vec<[[f64; HAND_TYPES]; PLAYERS]>,
}

struct Shared {
    equity: Vec<f32>,
    combos: [f64; HAND_TYPES],
    unblock: Vec<Vec<[u8; HAND_TYPES]>>,
    block: [[bool; HAND_TYPES]; HAND_TYPES],
    bucket: [u8; HAND_TYPES],
    bucket_mask: [u64; BUCKETS],
    three: ThreeWayRankCache,
}

pub fn apply_permutation_5way(perm: usize, contested: [f64; 5], uncalled: [f64; 5]) -> [f64; 5] {
    let order = perm5()[perm.min(119)];
    let mut score = [0_u8; 5];
    for (place, &player) in order.iter().enumerate() {
        score[player] = (4 - place) as u8;
    }

    let mut levels = contested;
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut unique = [0.0; 5];
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

    let mut won = [0.0; 5];
    let mut prev = 0.0;
    for level in unique.into_iter().take(n_unique) {
        let mut best = 0_u8;
        let mut n_in = 0_u32;
        let mut in_pot = [false; 5];
        for i in 0..5 {
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
        for i in 0..5 {
            if in_pot[i] && score[i] == best {
                won[i] += pot;
                break;
            }
        }
        prev = level;
    }
    for i in 0..5 {
        won[i] += uncalled[i];
    }
    won
}

fn perm5() -> &'static [[usize; 5]; 120] {
    static P: OnceLock<[[usize; 5]; 120]> = OnceLock::new();
    P.get_or_init(|| {
        let mut out = [[0usize; 5]; 120];
        let mut n = 0;
        let mut cur = [0, 1, 2, 3, 4];
        fill_perms(&mut cur, 0, &mut out, &mut n);
        out
    })
}

fn fill_perms(cur: &mut [usize; 5], k: usize, out: &mut [[usize; 5]; 120], n: &mut usize) {
    if k == 5 {
        out[*n] = *cur;
        *n += 1;
        return;
    }
    for i in k..5 {
        cur.swap(k, i);
        fill_perms(cur, k + 1, out, n);
        cur.swap(k, i);
    }
}

pub fn apply_permutation_6way(perm: usize, contested: [f64; 6], uncalled: [f64; 6]) -> [f64; 6] {
    let order = perm6()[perm.min(719)];
    let mut score = [0_u8; 6];
    for (place, &player) in order.iter().enumerate() {
        score[player] = (5 - place) as u8;
    }

    let mut levels = contested;
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut unique = [0.0; 6];
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

    let mut won = [0.0; 6];
    let mut prev = 0.0;
    for level in unique.into_iter().take(n_unique) {
        let mut best = 0_u8;
        let mut n_in = 0_u32;
        let mut in_pot = [false; 6];
        for i in 0..6 {
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
        for i in 0..6 {
            if in_pot[i] && score[i] == best {
                won[i] += pot;
                break;
            }
        }
        prev = level;
    }
    for i in 0..6 {
        won[i] += uncalled[i];
    }
    won
}

fn perm6() -> &'static [[usize; 6]; 720] {
    static P: OnceLock<[[usize; 6]; 720]> = OnceLock::new();
    P.get_or_init(|| {
        let mut out = [[0usize; 6]; 720];
        let mut n = 0;
        let mut cur = [0, 1, 2, 3, 4, 5];
        fill_perms6(&mut cur, 0, &mut out, &mut n);
        out
    })
}

fn fill_perms6(cur: &mut [usize; 6], k: usize, out: &mut [[usize; 6]; 720], n: &mut usize) {
    if k == 6 {
        out[*n] = *cur;
        *n += 1;
        return;
    }
    for i in k..6 {
        cur.swap(k, i);
        fill_perms6(cur, k + 1, out, n);
        cur.swap(k, i);
    }
}

fn parse_node(node: usize) -> (usize, u32) {
    match node {
        0 => (0, 0),
        1 | 2 => (1, (node - 1) as u32),
        3..=6 => (2, (node - 3) as u32),
        7..=14 => (3, (node - 7) as u32),
        15..=30 => (4, (node - 15) as u32),
        _ => (5, (node - 30) as u32),
    }
}

impl Cfr6Max {
    pub fn init(input: &SolverInput, cache: &EquityCache) -> Option<Self> {
        let solve_started = Instant::now();
        let model = Model::build(input, cache)?;
        Some(Self {
            input: input.clone(),
            model,
            regret: vec![[[0.0; 2]; HAND_TYPES]; NODES],
            strategy: vec![[[0.5, 0.5]; HAND_TYPES]; NODES],
            sum: vec![[0.0; HAND_TYPES]; NODES],
            weight_sum: 0.0,
            prev: vec![[0.5; HAND_TYPES]; NODES],
            prev_avg: vec![[0.5; HAND_TYPES]; NODES],
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
        for node in 0..NODES {
            for h in 0..HAND_TYPES {
                regret_match(&self.regret[node][h], &mut self.strategy[node][h]);
            }
        }
    }

    fn frequencies(&self) -> Vec<[f64; HAND_TYPES]> {
        let mut freq = vec![[0.0; HAND_TYPES]; NODES];
        for node in 0..NODES {
            for h in 0..HAND_TYPES {
                freq[node][h] = self.strategy[node][h][0];
            }
        }
        freq
    }

    fn accumulate(&mut self, freq: &[[f64; HAND_TYPES]], t: f64) {
        let tg = t.powf(DCFR_GAMMA);
        let beta = tg / (tg + 1.0);
        self.weight_sum = beta * self.weight_sum + 1.0;
        for node in 0..NODES {
            for h in 0..HAND_TYPES {
                self.sum[node][h] = beta * self.sum[node][h] + freq[node][h];
            }
        }
    }

    fn average_freq(&self) -> Vec<[f64; HAND_TYPES]> {
        if self.weight_sum <= 0.0 {
            return self.frequencies();
        }
        let mut freq = vec![[0.0; HAND_TYPES]; NODES];
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

impl SolverAlgorithm for Cfr6Max {
    fn iterate(&mut self) {
        let iter_started = Instant::now();
        self.recompute_strategies();
        let freq = self.frequencies();
        let opp = opponent_reach(&freq);
        let terminals = self.model.terminals(&freq);
        let t = (self.iterations_done + 1) as f64;

        for node in 0..NODES {
            for hand in 0..HAND_TYPES {
                let ev = action_ev(node, hand, &opp, &terminals);
                let sigma = self.strategy[node][hand];
                let reach = reach_of(node, hand, &opp);
                dcfr(&mut self.regret[node], hand, ev, sigma, reach, t);
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
        let mut evs = vec![[(0.0, 0.0); HAND_TYPES]; NODES];
        for node in 0..NODES {
            for hand in 0..HAND_TYPES {
                evs[node][hand] = action_ev(node, hand, &opp, &terminals);
            }
        }

        let shared = shared_cards();
        let mut equities = vec![0.0; PLAYERS];
        for canon in 0..PLAYERS {
            let mut num = 0.0;
            let mut den = 0.0;
            for hand in 0..HAND_TYPES {
                let w = shared.combos[hand];
                num += w * game_ev(canon, hand, &freq, &opp, &evs, &self.model);
                den += w;
            }
            let real = self.model.map[canon];
            equities[real] = if den > 0.0 { num / den } else { 1.0 / 6.0 };
        }

        let mut push_ranges = vec![[0.0; HAND_TYPES]; PLAYERS];
        let mut call_ranges = vec![[0.0; HAND_TYPES]; PLAYERS];
        let mut hand_evs = vec![[0.0; HAND_TYPES]; PLAYERS];
        let seats = self.model.map;
        push_ranges[seats[0]] = freq[six_node(0, 0)];
        push_ranges[seats[1]] = freq[six_node(1, 0)];
        push_ranges[seats[2]] = freq[six_node(2, 0)];
        push_ranges[seats[3]] = freq[six_node(3, 0)];
        push_ranges[seats[4]] = freq[six_node(4, 0)];
        call_ranges[seats[1]] = freq[six_node(1, 1)];
        call_ranges[seats[2]] = freq[six_node(2, 1)];
        call_ranges[seats[3]] = freq[six_node(3, 1)];
        call_ranges[seats[4]] = freq[six_node(4, 1)];
        call_ranges[seats[5]] = freq[six_node(5, 16)];
        for hand in 0..HAND_TYPES {
            hand_evs[seats[0]][hand] = diff(evs[six_node(0, 0)][hand]);
            hand_evs[seats[1]][hand] = diff(evs[six_node(1, 0)][hand]);
            hand_evs[seats[2]][hand] = diff(evs[six_node(2, 0)][hand]);
            hand_evs[seats[3]][hand] = diff(evs[six_node(3, 0)][hand]);
            hand_evs[seats[4]][hand] = diff(evs[six_node(4, 0)][hand]);
            hand_evs[seats[5]][hand] = diff(evs[six_node(5, 16)][hand]);
        }

        let last = self.frequencies();
        let elapsed = self.solve_started.elapsed().as_secs_f64();
        eprintln!(
            "6-max CFR: iterations={} converged={} time={elapsed:.2}s",
            self.iterations_done, self.has_converged
        );
        eprintln!(
            "6-max ranges last/avg: UTG {:.1}/{:.1} HJ {:.1}/{:.1} CO {:.1}/{:.1} BTN {:.1}/{:.1} SB {:.1}/{:.1} BBvsSB {:.1}/{:.1}",
            combo_share(&last[six_node(0, 0)]) * 100.0,
            combo_share(&freq[six_node(0, 0)]) * 100.0,
            combo_share(&last[six_node(1, 0)]) * 100.0,
            combo_share(&freq[six_node(1, 0)]) * 100.0,
            combo_share(&last[six_node(2, 0)]) * 100.0,
            combo_share(&freq[six_node(2, 0)]) * 100.0,
            combo_share(&last[six_node(3, 0)]) * 100.0,
            combo_share(&freq[six_node(3, 0)]) * 100.0,
            combo_share(&last[six_node(4, 0)]) * 100.0,
            combo_share(&freq[six_node(4, 0)]) * 100.0,
            combo_share(&last[six_node(5, 16)]) * 100.0,
            combo_share(&freq[six_node(5, 16)]) * 100.0,
        );

        let mut packed = [[0.0; HAND_TYPES]; NODES];
        for node in 0..NODES {
            packed[node] = freq[node];
        }

        SolverOutput {
            push_ranges,
            call_ranges,
            equities,
            iterations_used: self.iterations_done,
            converged: self.has_converged,
            three_max: None,
            hand_evs,
            three_max_hand_evs: None,
            four_max: None,
            five_max: None,
            six_max: Some(SixMaxRanges { freq: packed }),
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

fn opponent_reach(freq: &[[f64; HAND_TYPES]]) -> Vec<[f64; HAND_TYPES]> {
    let shared = shared_cards();
    let mut opp = vec![[0.0; HAND_TYPES]; NODES];
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

fn reach_of(node: usize, hand: usize, opp: &[[f64; HAND_TYPES]]) -> f64 {
    let (actor, mask) = parse_node(node);
    let mut p = 1.0;
    for a in 0..actor {
        let prior = mask & ((1 << a) - 1);
        let bit = (mask >> a) & 1;
        let q = opp[six_node(a, prior)][hand];
        p *= if bit == 1 { q } else { 1.0 - q };
    }
    p
}

fn action_ev(
    node: usize,
    h: usize,
    opp: &[[f64; HAND_TYPES]],
    t: &Terminals,
) -> (f64, f64) {
    let (actor, mask) = parse_node(node);
    let push = cont(actor + 1, mask | (1 << actor), actor, h, opp, t);
    let fold = cont(actor + 1, mask, actor, h, opp, t);
    (push, fold)
}

fn cont(
    next: usize,
    mask: u32,
    hero: usize,
    h: usize,
    opp: &[[f64; HAND_TYPES]],
    t: &Terminals,
) -> f64 {
    if next >= PLAYERS {
        return t.by_mask[mask as usize][hero][h];
    }
    if next == 5 && mask == 0 {
        return t.by_mask[0][hero][h];
    }
    let p = opp[six_node(next, mask)][h];
    let aggressive = cont(next + 1, mask | (1 << next), hero, h, opp, t);
    let passive = cont(next + 1, mask, hero, h, opp, t);
    p * aggressive + (1.0 - p) * passive
}

fn game_ev(
    seat: usize,
    h: usize,
    freq: &[[f64; HAND_TYPES]],
    opp: &[[f64; HAND_TYPES]],
    evs: &[[(f64, f64); HAND_TYPES]],
    model: &Model,
) -> f64 {
    let mix = |node: usize| {
        let (a, f) = evs[node][h];
        let p = freq[node][h];
        p * a + (1.0 - p) * f
    };
    match seat {
        0..=4 => {
            let mut ev = 0.0;
            let max_mask = 1u32 << seat;
            for mask in 0..max_mask {
                let mut p = 1.0;
                for a in 0..seat {
                    let prior = mask & ((1 << a) - 1);
                    let bit = (mask >> a) & 1;
                    let q = opp[six_node(a, prior)][h];
                    p *= if bit == 1 { q } else { 1.0 - q };
                }
                ev += p * mix(six_node(seat, mask));
            }
            ev
        }
        _ => {
            let mut ev = 0.0;
            for mask in 0..32u32 {
                let mut p = 1.0;
                for a in 0..5 {
                    let prior = mask & ((1 << a) - 1);
                    let bit = (mask >> a) & 1;
                    let q = opp[six_node(a, prior)][h];
                    p *= if bit == 1 { q } else { 1.0 - q };
                }
                if mask == 0 {
                    ev += p * model.walk[0][5];
                } else {
                    ev += p * mix(six_node(5, mask));
                }
            }
            ev
        }
    }
}

fn range_of<'a>(
    player: usize,
    final_mask: u32,
    freq: &'a [[f64; HAND_TYPES]],
) -> &'a [f64; HAND_TYPES] {
    let prior = final_mask & ((1 << player) - 1);
    &freq[six_node(player, prior)]
}

impl Model {
    fn build(input: &SolverInput, _cache: &EquityCache) -> Option<Self> {
        if input.stacks.len() != 6 || input.button_index >= 6 {
            return None;
        }
        if input.small_blind < 0.0 || input.big_blind < 0.0 || input.ante < 0.0 {
            return None;
        }
        if input.stacks.iter().any(|stack| *stack < 0.0) || input.payouts.is_empty() {
            return None;
        }

        let btn = input.button_index;
        let sb = (btn + 1) % 6;
        let bb = (btn + 2) % 6;
        let utg = (btn + 3) % 6;
        let hj = (btn + 4) % 6;
        let co = (btn + 5) % 6;
        let map = [utg, hj, co, btn, sb, bb];
        let base = [
            input.stacks[utg],
            input.stacks[hj],
            input.stacks[co],
            input.stacks[btn],
            input.stacks[sb],
            input.stacks[bb],
        ];
        let dead = [
            input.ante.min(base[0]).max(0.0),
            input.ante.min(base[1]).max(0.0),
            input.ante.min(base[2]).max(0.0),
            input.ante.min(base[3]).max(0.0),
            (input.small_blind + input.ante).min(base[4]).max(0.0),
            (input.big_blind + input.ante).min(base[5]).max(0.0),
        ];
        let payouts = &input.payouts;

        let mut walk = [[0.0; PLAYERS]; 64];
        walk[0] = icm6(&award(base, 5, &[0, 1, 2, 3, 4], dead), payouts);
        let mut hus = Vec::new();
        let mut threes = Vec::new();
        let mut fours = Vec::new();
        let mut fives = Vec::new();
        let mut six_icm = vec![[0.0; PLAYERS]; 720];

        for mask in 1u32..64 {
            let seats: Vec<usize> = (0..PLAYERS).filter(|&i| mask & (1 << i) != 0).collect();
            let folders: Vec<usize> = (0..PLAYERS).filter(|&i| mask & (1 << i) == 0).collect();
            match seats.len() {
                1 => {
                    walk[mask as usize] = icm6(&award(base, seats[0], &folders, dead), payouts);
                }
                2 => {
                    let a = seats[0];
                    let b = seats[1];
                    hus.push(HuTerm {
                        mask,
                        a,
                        b,
                        icm_a: icm6(&hu_end(base, a, b, &folders, dead, true), payouts),
                        icm_b: icm6(&hu_end(base, a, b, &folders, dead, false), payouts),
                    });
                }
                3 => {
                    let s = [seats[0], seats[1], seats[2]];
                    let mut icm = [[0.0; PLAYERS]; 6];
                    for perm in 0..6 {
                        icm[perm] = icm6(&three_end(base, dead, s, &folders, perm), payouts);
                    }
                    threes.push(ThreeTerm {
                        mask,
                        seats: s,
                        icm,
                    });
                }
                4 => {
                    let s = [seats[0], seats[1], seats[2], seats[3]];
                    let mut icm = [[0.0; PLAYERS]; 24];
                    for perm in 0..24 {
                        icm[perm] = icm6(&four_end(base, dead, s, &folders, perm), payouts);
                    }
                    fours.push(FourTerm {
                        mask,
                        seats: s,
                        icm,
                    });
                }
                5 => {
                    let s = [seats[0], seats[1], seats[2], seats[3], seats[4]];
                    let folder = folders[0];
                    let mut icm = vec![[0.0; PLAYERS]; 120];
                    for perm in 0..120 {
                        icm[perm] = icm6(&five_end(base, dead, s, folder, perm), payouts);
                    }
                    fives.push(FiveTerm {
                        mask,
                        seats: s,
                        folder,
                        icm,
                    });
                }
                6 => {
                    let (contested, uncalled) = effective6(base);
                    for perm in 0..720 {
                        six_icm[perm] =
                            icm6(&apply_permutation_6way(perm, contested, uncalled), payouts);
                    }
                }
                _ => {}
            }
        }

        cards()?;
        let four_cache = load_four_way()?;
        let bucket_eq = bucket_vs_bucket(shared_cards());

        Some(Self {
            map,
            walk,
            hus,
            threes,
            fours,
            fives,
            six_icm,
            four_cache,
            bucket_eq,
        })
    }

    fn terminals(&self, freq: &[[f64; HAND_TYPES]]) -> Terminals {
        let mut by_mask = vec![[[0.0; HAND_TYPES]; PLAYERS]; 64];
        for mask in 0u32..64 {
            if mask == 0 || mask.count_ones() == 1 {
                for seat in 0..PLAYERS {
                    by_mask[mask as usize][seat] = [self.walk[mask as usize][seat]; HAND_TYPES];
                }
            }
        }

        for term in &self.hus {
            let range_a = range_of(term.a, term.mask, freq);
            let range_b = range_of(term.b, term.mask, freq);
            fill_hu(
                &mut by_mask[term.mask as usize],
                term.a,
                term.b,
                range_a,
                range_b,
                term.icm_a,
                term.icm_b,
            );
        }

        let shared = shared_cards();
        for term in &self.threes {
            let ranges = [
                range_of(term.seats[0], term.mask, freq),
                range_of(term.seats[1], term.mask, freq),
                range_of(term.seats[2], term.mask, freq),
            ];
            fill_three(
                ranges,
                term.seats,
                &term.icm,
                (0..PLAYERS)
                    .filter(|&i| term.mask & (1 << i) == 0)
                    .collect::<Vec<_>>(),
                shared,
                &mut by_mask[term.mask as usize],
            );
        }

        for term in &self.fours {
            let ranges = [
                range_of(term.seats[0], term.mask, freq),
                range_of(term.seats[1], term.mask, freq),
                range_of(term.seats[2], term.mask, freq),
                range_of(term.seats[3], term.mask, freq),
            ];
            fill_four(
                ranges,
                term.seats,
                &term.icm,
                (0..PLAYERS)
                    .filter(|&i| term.mask & (1 << i) == 0)
                    .collect::<Vec<_>>(),
                &self.four_cache,
                shared,
                &mut by_mask[term.mask as usize],
            );
        }

        for term in &self.fives {
            let ranges = [
                range_of(term.seats[0], term.mask, freq),
                range_of(term.seats[1], term.mask, freq),
                range_of(term.seats[2], term.mask, freq),
                range_of(term.seats[3], term.mask, freq),
                range_of(term.seats[4], term.mask, freq),
            ];
            fill_five(
                ranges,
                term.seats,
                &term.icm,
                term.folder,
                &self.bucket_eq,
                shared,
                &mut by_mask[term.mask as usize],
            );
        }

        let six_ranges = [
            range_of(0, 63, freq),
            range_of(1, 63, freq),
            range_of(2, 63, freq),
            range_of(3, 63, freq),
            range_of(4, 63, freq),
            range_of(5, 63, freq),
        ];
        fill_six(
            six_ranges,
            &self.six_icm,
            &self.bucket_eq,
            shared,
            &mut by_mask[63],
        );

        Terminals { by_mask }
    }
}

fn fill_hu(
    out: &mut [[f64; HAND_TYPES]; PLAYERS],
    seat_a: usize,
    seat_b: usize,
    range_a: &[f64; HAND_TYPES],
    range_b: &[f64; HAND_TYPES],
    icm_a_wins: [f64; PLAYERS],
    icm_b_wins: [f64; PLAYERS],
) {
    let shared = shared_cards();
    for h in 0..HAND_TYPES {
        out[seat_a][h] = hu_vs(h, range_b, icm_a_wins[seat_a], icm_b_wins[seat_a], shared);
        out[seat_b][h] = hu_vs(h, range_a, icm_b_wins[seat_b], icm_a_wins[seat_b], shared);
    }
    let p_a = hu_uncond(range_a, range_b, shared);
    for seat in 0..PLAYERS {
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
            if shared.block[a][b] {
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

fn live_hands(range: &[f64; HAND_TYPES], cap: usize) -> Vec<usize> {
    let mut items: Vec<(usize, f64)> = (0..HAND_TYPES)
        .filter_map(|h| {
            let w = range[h].clamp(0.0, 1.0);
            (w > LIVE).then_some((h, w))
        })
        .collect();
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    items.into_iter().take(cap).map(|(h, _)| h).collect()
}

fn live_buckets(mass: &[f64; BUCKETS], cap: usize) -> Vec<usize> {
    let mut items: Vec<(usize, f64)> = (0..BUCKETS)
        .filter_map(|b| (mass[b] > LIVE).then_some((b, mass[b])))
        .collect();
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    items.into_iter().take(cap).map(|(b, _)| b).collect()
}

fn fill_three(
    ranges: [&[f64; HAND_TYPES]; 3],
    seats: [usize; 3],
    icm: &[[f64; PLAYERS]; 6],
    folders: Vec<usize>,
    shared: &Shared,
    out: &mut [[f64; HAND_TYPES]; PLAYERS],
) {
    let mut weight = [[0.0; HAND_TYPES]; 3];
    for slot in 0..3 {
        for h in 0..HAND_TYPES {
            weight[slot][h] = shared.combos[h] * ranges[slot][h].clamp(0.0, 1.0);
        }
    }
    let live = [
        live_hands(ranges[0], 40),
        live_hands(ranges[1], 40),
        live_hands(ranges[2], 40),
    ];
    let mut num = [[0.0; HAND_TYPES]; 3];
    let mut den = [[0.0; HAND_TYPES]; 3];
    let mut folder_num = [0.0; PLAYERS];
    let mut folder_den = 0.0;

    for slot in 0..3 {
        let (o1, o2) = match slot {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        if live[o1].is_empty() || live[o2].is_empty() {
            continue;
        }
        for h in 0..HAND_TYPES {
            for &h1 in &live[o1] {
                if shared.block[h][h1] {
                    continue;
                }
                let w1 = weight[o1][h1];
                for &h2 in &live[o2] {
                    if shared.block[h][h2] || shared.block[h1][h2] {
                        continue;
                    }
                    let w2 = weight[o2][h2];
                    let w = w1 * w2;
                    if w <= 0.0 {
                        continue;
                    }
                    let (a, b, c) = match slot {
                        0 => (h, h1, h2),
                        1 => (h1, h, h2),
                        _ => (h1, h2, h),
                    };
                    let probs = shared
                        .three
                        .get(a as u8, b as u8, c as u8)
                        .unwrap_or([1.0 / 6.0; 6]);
                    num[slot][h] += w * dot_icm(&probs, icm, seats[slot]);
                    den[slot][h] += w;
                }
            }
        }
    }

    if !live[0].is_empty() && !live[1].is_empty() && !live[2].is_empty() {
        for &h0 in &live[0] {
            let w0 = weight[0][h0];
            for &h1 in &live[1] {
                if shared.block[h0][h1] {
                    continue;
                }
                let w1 = weight[1][h1];
                for &h2 in &live[2] {
                    if shared.block[h0][h2] || shared.block[h1][h2] {
                        continue;
                    }
                    let w = w0 * w1 * weight[2][h2];
                    if w <= 0.0 {
                        continue;
                    }
                    let probs = shared
                        .three
                        .get(h0 as u8, h1 as u8, h2 as u8)
                        .unwrap_or([1.0 / 6.0; 6]);
                    for &folder in &folders {
                        folder_num[folder] += w * dot_icm(&probs, icm, folder);
                    }
                    folder_den += w;
                }
            }
        }
    }

    let fallback = mean_seat(icm);
    for slot in 0..3 {
        let seat = seats[slot];
        for h in 0..HAND_TYPES {
            out[seat][h] = if den[slot][h] > 0.0 {
                num[slot][h] / den[slot][h]
            } else {
                fallback[seat]
            };
        }
    }
    for folder in folders {
        let ev = if folder_den > 0.0 {
            folder_num[folder] / folder_den
        } else {
            fallback[folder]
        };
        out[folder] = [ev; HAND_TYPES];
    }
}

fn fill_four(
    ranges: [&[f64; HAND_TYPES]; 4],
    seats: [usize; 4],
    icm: &[[f64; PLAYERS]; 24],
    folders: Vec<usize>,
    cache: &FourWayRankCache,
    shared: &Shared,
    out: &mut [[f64; HAND_TYPES]; PLAYERS],
) {
    let mut mass = [[0.0; BUCKETS]; 4];
    for slot in 0..4 {
        for h in 0..HAND_TYPES {
            mass[slot][shared.bucket[h] as usize] +=
                shared.combos[h] * ranges[slot][h].clamp(0.0, 1.0);
        }
    }
    let live: [Vec<usize>; 4] = std::array::from_fn(|slot| live_buckets(&mass[slot], 12));

    let mut num = [[0.0; BUCKETS]; 4];
    let mut den = [[0.0; BUCKETS]; 4];
    let mut folder_num = [0.0; PLAYERS];
    let mut folder_den = 0.0;

    let hero_buckets: [Vec<usize>; 4] =
        std::array::from_fn(|_| (0..BUCKETS).filter(|&b| shared.bucket_mask[b] != 0).collect());

    for slot in 0..4 {
        let others: Vec<usize> = (0..4).filter(|&s| s != slot).collect();
        for &hb in &hero_buckets[slot] {
            for &b1 in &live[others[0]] {
                if buckets_conflict(hb, b1, shared) {
                    continue;
                }
                for &b2 in &live[others[1]] {
                    if buckets_conflict(hb, b2, shared) || buckets_conflict(b1, b2, shared) {
                        continue;
                    }
                    for &b3 in &live[others[2]] {
                        if buckets_conflict(hb, b3, shared)
                            || buckets_conflict(b1, b3, shared)
                            || buckets_conflict(b2, b3, shared)
                        {
                            continue;
                        }
                        let mut buckets = [0usize; 4];
                        buckets[slot] = hb;
                        buckets[others[0]] = b1;
                        buckets[others[1]] = b2;
                        buckets[others[2]] = b3;
                        let w = mass[others[0]][b1] * mass[others[1]][b2] * mass[others[2]][b3];
                        if w <= 0.0 {
                            continue;
                        }
                        let dist = cache.lookup(
                            buckets[0] as u8,
                            buckets[1] as u8,
                            buckets[2] as u8,
                            buckets[3] as u8,
                        );
                        let pay = dot24(&dist, icm, seats[slot]);
                        num[slot][hb] += w * pay;
                        den[slot][hb] += w;
                    }
                }
            }
        }
    }

    if live.iter().all(|v| !v.is_empty()) {
        for &b0 in &live[0] {
            for &b1 in &live[1] {
                if buckets_conflict(b0, b1, shared) {
                    continue;
                }
                for &b2 in &live[2] {
                    if buckets_conflict(b0, b2, shared) || buckets_conflict(b1, b2, shared) {
                        continue;
                    }
                    for &b3 in &live[3] {
                        if buckets_conflict(b0, b3, shared)
                            || buckets_conflict(b1, b3, shared)
                            || buckets_conflict(b2, b3, shared)
                        {
                            continue;
                        }
                        let w = mass[0][b0] * mass[1][b1] * mass[2][b2] * mass[3][b3];
                        if w <= 0.0 {
                            continue;
                        }
                        let dist = cache.lookup(b0 as u8, b1 as u8, b2 as u8, b3 as u8);
                        for &folder in &folders {
                            folder_num[folder] += w * dot24(&dist, icm, folder);
                        }
                        folder_den += w;
                    }
                }
            }
        }
    }

    let fallback = mean24(icm);
    for slot in 0..4 {
        let seat = seats[slot];
        let mut by_bucket = [fallback[seat]; BUCKETS];
        for b in 0..BUCKETS {
            if den[slot][b] > 0.0 {
                by_bucket[b] = num[slot][b] / den[slot][b];
            }
        }
        for h in 0..HAND_TYPES {
            out[seat][h] = by_bucket[shared.bucket[h] as usize];
        }
    }
    for folder in folders {
        let folder_ev = if folder_den > 0.0 {
            folder_num[folder] / folder_den
        } else {
            fallback[folder]
        };
        out[folder] = [folder_ev; HAND_TYPES];
    }
}

fn fill_five(
    ranges: [&[f64; HAND_TYPES]; 5],
    seats: [usize; 5],
    icm: &[[f64; PLAYERS]],
    folder: usize,
    bucket_eq: &[[f64; BUCKETS]; BUCKETS],
    shared: &Shared,
    out: &mut [[f64; HAND_TYPES]; PLAYERS],
) {
    let mut mass = [[0.0; BUCKETS]; 5];
    for seat in 0..5 {
        for h in 0..HAND_TYPES {
            mass[seat][shared.bucket[h] as usize] +=
                shared.combos[h] * ranges[seat][h].clamp(0.0, 1.0);
        }
    }
    let live: [Vec<usize>; 5] = std::array::from_fn(|seat| live_buckets(&mass[seat], 6));
    let hero_buckets: Vec<usize> = (0..BUCKETS)
        .filter(|&b| shared.bucket_mask[b] != 0)
        .collect();

    let mut num = [[0.0; BUCKETS]; 5];
    let mut den = [[0.0; BUCKETS]; 5];
    let mut folder_num = 0.0;
    let mut folder_den = 0.0;
    let orders = perm5();

    for hero in 0..5 {
        let others: Vec<usize> = (0..5).filter(|&s| s != hero).collect();
        for &hb in &hero_buckets {
            nested_four(&live, &others, |ob| {
                let mut buckets = [0usize; 5];
                buckets[hero] = hb;
                for i in 0..4 {
                    buckets[others[i]] = ob[i];
                    if buckets_conflict(hb, ob[i], shared) {
                        return;
                    }
                    for j in 0..i {
                        if buckets_conflict(ob[j], ob[i], shared) {
                            return;
                        }
                    }
                }
                let w = mass[others[0]][ob[0]]
                    * mass[others[1]][ob[1]]
                    * mass[others[2]][ob[2]]
                    * mass[others[3]][ob[3]];
                if w <= 0.0 {
                    return;
                }
                let strength = five_strength(buckets, bucket_eq);
                let pay = harville_pay(strength, orders, icm, seats[hero]);
                num[hero][hb] += w * pay;
                den[hero][hb] += w;
                if hero == 0 {
                    folder_num += w * harville_pay(strength, orders, icm, folder);
                    folder_den += w;
                }
            });
        }
    }

    let fallback = mean_icm(icm);
    for slot in 0..5 {
        let seat = seats[slot];
        let mut by_bucket = [fallback[seat]; BUCKETS];
        for b in 0..BUCKETS {
            if den[slot][b] > 0.0 {
                by_bucket[b] = num[slot][b] / den[slot][b];
            }
        }
        for h in 0..HAND_TYPES {
            out[seat][h] = by_bucket[shared.bucket[h] as usize];
        }
    }
    let folder_ev = if folder_den > 0.0 {
        folder_num / folder_den
    } else {
        fallback[folder]
    };
    out[folder] = [folder_ev; HAND_TYPES];
}

fn fill_six(
    ranges: [&[f64; HAND_TYPES]; 6],
    icm: &[[f64; PLAYERS]],
    bucket_eq: &[[f64; BUCKETS]; BUCKETS],
    shared: &Shared,
    out: &mut [[f64; HAND_TYPES]; PLAYERS],
) {
    let mut mass = [[0.0; BUCKETS]; 6];
    for seat in 0..6 {
        for h in 0..HAND_TYPES {
            mass[seat][shared.bucket[h] as usize] +=
                shared.combos[h] * ranges[seat][h].clamp(0.0, 1.0);
        }
    }
    let live: [Vec<usize>; 6] = std::array::from_fn(|seat| live_buckets(&mass[seat], 4));
    let hero_buckets: Vec<usize> = (0..BUCKETS)
        .filter(|&b| shared.bucket_mask[b] != 0)
        .collect();

    let mut num = [[0.0; BUCKETS]; 6];
    let mut den = [[0.0; BUCKETS]; 6];
    let orders = perm6();

    for hero in 0..6 {
        let others: Vec<usize> = (0..6).filter(|&s| s != hero).collect();
        for &hb in &hero_buckets {
            nested_five(&live, &others, |ob| {
                let mut buckets = [0usize; 6];
                buckets[hero] = hb;
                for i in 0..5 {
                    buckets[others[i]] = ob[i];
                    if buckets_conflict(hb, ob[i], shared) {
                        return;
                    }
                    for j in 0..i {
                        if buckets_conflict(ob[j], ob[i], shared) {
                            return;
                        }
                    }
                }
                let w = mass[others[0]][ob[0]]
                    * mass[others[1]][ob[1]]
                    * mass[others[2]][ob[2]]
                    * mass[others[3]][ob[3]]
                    * mass[others[4]][ob[4]];
                if w <= 0.0 {
                    return;
                }
                let strength = six_strength(buckets, bucket_eq);
                let pay = harville_pay6(strength, orders, icm, hero);
                num[hero][hb] += w * pay;
                den[hero][hb] += w;
            });
        }
    }

    let fallback = mean_icm(icm);
    for seat in 0..6 {
        let mut by_bucket = [fallback[seat]; BUCKETS];
        for b in 0..BUCKETS {
            if den[seat][b] > 0.0 {
                by_bucket[b] = num[seat][b] / den[seat][b];
            }
        }
        for h in 0..HAND_TYPES {
            out[seat][h] = by_bucket[shared.bucket[h] as usize];
        }
    }
}

fn nested_four(live: &[Vec<usize>; 5], others: &[usize], mut f: impl FnMut([usize; 4])) {
    if others.iter().any(|&s| live[s].is_empty()) {
        return;
    }
    for &b0 in &live[others[0]] {
        for &b1 in &live[others[1]] {
            for &b2 in &live[others[2]] {
                for &b3 in &live[others[3]] {
                    f([b0, b1, b2, b3]);
                }
            }
        }
    }
}

fn nested_five(live: &[Vec<usize>; 6], others: &[usize], mut f: impl FnMut([usize; 5])) {
    if others.iter().any(|&s| live[s].is_empty()) {
        return;
    }
    for &b0 in &live[others[0]] {
        for &b1 in &live[others[1]] {
            for &b2 in &live[others[2]] {
                for &b3 in &live[others[3]] {
                    for &b4 in &live[others[4]] {
                        f([b0, b1, b2, b3, b4]);
                    }
                }
            }
        }
    }
}

fn five_strength(buckets: [usize; 5], beq: &[[f64; BUCKETS]; BUCKETS]) -> [f64; 5] {
    let mut s = [0.0; 5];
    for i in 0..5 {
        let mut prod = 1.0;
        for j in 0..5 {
            if i != j {
                prod *= beq[buckets[i]][buckets[j]].max(0.02);
            }
        }
        s[i] = prod.powf(0.25);
    }
    s
}

fn six_strength(buckets: [usize; 6], beq: &[[f64; BUCKETS]; BUCKETS]) -> [f64; 6] {
    let mut s = [0.0; 6];
    for i in 0..6 {
        let mut prod = 1.0;
        for j in 0..6 {
            if i != j {
                prod *= beq[buckets[i]][buckets[j]].max(0.02);
            }
        }
        s[i] = prod.powf(0.2);
    }
    s
}

fn harville_pay(
    strength: [f64; 5],
    orders: &[[usize; 5]; 120],
    icm: &[[f64; PLAYERS]],
    seat: usize,
) -> f64 {
    let mut pay = 0.0;
    for (perm, order) in orders.iter().enumerate() {
        pay += harville(&strength, order) * icm[perm][seat];
    }
    pay
}

fn harville_pay6(
    strength: [f64; 6],
    orders: &[[usize; 6]; 720],
    icm: &[[f64; PLAYERS]],
    seat: usize,
) -> f64 {
    let mut pay = 0.0;
    for (perm, order) in orders.iter().enumerate() {
        pay += harville(&strength, order) * icm[perm][seat];
    }
    pay
}

fn harville(strength: &[f64], order: &[usize]) -> f64 {
    let mut remaining: f64 = strength.iter().sum();
    let mut p = 1.0;
    for &player in order {
        if remaining <= 1e-15 {
            return 0.0;
        }
        p *= strength[player] / remaining;
        remaining -= strength[player];
    }
    p
}

fn dot_icm(probs: &[f64; 6], icm: &[[f64; PLAYERS]; 6], seat: usize) -> f64 {
    let mut sum = 0.0;
    for perm in 0..6 {
        sum += probs[perm] * icm[perm][seat];
    }
    sum
}

fn dot24(probs: &[f64; 24], icm: &[[f64; PLAYERS]; 24], seat: usize) -> f64 {
    let mut sum = 0.0;
    for perm in 0..24 {
        sum += probs[perm] * icm[perm][seat];
    }
    sum
}

fn mean_seat(icm: &[[f64; PLAYERS]; 6]) -> [f64; PLAYERS] {
    let mut out = [0.0; PLAYERS];
    for perm in 0..6 {
        for seat in 0..PLAYERS {
            out[seat] += icm[perm][seat];
        }
    }
    for seat in out.iter_mut() {
        *seat /= 6.0;
    }
    out
}

fn mean24(icm: &[[f64; PLAYERS]; 24]) -> [f64; PLAYERS] {
    let mut out = [0.0; PLAYERS];
    for perm in 0..24 {
        for seat in 0..PLAYERS {
            out[seat] += icm[perm][seat];
        }
    }
    for seat in out.iter_mut() {
        *seat /= 24.0;
    }
    out
}

fn mean_icm(icm: &[[f64; PLAYERS]]) -> [f64; PLAYERS] {
    let mut out = [0.0; PLAYERS];
    if icm.is_empty() {
        return out;
    }
    for row in icm {
        for seat in 0..PLAYERS {
            out[seat] += row[seat];
        }
    }
    let n = icm.len() as f64;
    for seat in out.iter_mut() {
        *seat /= n;
    }
    out
}

fn buckets_conflict(a: usize, b: usize, shared: &Shared) -> bool {
    let left = shared.bucket_mask[a];
    let right = shared.bucket_mask[b];
    left == 0 || right == 0 || left & right != 0
}

fn icm6(stacks: &[f64; 6], payouts: &[f64]) -> [f64; 6] {
    let eq = tournament_equity(stacks, payouts);
    [eq[0], eq[1], eq[2], eq[3], eq[4], eq[5]]
}

fn award(base: [f64; 6], winner: usize, folders: &[usize], dead: [f64; 6]) -> [f64; 6] {
    let mut stacks = base;
    for &folder in folders {
        let take = dead[folder].min(stacks[folder]).max(0.0);
        stacks[folder] -= take;
        stacks[winner] += take;
    }
    stacks
}

fn hu_end(
    base: [f64; 6],
    a: usize,
    b: usize,
    folders: &[usize],
    dead: [f64; 6],
    a_wins: bool,
) -> [f64; 6] {
    let mut stacks = base;
    let mut pot = 0.0;
    for &folder in folders {
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
    base: [f64; 6],
    dead: [f64; 6],
    seats: [usize; 3],
    folders: &[usize],
    perm: usize,
) -> [f64; 6] {
    let active = [base[seats[0]], base[seats[1]], base[seats[2]]];
    let (contested, uncalled) = effective3(active);
    let won = apply_permutation(perm, contested, uncalled);
    let mut stacks = [0.0; 6];
    for slot in 0..3 {
        stacks[seats[slot]] = won[slot];
    }
    let best = PERM3[perm][0];
    for &folder in folders {
        let take = dead[folder].min(base[folder]).max(0.0);
        stacks[folder] = base[folder] - take;
        stacks[seats[best]] += take;
    }
    stacks
}

fn four_end(
    base: [f64; 6],
    dead: [f64; 6],
    seats: [usize; 4],
    folders: &[usize],
    perm: usize,
) -> [f64; 6] {
    let active = [
        base[seats[0]],
        base[seats[1]],
        base[seats[2]],
        base[seats[3]],
    ];
    let (contested, uncalled) = effective4(active);
    let won = apply_permutation_4way(perm, contested, uncalled);
    let mut stacks = [0.0; 6];
    for slot in 0..4 {
        stacks[seats[slot]] = won[slot];
    }
    let best = seats[PERM4[perm][0]];
    for &folder in folders {
        let take = dead[folder].min(base[folder]).max(0.0);
        stacks[folder] = base[folder] - take;
        stacks[best] += take;
    }
    stacks
}

fn five_end(
    base: [f64; 6],
    dead: [f64; 6],
    seats: [usize; 5],
    folder: usize,
    perm: usize,
) -> [f64; 6] {
    let active = [
        base[seats[0]],
        base[seats[1]],
        base[seats[2]],
        base[seats[3]],
        base[seats[4]],
    ];
    let (contested, uncalled) = effective5(active);
    let won = apply_permutation_5way(perm, contested, uncalled);
    let mut stacks = [0.0; 6];
    for slot in 0..5 {
        stacks[seats[slot]] = won[slot];
    }
    let take = dead[folder].min(base[folder]).max(0.0);
    stacks[folder] = base[folder] - take;
    stacks[seats[perm5()[perm][0]]] += take;
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

fn effective5(stacks: [f64; 5]) -> ([f64; 5], [f64; 5]) {
    let mut contested = [0.0; 5];
    let mut uncalled = [0.0; 5];
    for i in 0..5 {
        let mut max_other = 0.0_f64;
        for j in 0..5 {
            if i != j {
                max_other = max_other.max(stacks[j]);
            }
        }
        contested[i] = stacks[i].min(max_other).max(0.0);
        uncalled[i] = (stacks[i] - contested[i]).max(0.0);
    }
    (contested, uncalled)
}

fn effective6(stacks: [f64; 6]) -> ([f64; 6], [f64; 6]) {
    let mut contested = [0.0; 6];
    let mut uncalled = [0.0; 6];
    for i in 0..6 {
        let mut max_other = 0.0_f64;
        for j in 0..6 {
            if i != j {
                max_other = max_other.max(stacks[j]);
            }
        }
        contested[i] = stacks[i].min(max_other).max(0.0);
        uncalled[i] = (stacks[i] - contested[i]).max(0.0);
    }
    (contested, uncalled)
}

fn bucket_vs_bucket(shared: &Shared) -> [[f64; BUCKETS]; BUCKETS] {
    let mut num = [[0.0; BUCKETS]; BUCKETS];
    let mut den = [[0.0; BUCKETS]; BUCKETS];
    for a in 0..HAND_TYPES {
        for b in 0..HAND_TYPES {
            if shared.block[a][b] {
                continue;
            }
            let ba = shared.bucket[a] as usize;
            let bb = shared.bucket[b] as usize;
            let w = shared.combos[a] * shared.combos[b];
            num[ba][bb] += w * f64::from(shared.equity[a * HAND_TYPES + b]);
            den[ba][bb] += w;
        }
    }
    let mut out = [[0.5; BUCKETS]; BUCKETS];
    for a in 0..BUCKETS {
        for b in 0..BUCKETS {
            if den[a][b] > 0.0 {
                out[a][b] = num[a][b] / den[a][b];
            }
        }
    }
    out
}

fn cards() -> Option<&'static Shared> {
    static SHARED: OnceLock<Option<Shared>> = OnceLock::new();
    SHARED
        .get_or_init(|| match Shared::load() {
            Ok(shared) => Some(shared),
            Err(error) => {
                eprintln!("6-max caches: {error}");
                None
            }
        })
        .as_ref()
}

fn shared_cards() -> &'static Shared {
    cards().expect("6-max card tables")
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
        let mut combos = [0.0; HAND_TYPES];
        for hand in 0..HAND_TYPES {
            let list = expand_combo(hand as u8);
            combos[hand] = list.len() as f64;
            masks[hand] = card_mask(list[0]);
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
        let mut seen = [false; BUCKETS];
        for hand in 0..HAND_TYPES {
            let b = bucketing.bucket_of(hand as u8);
            bucket[hand] = b;
            let idx = b as usize;
            if !seen[idx] {
                seen[idx] = true;
                bucket_mask[idx] = masks[hand];
            }
        }

        Ok(Self {
            equity,
            combos,
            unblock,
            block,
            bucket,
            bucket_mask,
            three,
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
mod tests {
    use super::*;
    use crate::equity_cache::combo_index;
    use crate::solver::{solve, SolverInput};
    use crate::Algorithm;

    fn assert_stacks(got: [f64; 5], expected: [f64; 5]) {
        for i in 0..5 {
            assert!(
                (got[i] - expected[i]).abs() < 1e-6,
                "seat {i}: got {got:?} expected {expected:?}"
            );
        }
    }

    fn assert_stacks6(got: [f64; 6], expected: [f64; 6]) {
        for i in 0..6 {
            assert!(
                (got[i] - expected[i]).abs() < 1e-6,
                "seat {i}: got {got:?} expected {expected:?}"
            );
        }
    }

    #[test]
    fn six_node_roundtrip() {
        assert_eq!(six_node(0, 0), 0);
        assert_eq!(six_node(1, 0), 1);
        assert_eq!(six_node(1, 1), 2);
        assert_eq!(six_node(2, 0), 3);
        assert_eq!(six_node(3, 0), 7);
        assert_eq!(six_node(4, 0), 15);
        assert_eq!(six_node(4, 1), 16);
        assert_eq!(six_node(4, 15), 30);
        assert_eq!(six_node(5, 1), 31);
        assert_eq!(six_node(5, 16), 46);
        assert_eq!(six_node(5, 31), 61);
        for node in 0..NODES {
            let (actor, mask) = parse_node(node);
            assert_eq!(six_node(actor, mask), node, "node {node}");
        }
    }

    #[test]
    fn apply_permutation_5way_correct() {
        let uncalled = [0.0; 5];
        assert_stacks(
            apply_permutation_5way(0, [100.0; 5], uncalled),
            [500.0, 0.0, 0.0, 0.0, 0.0],
        );
        for perm in 0..120 {
            let stacks = apply_permutation_5way(perm, [100.0; 5], uncalled);
            assert!(
                (stacks.iter().sum::<f64>() - 500.0).abs() < 1e-6,
                "perm {perm} chips {stacks:?}"
            );
        }
    }

    #[test]
    fn apply_permutation_6way_correct() {
        let uncalled = [0.0; 6];
        assert_stacks6(
            apply_permutation_6way(0, [100.0; 6], uncalled),
            [600.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        for perm in 0..720 {
            let stacks = apply_permutation_6way(perm, [100.0; 6], uncalled);
            assert!(
                (stacks.iter().sum::<f64>() - 600.0).abs() < 1e-6,
                "perm {perm} chips {stacks:?}"
            );
        }
    }

    fn input(stacks: [f64; 6], iterations: usize, tolerance: f64) -> SolverInput {
        SolverInput {
            stacks: stacks.to_vec(),
            payouts: vec![0.5, 0.3, 0.2],
            small_blind: 50.0,
            big_blind: 100.0,
            ante: 0.0,
            button_index: 3,
            max_iterations: iterations,
            tolerance,
            num_players: 6,
            verbose_convergence: false,
            profile: false,
            algorithm: Algorithm::Cfr6Max,
            rank_cache_strict: false,
        }
    }

    #[test]
    fn cfr_6max_runs() {
        let cache = EquityCache::load(&find_file("equity_cache.bin").expect("equity_cache.bin"))
            .expect("equity cache");
        let output = solve(&input([1000.0; 6], 8, 0.05), &cache);
        assert_eq!(output.equities.len(), 6);
        assert!(output.six_max.is_some());
        let ranges = output.six_max.expect("6-max ranges");
        let aa = combo_index([Card::new(14, 0), Card::new(14, 1)]) as usize;
        let trash = combo_index([Card::new(7, 0), Card::new(2, 1)]) as usize;
        let utg = &ranges.freq[six_node(0, 0)];
        assert!(
            utg[aa] >= utg[trash],
            "AA {:.3} should push at least as much as 72o {:.3}",
            utg[aa],
            utg[trash]
        );
        let sum: f64 = output.equities.iter().sum();
        assert!((sum - 1.0).abs() < 0.15, "equity sum {sum}");
    }
}
