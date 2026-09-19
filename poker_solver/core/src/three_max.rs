use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;

use crate::card::Card;
use crate::equity_3way::equity_3way_icm_with_cache;
use crate::equity_cache::{combo_index, expand_combo, EquityCache};
use crate::icm::{icm_equity_call_count, reset_icm_equity_call_count, IcmCache};
use crate::solver::{
    best_response, build_unblocked, empty_output, tournament_equity, transfer, SolverInput,
    SolverOutput, ThreeMaxHandEvs, ThreeMaxRanges, HAND_EV_SCALE, HAND_TYPES,
};
use crate::three_way_cache::ThreeWayCache;

const THREE_WAY_SAMPLES: u32 = 64;
const THREE_WAY_BOARDS: u64 = 100;
const THREE_WAY_CACHE_FILE: &str = "equity_3way_cache.bin";

pub(crate) fn solve_3max(input: &SolverInput, cache: &EquityCache) -> SolverOutput {
    let solve_started = Instant::now();
    let three_way_path = Path::new(THREE_WAY_CACHE_FILE);
    let stacks_equal = input.stacks.len() == 3
        && input.stacks[0] == input.stacks[1]
        && input.stacks[1] == input.stacks[2];
    let three_way = if stacks_equal {
        ThreeWayCache::load(three_way_path).unwrap_or_else(|_| ThreeWayCache::new())
    } else {
        ThreeWayCache::new()
    };
    let Some(ctx) = ThreeMaxContext::new(input, three_way) else {
        return empty_output(input.stacks.len());
    };

    let mut btn_push = [1.0; HAND_TYPES];
    let mut sb_call_vs_btn = [1.0; HAND_TYPES];
    let mut bb_call_vs_btn = [1.0; HAND_TYPES];
    let mut bb_call_vs_both = [1.0; HAND_TYPES];
    let mut sb_push = [1.0; HAND_TYPES];
    let mut bb_call_vs_sb = [1.0; HAND_TYPES];

    if input.profile {
        let init_icm = icm_equity_call_count();
        eprintln!("icm_equity at init (via tournament_equity): {init_icm}");
        reset_icm_equity_call_count();
        profile_hot_functions(
            &btn_push,
            &sb_call_vs_btn,
            &bb_call_vs_btn,
            &bb_call_vs_both,
            &sb_push,
            &bb_call_vs_sb,
            &ctx,
            cache,
        );
    }

    let mut sum_btn_push = [0.0; HAND_TYPES];
    let mut sum_sb_call = [0.0; HAND_TYPES];
    let mut sum_bb_vs_btn = [0.0; HAND_TYPES];
    let mut sum_bb_vs_both = [0.0; HAND_TYPES];
    let mut sum_sb_push = [0.0; HAND_TYPES];
    let mut sum_bb_vs_sb = [0.0; HAND_TYPES];

    let mut converged = false;
    let mut iterations_used = 0;

    for iteration in 1..=input.max_iterations {
        iterations_used = iteration;
        let iter_started = Instant::now();
        let hits_before = ctx.cache_hits();
        let misses_before = ctx.cache_misses();

        let br = ctx.map_range_jobs(|kind, hand| match kind {
            0 => {
                let ev_push = ev_btn_push(
                    hand,
                    &sb_call_vs_btn,
                    &bb_call_vs_btn,
                    &bb_call_vs_both,
                    &ctx,
                    cache,
                );
                let ev_fold = ev_btn_fold(hand, &sb_push, &bb_call_vs_sb, &ctx, cache);
                best_response(ev_push, ev_fold, btn_push[hand])
            }
            1 => {
                let ev_call = ev_sb_call_vs_btn(hand, &btn_push, &bb_call_vs_both, &ctx, cache);
                let ev_fold = ev_sb_fold_vs_btn(hand, &btn_push, &bb_call_vs_btn, &ctx, cache);
                best_response(ev_call, ev_fold, sb_call_vs_btn[hand])
            }
            2 => {
                let ev_call = ev_bb_call_vs_btn(hand, &btn_push, &sb_call_vs_btn, &ctx, cache);
                best_response(
                    ev_call,
                    ctx.icm_btn_takes_blinds[ctx.bb],
                    bb_call_vs_btn[hand],
                )
            }
            3 => {
                let ev_call = ev_bb_call_vs_both(hand, &btn_push, &sb_call_vs_btn, &ctx, cache);
                let ev_fold = ev_bb_fold_vs_both(hand, &btn_push, &sb_call_vs_btn, &ctx, cache);
                best_response(ev_call, ev_fold, bb_call_vs_both[hand])
            }
            4 => {
                let ev_push = ev_sb_push_after_btn_fold(hand, &bb_call_vs_sb, &ctx, cache);
                best_response(ev_push, ctx.icm_bb_walks[ctx.sb], sb_push[hand])
            }
            _ => {
                let ev_call = ev_bb_call_vs_sb(hand, &sb_push, &ctx, cache);
                best_response(ev_call, ctx.icm_sb_walks[ctx.bb], bb_call_vs_sb[hand])
            }
        });
        let br_btn_push = &br[0];
        let br_sb_call = &br[1];
        let br_bb_vs_btn = &br[2];
        let br_bb_vs_both = &br[3];
        let br_sb_push = &br[4];
        let br_bb_vs_sb = &br[5];

        let t = iteration as f64;
        let mut change = 0.0;
        let mut push_change = 0.0;
        let mut call_change = 0.0;
        for hand in 0..HAND_TYPES {
            sum_btn_push[hand] += br_btn_push[hand];
            sum_sb_call[hand] += br_sb_call[hand];
            sum_bb_vs_btn[hand] += br_bb_vs_btn[hand];
            sum_bb_vs_both[hand] += br_bb_vs_both[hand];
            sum_sb_push[hand] += br_sb_push[hand];
            sum_bb_vs_sb[hand] += br_bb_vs_sb[hand];

            let next_btn = sum_btn_push[hand] / t;
            let next_sb_call = sum_sb_call[hand] / t;
            let next_bb_btn = sum_bb_vs_btn[hand] / t;
            let next_bb_both = sum_bb_vs_both[hand] / t;
            let next_sb_push = sum_sb_push[hand] / t;
            let next_bb_sb = sum_bb_vs_sb[hand] / t;

            change += (next_btn - btn_push[hand]).abs();
            change += (next_sb_call - sb_call_vs_btn[hand]).abs();
            change += (next_bb_btn - bb_call_vs_btn[hand]).abs();
            change += (next_bb_both - bb_call_vs_both[hand]).abs();
            change += (next_sb_push - sb_push[hand]).abs();
            change += (next_bb_sb - bb_call_vs_sb[hand]).abs();

            push_change += (next_btn - btn_push[hand]).abs();
            push_change += (next_sb_push - sb_push[hand]).abs();
            call_change += (next_sb_call - sb_call_vs_btn[hand]).abs();
            call_change += (next_bb_btn - bb_call_vs_btn[hand]).abs();
            call_change += (next_bb_both - bb_call_vs_both[hand]).abs();
            call_change += (next_bb_sb - bb_call_vs_sb[hand]).abs();

            btn_push[hand] = next_btn;
            sb_call_vs_btn[hand] = next_sb_call;
            bb_call_vs_btn[hand] = next_bb_btn;
            bb_call_vs_both[hand] = next_bb_both;
            sb_push[hand] = next_sb_push;
            bb_call_vs_sb[hand] = next_bb_sb;
        }

        let mean_change = change / (6.0 * HAND_TYPES as f64);
        let hits = ctx.cache_hits() - hits_before;
        let misses = ctx.cache_misses() - misses_before;
        let looked = hits + misses;
        let hit_pct = if looked == 0 {
            100.0
        } else {
            100.0 * hits as f64 / looked as f64
        };
        if input.verbose_convergence {
            let last_ten_start = input.max_iterations.saturating_sub(9);
            if iteration <= 50 || iteration >= last_ten_start {
                eprintln!(
                    "Iter {iteration}: push_change={push_change:.6} call_change={call_change:.6}"
                );
            }
        } else {
            eprintln!(
                "Iter {iteration}: {:.2}s, hits: {hit_pct:.1}%",
                iter_started.elapsed().as_secs_f64()
            );
        }
        if change < input.tolerance || mean_change < input.tolerance {
            converged = true;
            break;
        }
    }

    let mut push_ranges = vec![[0.0; HAND_TYPES]; 3];
    let mut call_ranges = vec![[0.0; HAND_TYPES]; 3];
    push_ranges[ctx.btn] = btn_push;
    push_ranges[ctx.sb] = sb_push;
    call_ranges[ctx.sb] = sb_call_vs_btn;
    call_ranges[ctx.bb] = bb_call_vs_btn;

    let mut equities = vec![0.0; 3];
    equities[ctx.btn] = average_btn_equity(
        &btn_push,
        &sb_call_vs_btn,
        &bb_call_vs_btn,
        &bb_call_vs_both,
        &sb_push,
        &bb_call_vs_sb,
        &ctx,
        cache,
    );
    equities[ctx.sb] = average_sb_equity(
        &btn_push,
        &sb_call_vs_btn,
        &bb_call_vs_btn,
        &bb_call_vs_both,
        &sb_push,
        &bb_call_vs_sb,
        &ctx,
        cache,
    );
    equities[ctx.bb] = average_bb_equity(
        &btn_push,
        &sb_call_vs_btn,
        &bb_call_vs_btn,
        &bb_call_vs_both,
        &sb_push,
        &bb_call_vs_sb,
        &ctx,
        cache,
    );

    let three_max_hand_evs = compute_three_max_hand_evs(
        &btn_push,
        &sb_call_vs_btn,
        &bb_call_vs_btn,
        &bb_call_vs_both,
        &sb_push,
        &bb_call_vs_sb,
        &ctx,
        cache,
    );

    let mut hand_evs = vec![[0.0; HAND_TYPES]; 3];
    hand_evs[ctx.btn] = three_max_hand_evs.btn_push;
    hand_evs[ctx.sb] = three_max_hand_evs.sb_push;
    hand_evs[ctx.bb] = three_max_hand_evs.bb_call_vs_btn;

    let hits = ctx.cache_hits();
    let misses = ctx.cache_misses();
    let looked = hits + misses;
    let hit_pct = if looked == 0 {
        100.0
    } else {
        100.0 * hits as f64 / looked as f64
    };
    eprintln!("Cache 3-way: {hits} hits / {misses} misses ({hit_pct:.1}% hit rate)");
    let icm_hits = ctx.icm_cache.hits();
    let icm_misses = ctx.icm_cache.misses();
    let icm_looked = icm_hits + icm_misses;
    let icm_hit_pct = if icm_looked == 0 {
        100.0
    } else {
        100.0 * icm_hits as f64 / icm_looked as f64
    };
    eprintln!("ICM cache: {icm_hits} hits / {icm_misses} misses ({icm_hit_pct:.1}% hit rate)");
    eprintln!("Solve time: {:.2}s", solve_started.elapsed().as_secs_f64());
    if misses > 0 && ctx.stacks_equal {
        if let Err(error) = ctx.three_way.save(three_way_path) {
            eprintln!("failed to save 3-way cache: {error}");
        }
    }

    SolverOutput {
        push_ranges,
        call_ranges,
        equities,
        iterations_used,
        converged,
        three_max: Some(ThreeMaxRanges {
            btn_push,
            sb_call_vs_btn,
            bb_call_vs_btn,
            bb_call_vs_btn_and_sb: bb_call_vs_both,
            sb_push,
            bb_call_vs_sb,
        }),
        hand_evs,
        three_max_hand_evs: Some(three_max_hand_evs),
    }
}

fn compute_three_max_hand_evs(
    btn_push: &[f64; HAND_TYPES],
    sb_call_vs_btn: &[f64; HAND_TYPES],
    bb_call_vs_btn: &[f64; HAND_TYPES],
    bb_call_vs_both: &[f64; HAND_TYPES],
    sb_push: &[f64; HAND_TYPES],
    bb_call_vs_sb: &[f64; HAND_TYPES],
    ctx: &ThreeMaxContext,
    cache: &EquityCache,
) -> ThreeMaxHandEvs {
    let mut btn_push_evs = [0.0; HAND_TYPES];
    let mut sb_call_evs = [0.0; HAND_TYPES];
    let mut bb_call_btn_evs = [0.0; HAND_TYPES];
    let mut bb_call_both_evs = [0.0; HAND_TYPES];
    let mut sb_push_evs = [0.0; HAND_TYPES];
    let mut bb_call_sb_evs = [0.0; HAND_TYPES];

    for hand in 0..HAND_TYPES {
        let ev_push = ev_btn_push(
            hand,
            sb_call_vs_btn,
            bb_call_vs_btn,
            bb_call_vs_both,
            ctx,
            cache,
        );
        let ev_fold = ev_btn_fold(hand, sb_push, bb_call_vs_sb, ctx, cache);
        btn_push_evs[hand] = (ev_push - ev_fold) * HAND_EV_SCALE;

        let ev_call = ev_sb_call_vs_btn(hand, btn_push, bb_call_vs_both, ctx, cache);
        let ev_sb_fold = ev_sb_fold_vs_btn(hand, btn_push, bb_call_vs_btn, ctx, cache);
        sb_call_evs[hand] = (ev_call - ev_sb_fold) * HAND_EV_SCALE;

        let ev_bb_call = ev_bb_call_vs_btn(hand, btn_push, sb_call_vs_btn, ctx, cache);
        let ev_bb_fold = ctx.icm_btn_takes_blinds[ctx.bb];
        bb_call_btn_evs[hand] = (ev_bb_call - ev_bb_fold) * HAND_EV_SCALE;

        let ev_bb_call_both = ev_bb_call_vs_both(hand, btn_push, sb_call_vs_btn, ctx, cache);
        let ev_bb_fold_both = ev_bb_fold_vs_both(hand, btn_push, sb_call_vs_btn, ctx, cache);
        bb_call_both_evs[hand] = (ev_bb_call_both - ev_bb_fold_both) * HAND_EV_SCALE;

        let ev_sb_push = ev_sb_push_after_btn_fold(hand, bb_call_vs_sb, ctx, cache);
        let ev_sb_push_fold = ctx.icm_bb_walks[ctx.sb];
        sb_push_evs[hand] = (ev_sb_push - ev_sb_push_fold) * HAND_EV_SCALE;

        let ev_bb_call_sb = ev_bb_call_vs_sb(hand, sb_push, ctx, cache);
        let ev_bb_call_sb_fold = ctx.icm_sb_walks[ctx.bb];
        bb_call_sb_evs[hand] = (ev_bb_call_sb - ev_bb_call_sb_fold) * HAND_EV_SCALE;
    }

    ThreeMaxHandEvs {
        btn_push: btn_push_evs,
        sb_call_vs_btn: sb_call_evs,
        bb_call_vs_btn: bb_call_btn_evs,
        bb_call_vs_btn_and_sb: bb_call_both_evs,
        sb_push: sb_push_evs,
        bb_call_vs_sb: bb_call_sb_evs,
    }
}

struct ThreeMaxContext {
    btn: usize,
    sb: usize,
    bb: usize,
    payouts: Vec<f64>,
    combos: Vec<Vec<[Card; 2]>>,
    unblocked: Vec<Vec<[u8; HAND_TYPES]>>,
    icm_btn_takes_blinds: Vec<f64>,
    icm_sb_walks: Vec<f64>,
    icm_bb_walks: Vec<f64>,
    icm_hu_btn_bb_sb_fold_btn_win: Vec<f64>,
    icm_hu_btn_bb_sb_fold_bb_win: Vec<f64>,
    icm_hu_btn_sb_bb_fold_btn_win: Vec<f64>,
    icm_hu_btn_sb_bb_fold_sb_win: Vec<f64>,
    icm_hu_sb_bb_btn_fold_sb_win: Vec<f64>,
    icm_hu_sb_bb_btn_fold_bb_win: Vec<f64>,
    all_in_stacks: [f64; 3],
    three_way_uncalled: [f64; 3],
    stacks_equal: bool,
    three_way: ThreeWayCache,
    icm_cache: IcmCache,
    use_icm_cache: bool,
    parallel_hands: bool,
}

impl ThreeMaxContext {
    fn new(input: &SolverInput, three_way: ThreeWayCache) -> Option<Self> {
        if input.stacks.len() != 3 {
            return None;
        }
        if input.button_index >= 3 {
            return None;
        }
        if input.small_blind < 0.0 || input.big_blind < 0.0 || input.ante < 0.0 {
            return None;
        }
        if input.stacks.iter().any(|&stack| stack < 0.0) {
            return None;
        }

        let btn = input.button_index;
        let sb = (btn + 1) % 3;
        let bb = (btn + 2) % 3;
        let payouts = input.payouts.clone();
        let combos: Vec<Vec<[Card; 2]>> = (0..HAND_TYPES as u8).map(expand_combo).collect();
        let unblocked = build_unblocked(&combos);

        let sb_dead = input.small_blind + input.ante;
        let bb_dead = input.big_blind + input.ante;
        let btn_dead = input.ante;
        let stacks = &input.stacks;

        let icm_btn_takes_blinds = tournament_equity(
            &collect_dead(stacks, &[(sb, sb_dead), (bb, bb_dead)], btn),
            &payouts,
        );
        let icm_sb_walks = tournament_equity(
            &collect_dead(stacks, &[(btn, btn_dead), (bb, bb_dead)], sb),
            &payouts,
        );
        let icm_bb_walks = tournament_equity(
            &collect_dead(stacks, &[(btn, btn_dead), (sb, sb_dead)], bb),
            &payouts,
        );

        let icm_hu_btn_bb_sb_fold_btn_win =
            tournament_equity(&hu_with_dead(stacks, sb, sb_dead, btn, bb, btn), &payouts);
        let icm_hu_btn_bb_sb_fold_bb_win =
            tournament_equity(&hu_with_dead(stacks, sb, sb_dead, btn, bb, bb), &payouts);
        let icm_hu_btn_sb_bb_fold_btn_win =
            tournament_equity(&hu_with_dead(stacks, bb, bb_dead, btn, sb, btn), &payouts);
        let icm_hu_btn_sb_bb_fold_sb_win =
            tournament_equity(&hu_with_dead(stacks, bb, bb_dead, btn, sb, sb), &payouts);
        let icm_hu_sb_bb_btn_fold_sb_win =
            tournament_equity(&hu_with_dead(stacks, btn, btn_dead, sb, bb, sb), &payouts);
        let icm_hu_sb_bb_btn_fold_bb_win =
            tournament_equity(&hu_with_dead(stacks, btn, btn_dead, sb, bb, bb), &payouts);

        let (all_in_stacks, three_way_uncalled) = three_way_effective_stacks(stacks, btn, sb, bb);
        let stacks_equal = stacks[btn] == stacks[sb] && stacks[sb] == stacks[bb];
        let use_icm_cache = std::env::var_os("POKER_NO_ICM_CACHE").is_none();
        let parallel_hands = std::env::var_os("POKER_3MAX_SEQUENTIAL").is_none();
        let mut icm_cache = IcmCache::new();
        if use_icm_cache {
            crate::equity_3way::prefill_icm_cache(
                &mut icm_cache,
                all_in_stacks,
                three_way_uncalled,
                &payouts,
            );
        }

        Some(Self {
            btn,
            sb,
            bb,
            payouts,
            combos,
            unblocked,
            icm_btn_takes_blinds,
            icm_sb_walks,
            icm_bb_walks,
            icm_hu_btn_bb_sb_fold_btn_win,
            icm_hu_btn_bb_sb_fold_bb_win,
            icm_hu_btn_sb_bb_fold_btn_win,
            icm_hu_btn_sb_bb_fold_sb_win,
            icm_hu_sb_bb_btn_fold_sb_win,
            icm_hu_sb_bb_btn_fold_bb_win,
            all_in_stacks,
            three_way_uncalled,
            stacks_equal,
            three_way,
            icm_cache,
            use_icm_cache,
            parallel_hands,
        })
    }

    fn icm_cache_ref(&self) -> Option<&IcmCache> {
        self.use_icm_cache.then_some(&self.icm_cache)
    }

    fn map_range_jobs<F>(&self, f: F) -> [Vec<f64>; 6]
    where
        F: Fn(usize, usize) -> f64 + Sync + Send,
    {
        let n = HAND_TYPES * 6;
        let raw: Vec<f64> = if self.parallel_hands {
            (0..n)
                .into_par_iter()
                .map(|job| f(job / HAND_TYPES, job % HAND_TYPES))
                .collect()
        } else {
            (0..n)
                .map(|job| f(job / HAND_TYPES, job % HAND_TYPES))
                .collect()
        };
        let mut out = std::array::from_fn(|_| vec![0.0; HAND_TYPES]);
        for kind in 0..6 {
            let start = kind * HAND_TYPES;
            out[kind].copy_from_slice(&raw[start..start + HAND_TYPES]);
        }
        out
    }

    fn cache_hits(&self) -> usize {
        self.three_way.hit_count()
    }

    fn cache_misses(&self) -> usize {
        self.three_way.miss_count()
    }
}

#[allow(clippy::too_many_arguments)]
fn profile_hot_functions(
    btn_push: &[f64; HAND_TYPES],
    sb_call_vs_btn: &[f64; HAND_TYPES],
    bb_call_vs_btn: &[f64; HAND_TYPES],
    bb_call_vs_both: &[f64; HAND_TYPES],
    sb_push: &[f64; HAND_TYPES],
    bb_call_vs_sb: &[f64; HAND_TYPES],
    ctx: &ThreeMaxContext,
    cache: &EquityCache,
) {
    eprintln!("--- profile (1 pass, all {HAND_TYPES} hands, sequential) ---");

    let t0 = Instant::now();
    for hand in 0..HAND_TYPES {
        let _ = ev_btn_push(
            hand,
            sb_call_vs_btn,
            bb_call_vs_btn,
            bb_call_vs_both,
            ctx,
            cache,
        );
    }
    eprintln!("ev_btn_push: {:?}", t0.elapsed());

    let t0 = Instant::now();
    for hand in 0..HAND_TYPES {
        let _ = ev_sb_call_vs_btn(hand, btn_push, bb_call_vs_both, ctx, cache);
    }
    eprintln!("ev_sb_call: {:?}", t0.elapsed());

    let t0 = Instant::now();
    for hand in 0..HAND_TYPES {
        let _ = ev_bb_call_vs_btn(hand, btn_push, sb_call_vs_btn, ctx, cache);
    }
    eprintln!("ev_bb_call_vs_btn: {:?}", t0.elapsed());

    let t0 = Instant::now();
    for hand in 0..HAND_TYPES {
        let _ = ev_bb_call_vs_both(hand, btn_push, sb_call_vs_btn, ctx, cache);
    }
    eprintln!("ev_bb_call_vs_both: {:?}", t0.elapsed());

    let t0 = Instant::now();
    for hand in 0..HAND_TYPES {
        let _ = ev_sb_push_after_btn_fold(hand, bb_call_vs_sb, ctx, cache);
    }
    eprintln!("ev_sb_push: {:?}", t0.elapsed());

    let t0 = Instant::now();
    for hand in 0..HAND_TYPES {
        let _ = ev_bb_call_vs_sb(hand, sb_push, ctx, cache);
    }
    eprintln!("ev_bb_call_vs_sb: {:?}", t0.elapsed());

    let icm_calls = icm_equity_call_count();
    eprintln!("icm_equity calls this pass: {icm_calls}");
    let icm_hits = ctx.icm_cache.hits();
    let icm_misses = ctx.icm_cache.misses();
    let icm_looked = icm_hits + icm_misses;
    let icm_hit_pct = if icm_looked == 0 {
        100.0
    } else {
        100.0 * icm_hits as f64 / icm_looked as f64
    };
    eprintln!(
        "ICM cache this pass: {icm_hits} hits / {icm_misses} misses ({icm_hit_pct:.1}% hit rate)"
    );
    eprintln!("--- end profile ---");
}

/// Эффективные олл-ин стеки (btn, sb, bb) и uncalled-возвраты для 3-way showdown.
fn three_way_effective_stacks(
    stacks: &[f64],
    btn: usize,
    sb: usize,
    bb: usize,
) -> ([f64; 3], [f64; 3]) {
    let seat = [stacks[btn], stacks[sb], stacks[bb]];
    let max_other = [
        seat[1].max(seat[2]),
        seat[0].max(seat[2]),
        seat[0].max(seat[1]),
    ];
    let contested = [
        seat[0].min(max_other[0]),
        seat[1].min(max_other[1]),
        seat[2].min(max_other[2]),
    ];
    let uncalled = [
        seat[0] - contested[0],
        seat[1] - contested[1],
        seat[2] - contested[2],
    ];
    (contested, uncalled)
}

fn collect_dead(stacks: &[f64], from: &[(usize, f64)], winner: usize) -> Vec<f64> {
    let mut next = stacks.to_vec();
    for &(loser, amount) in from {
        next = transfer(&next, loser, winner, amount);
    }
    next
}

fn hu_with_dead(
    stacks: &[f64],
    folder: usize,
    folder_dead: f64,
    hero: usize,
    villain: usize,
    winner: usize,
) -> Vec<f64> {
    let mut next = stacks.to_vec();
    let dead = folder_dead.max(0.0).min(next[folder]);
    next[folder] -= dead;
    let loser = if winner == hero { villain } else { hero };
    let contested = next[hero].min(next[villain]).max(0.0);
    next[loser] -= contested;
    next[winner] += contested + dead;
    next
}

fn ev_btn_push(
    hand_idx: usize,
    sb_call: &[f64; HAND_TYPES],
    bb_call_vs_btn: &[f64; HAND_TYPES],
    bb_call_vs_both: &[f64; HAND_TYPES],
    ctx: &ThreeMaxContext,
    cache: &EquityCache,
) -> f64 {
    average_over_combos(hand_idx, ctx, |combo_idx| {
        ev_btn_push_combo(
            hand_idx,
            combo_idx,
            sb_call,
            bb_call_vs_btn,
            bb_call_vs_both,
            ctx,
            cache,
        )
    })
}

fn ev_btn_push_combo(
    hand_idx: usize,
    combo_idx: usize,
    sb_call: &[f64; HAND_TYPES],
    bb_call_vs_btn: &[f64; HAND_TYPES],
    bb_call_vs_both: &[f64; HAND_TYPES],
    ctx: &ThreeMaxContext,
    cache: &EquityCache,
) -> f64 {
    let hero = ctx.combos[hand_idx][combo_idx];
    let unblocked = &ctx.unblocked[hand_idx][combo_idx];
    let p_sb_call = p_action(unblocked, sb_call);
    let p_sb_fold = 1.0 - p_sb_call;
    let p_bb_call_alone = p_action(unblocked, bb_call_vs_btn);
    let p_bb_fold_alone = 1.0 - p_bb_call_alone;

    let hu_vs_bb = hu_ev(
        hand_idx,
        unblocked,
        bb_call_vs_btn,
        cache,
        ctx.icm_hu_btn_bb_sb_fold_btn_win[ctx.btn],
        ctx.icm_hu_btn_bb_sb_fold_bb_win[ctx.btn],
    );
    let hu_vs_sb = hu_ev(
        hand_idx,
        unblocked,
        sb_call,
        cache,
        ctx.icm_hu_btn_sb_bb_fold_btn_win[ctx.btn],
        ctx.icm_hu_btn_sb_bb_fold_sb_win[ctx.btn],
    );
    let p_bb_call_both = p_action(unblocked, bb_call_vs_both);
    let ev_3way = three_way_ev(
        hero,
        ctx.btn,
        sb_call,
        bb_call_vs_both,
        ctx,
        hand_idx as u64 ^ ((combo_idx as u64) << 8),
    );

    p_sb_fold * p_bb_fold_alone * ctx.icm_btn_takes_blinds[ctx.btn]
        + p_sb_fold * p_bb_call_alone * hu_vs_bb
        + p_sb_call * (1.0 - p_bb_call_both) * hu_vs_sb
        + p_sb_call * p_bb_call_both * ev_3way
}

fn ev_btn_fold(
    hand_idx: usize,
    sb_push: &[f64; HAND_TYPES],
    bb_call_vs_sb: &[f64; HAND_TYPES],
    ctx: &ThreeMaxContext,
    cache: &EquityCache,
) -> f64 {
    average_over_combos(hand_idx, ctx, |combo_idx| {
        let unblocked = &ctx.unblocked[hand_idx][combo_idx];
        let p_sb_push = p_action(unblocked, sb_push);
        let p_sb_fold = 1.0 - p_sb_push;
        let p_bb_call = p_action(unblocked, bb_call_vs_sb);
        let hu = hu_ev_spectator(
            unblocked,
            sb_push,
            bb_call_vs_sb,
            cache,
            ctx.icm_hu_sb_bb_btn_fold_sb_win[ctx.btn],
            ctx.icm_hu_sb_bb_btn_fold_bb_win[ctx.btn],
        );
        p_sb_fold * ctx.icm_bb_walks[ctx.btn]
            + p_sb_push * (1.0 - p_bb_call) * ctx.icm_sb_walks[ctx.btn]
            + p_sb_push * p_bb_call * hu
    })
}

fn ev_sb_call_vs_btn(
    hand_idx: usize,
    btn_push: &[f64; HAND_TYPES],
    bb_call_vs_both: &[f64; HAND_TYPES],
    ctx: &ThreeMaxContext,
    cache: &EquityCache,
) -> f64 {
    average_over_combos(hand_idx, ctx, |combo_idx| {
        let hero = ctx.combos[hand_idx][combo_idx];
        let unblocked = &ctx.unblocked[hand_idx][combo_idx];
        let p_bb_call = p_action(unblocked, bb_call_vs_both);
        let hu = hu_ev(
            hand_idx,
            unblocked,
            btn_push,
            cache,
            ctx.icm_hu_btn_sb_bb_fold_sb_win[ctx.sb],
            ctx.icm_hu_btn_sb_bb_fold_btn_win[ctx.sb],
        );
        let ev_3way = three_way_ev(
            hero,
            ctx.sb,
            btn_push,
            bb_call_vs_both,
            ctx,
            0xA5 + ((hand_idx as u64 + combo_idx as u64) << 9),
        );
        (1.0 - p_bb_call) * hu + p_bb_call * ev_3way
    })
}

fn ev_sb_fold_vs_btn(
    hand_idx: usize,
    btn_push: &[f64; HAND_TYPES],
    bb_call_vs_btn: &[f64; HAND_TYPES],
    ctx: &ThreeMaxContext,
    cache: &EquityCache,
) -> f64 {
    average_over_combos(hand_idx, ctx, |combo_idx| {
        let unblocked = &ctx.unblocked[hand_idx][combo_idx];
        let p_bb_call = p_action(unblocked, bb_call_vs_btn);
        let hu = hu_ev_spectator(
            unblocked,
            btn_push,
            bb_call_vs_btn,
            cache,
            ctx.icm_hu_btn_bb_sb_fold_btn_win[ctx.sb],
            ctx.icm_hu_btn_bb_sb_fold_bb_win[ctx.sb],
        );
        p_bb_call * hu + (1.0 - p_bb_call) * ctx.icm_btn_takes_blinds[ctx.sb]
    })
}

fn ev_bb_call_vs_btn(
    hand_idx: usize,
    btn_push: &[f64; HAND_TYPES],
    _sb_call: &[f64; HAND_TYPES],
    ctx: &ThreeMaxContext,
    cache: &EquityCache,
) -> f64 {
    average_over_combos(hand_idx, ctx, |combo_idx| {
        let unblocked = &ctx.unblocked[hand_idx][combo_idx];
        hu_ev(
            hand_idx,
            unblocked,
            btn_push,
            cache,
            ctx.icm_hu_btn_bb_sb_fold_bb_win[ctx.bb],
            ctx.icm_hu_btn_bb_sb_fold_btn_win[ctx.bb],
        )
    })
}

fn ev_bb_fold_vs_both(
    hand_idx: usize,
    btn_push: &[f64; HAND_TYPES],
    sb_call: &[f64; HAND_TYPES],
    ctx: &ThreeMaxContext,
    cache: &EquityCache,
) -> f64 {
    average_over_combos(hand_idx, ctx, |combo_idx| {
        let unblocked = &ctx.unblocked[hand_idx][combo_idx];
        hu_ev_spectator(
            unblocked,
            btn_push,
            sb_call,
            cache,
            ctx.icm_hu_btn_sb_bb_fold_btn_win[ctx.bb],
            ctx.icm_hu_btn_sb_bb_fold_sb_win[ctx.bb],
        )
    })
}

fn ev_bb_call_vs_both(
    hand_idx: usize,
    btn_push: &[f64; HAND_TYPES],
    sb_call: &[f64; HAND_TYPES],
    ctx: &ThreeMaxContext,
    _cache: &EquityCache,
) -> f64 {
    average_over_combos(hand_idx, ctx, |combo_idx| {
        let hero = ctx.combos[hand_idx][combo_idx];
        three_way_ev(
            hero,
            ctx.bb,
            btn_push,
            sb_call,
            ctx,
            (0xC01 + hand_idx as u64) << 7,
        )
    })
}

fn ev_sb_push_after_btn_fold(
    hand_idx: usize,
    bb_call: &[f64; HAND_TYPES],
    ctx: &ThreeMaxContext,
    cache: &EquityCache,
) -> f64 {
    average_over_combos(hand_idx, ctx, |combo_idx| {
        let unblocked = &ctx.unblocked[hand_idx][combo_idx];
        let p_bb_call = p_action(unblocked, bb_call);
        let hu = hu_ev(
            hand_idx,
            unblocked,
            bb_call,
            cache,
            ctx.icm_hu_sb_bb_btn_fold_sb_win[ctx.sb],
            ctx.icm_hu_sb_bb_btn_fold_bb_win[ctx.sb],
        );
        (1.0 - p_bb_call) * ctx.icm_sb_walks[ctx.sb] + p_bb_call * hu
    })
}

fn ev_bb_call_vs_sb(
    hand_idx: usize,
    sb_push: &[f64; HAND_TYPES],
    ctx: &ThreeMaxContext,
    cache: &EquityCache,
) -> f64 {
    average_over_combos(hand_idx, ctx, |combo_idx| {
        let unblocked = &ctx.unblocked[hand_idx][combo_idx];
        hu_ev(
            hand_idx,
            unblocked,
            sb_push,
            cache,
            ctx.icm_hu_sb_bb_btn_fold_bb_win[ctx.bb],
            ctx.icm_hu_sb_bb_btn_fold_sb_win[ctx.bb],
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn average_btn_equity(
    btn_push: &[f64; HAND_TYPES],
    sb_call: &[f64; HAND_TYPES],
    bb_call_vs_btn: &[f64; HAND_TYPES],
    bb_call_vs_both: &[f64; HAND_TYPES],
    sb_push: &[f64; HAND_TYPES],
    bb_call_vs_sb: &[f64; HAND_TYPES],
    ctx: &ThreeMaxContext,
    cache: &EquityCache,
) -> f64 {
    let mut weighted = 0.0;
    let mut total = 0.0;
    for (hand, &freq) in btn_push.iter().enumerate() {
        let weight = ctx.combos[hand].len() as f64;
        if weight == 0.0 {
            continue;
        }
        let ev_push = ev_btn_push(hand, sb_call, bb_call_vs_btn, bb_call_vs_both, ctx, cache);
        let ev_fold = ev_btn_fold(hand, sb_push, bb_call_vs_sb, ctx, cache);
        let freq = freq.clamp(0.0, 1.0);
        weighted += weight * (freq * ev_push + (1.0 - freq) * ev_fold);
        total += weight;
    }
    if total == 0.0 {
        1.0 / 3.0
    } else {
        weighted / total
    }
}

#[allow(clippy::too_many_arguments)]
fn average_sb_equity(
    btn_push: &[f64; HAND_TYPES],
    sb_call: &[f64; HAND_TYPES],
    bb_call_vs_btn: &[f64; HAND_TYPES],
    bb_call_vs_both: &[f64; HAND_TYPES],
    sb_push: &[f64; HAND_TYPES],
    bb_call_vs_sb: &[f64; HAND_TYPES],
    ctx: &ThreeMaxContext,
    cache: &EquityCache,
) -> f64 {
    let mut weighted = 0.0;
    let mut total = 0.0;
    for hand in 0..HAND_TYPES {
        let weight = ctx.combos[hand].len() as f64;
        if weight == 0.0 {
            continue;
        }
        let unblocked = &ctx.unblocked[hand][0];
        let p_btn_push = p_action(unblocked, btn_push);
        let call_freq = sb_call[hand].clamp(0.0, 1.0);
        let push_freq = sb_push[hand].clamp(0.0, 1.0);
        let ev_vs_push = call_freq * ev_sb_call_vs_btn(hand, btn_push, bb_call_vs_both, ctx, cache)
            + (1.0 - call_freq) * ev_sb_fold_vs_btn(hand, btn_push, bb_call_vs_btn, ctx, cache);
        let ev_vs_fold = push_freq * ev_sb_push_after_btn_fold(hand, bb_call_vs_sb, ctx, cache)
            + (1.0 - push_freq) * ctx.icm_bb_walks[ctx.sb];
        weighted += weight * (p_btn_push * ev_vs_push + (1.0 - p_btn_push) * ev_vs_fold);
        total += weight;
    }
    if total == 0.0 {
        1.0 / 3.0
    } else {
        weighted / total
    }
}

#[allow(clippy::too_many_arguments)]
fn average_bb_equity(
    btn_push: &[f64; HAND_TYPES],
    sb_call: &[f64; HAND_TYPES],
    bb_call_vs_btn: &[f64; HAND_TYPES],
    bb_call_vs_both: &[f64; HAND_TYPES],
    sb_push: &[f64; HAND_TYPES],
    bb_call_vs_sb: &[f64; HAND_TYPES],
    ctx: &ThreeMaxContext,
    cache: &EquityCache,
) -> f64 {
    let mut weighted = 0.0;
    let mut total = 0.0;
    for hand in 0..HAND_TYPES {
        let weight = ctx.combos[hand].len() as f64;
        if weight == 0.0 {
            continue;
        }
        let unblocked = &ctx.unblocked[hand][0];
        let p_btn_push = p_action(unblocked, btn_push);
        let p_sb_call = p_action(unblocked, sb_call);
        let p_sb_push = p_action(unblocked, sb_push);
        let vs_both = bb_call_vs_both[hand].clamp(0.0, 1.0);
        let vs_btn = bb_call_vs_btn[hand].clamp(0.0, 1.0);
        let vs_sb = bb_call_vs_sb[hand].clamp(0.0, 1.0);

        let ev_btn_push_sb_call = vs_both * ev_bb_call_vs_both(hand, btn_push, sb_call, ctx, cache)
            + (1.0 - vs_both) * ev_bb_fold_vs_both(hand, btn_push, sb_call, ctx, cache);
        let ev_btn_push_sb_fold = vs_btn * ev_bb_call_vs_btn(hand, btn_push, sb_call, ctx, cache)
            + (1.0 - vs_btn) * ctx.icm_btn_takes_blinds[ctx.bb];
        let ev_btn_fold_sb_push = vs_sb * ev_bb_call_vs_sb(hand, sb_push, ctx, cache)
            + (1.0 - vs_sb) * ctx.icm_sb_walks[ctx.bb];
        let ev_btn_fold_sb_fold = ctx.icm_bb_walks[ctx.bb];

        let ev = p_btn_push
            * (p_sb_call * ev_btn_push_sb_call + (1.0 - p_sb_call) * ev_btn_push_sb_fold)
            + (1.0 - p_btn_push)
                * (p_sb_push * ev_btn_fold_sb_push + (1.0 - p_sb_push) * ev_btn_fold_sb_fold);
        weighted += weight * ev;
        total += weight;
    }
    if total == 0.0 {
        1.0 / 3.0
    } else {
        weighted / total
    }
}

fn average_over_combos(
    hand_idx: usize,
    ctx: &ThreeMaxContext,
    mut ev_combo: impl FnMut(usize) -> f64,
) -> f64 {
    if ctx.combos[hand_idx].is_empty() {
        return 0.0;
    }
    ev_combo(0)
}

fn p_action(unblocked: &[u8; HAND_TYPES], freq: &[f64; HAND_TYPES]) -> f64 {
    let mut live = 0.0;
    let mut acted = 0.0;
    for (idx, &count) in unblocked.iter().enumerate() {
        let live_f = count as f64;
        if live_f == 0.0 {
            continue;
        }
        live += live_f;
        acted += live_f * freq[idx].clamp(0.0, 1.0);
    }
    if live <= 0.0 {
        0.0
    } else {
        acted / live
    }
}

fn hu_ev(
    hero_idx: usize,
    unblocked: &[u8; HAND_TYPES],
    villain_freq: &[f64; HAND_TYPES],
    cache: &EquityCache,
    icm_win: f64,
    icm_lose: f64,
) -> f64 {
    let mut weight = 0.0;
    let mut equity_weight = 0.0;
    for (villain_idx, &count) in unblocked.iter().enumerate() {
        let w = count as f64 * villain_freq[villain_idx].clamp(0.0, 1.0);
        if w <= 0.0 {
            continue;
        }
        weight += w;
        equity_weight += w * cache.equity(hero_idx as u8, villain_idx as u8);
    }
    if weight <= 0.0 {
        return 0.5 * icm_win + 0.5 * icm_lose;
    }
    let equity = equity_weight / weight;
    equity * icm_win + (1.0 - equity) * icm_lose
}

fn hu_ev_spectator(
    unblocked: &[u8; HAND_TYPES],
    range_a: &[f64; HAND_TYPES],
    range_b: &[f64; HAND_TYPES],
    cache: &EquityCache,
    icm_a_wins: f64,
    icm_b_wins: f64,
) -> f64 {
    let mut weight = 0.0;
    let mut a_eq = 0.0;
    for i in 0..HAND_TYPES {
        let wa = unblocked[i] as f64 * range_a[i].clamp(0.0, 1.0);
        if wa <= 0.0 {
            continue;
        }
        for j in 0..HAND_TYPES {
            let wb = unblocked[j] as f64 * range_b[j].clamp(0.0, 1.0);
            if wb <= 0.0 {
                continue;
            }
            let w = wa * wb;
            weight += w;
            a_eq += w * cache.equity(i as u8, j as u8);
        }
    }
    if weight <= 0.0 {
        return 0.5 * icm_a_wins + 0.5 * icm_b_wins;
    }
    let equity = a_eq / weight;
    equity * icm_a_wins + (1.0 - equity) * icm_b_wins
}

fn three_way_ev(
    hero: [Card; 2],
    hero_seat: usize,
    range_a: &[f64; HAND_TYPES],
    range_b: &[f64; HAND_TYPES],
    ctx: &ThreeMaxContext,
    seed: u64,
) -> f64 {
    let (weights_a, total_a) = type_weights(range_a, ctx);
    let (weights_b, total_b) = type_weights(range_b, ctx);
    if total_a <= 0.0 || total_b <= 0.0 {
        return ctx.payouts.get(2).copied().unwrap_or(0.2);
    }

    let mut rng = Rng::new(0xD1B54A32D192ED03 ^ seed ^ card_seed(hero));
    let mut both_ok = 0.0;
    let mut ev = 0.0;

    for _ in 0..THREE_WAY_SAMPLES {
        let Some(hand_a) =
            sample_from_weights(&weights_a, total_a, &[hero[0], hero[1]], ctx, &mut rng)
        else {
            continue;
        };
        let dead = [hero[0], hero[1], hand_a[0], hand_a[1]];
        let Some(hand_b) = sample_from_weights(&weights_b, total_b, &dead, ctx, &mut rng) else {
            continue;
        };
        both_ok += 1.0;
        let (btn_h, sb_h, bb_h, hero_idx) = if hero_seat == ctx.btn {
            (hero, hand_a, hand_b, 0)
        } else if hero_seat == ctx.sb {
            (hand_a, hero, hand_b, 1)
        } else {
            (hand_a, hand_b, hero, 2)
        };
        ev += ctx.three_way.get_or_compute(
            combo_index(btn_h),
            combo_index(sb_h),
            combo_index(bb_h),
            hero_idx,
            || {
                equity_3way_icm_with_cache(
                    btn_h,
                    sb_h,
                    bb_h,
                    ctx.all_in_stacks,
                    ctx.three_way_uncalled,
                    &ctx.payouts,
                    THREE_WAY_BOARDS,
                    ctx.icm_cache_ref(),
                )
            },
        );
    }

    if both_ok > 0.0 {
        ev / both_ok
    } else {
        ctx.payouts.get(2).copied().unwrap_or(0.2)
    }
}

fn type_weights(range: &[f64; HAND_TYPES], ctx: &ThreeMaxContext) -> ([f64; HAND_TYPES], f64) {
    let mut weights = [0.0; HAND_TYPES];
    let mut total = 0.0;
    for (idx, &freq) in range.iter().enumerate() {
        let weight = ctx.combos[idx].len() as f64 * freq.clamp(0.0, 1.0);
        weights[idx] = weight;
        total += weight;
    }
    (weights, total)
}

fn sample_from_weights(
    weights: &[f64; HAND_TYPES],
    total: f64,
    dead: &[Card],
    ctx: &ThreeMaxContext,
    rng: &mut Rng,
) -> Option<[Card; 2]> {
    if total <= 0.0 {
        return None;
    }
    for _ in 0..48 {
        let mut pick = rng.gen_f64() * total;
        let mut chosen = 0;
        for (idx, &weight) in weights.iter().enumerate() {
            if pick < weight {
                chosen = idx;
                break;
            }
            pick -= weight;
            chosen = idx;
        }
        let combos = &ctx.combos[chosen];
        if combos.is_empty() {
            continue;
        }
        let combo = combos[rng.gen_range(combos.len())];
        if !overlaps(dead, combo) {
            return Some(combo);
        }
    }
    None
}

fn overlaps(dead: &[Card], hand: [Card; 2]) -> bool {
    dead.iter().any(|card| *card == hand[0] || *card == hand[1])
}

fn card_seed(hand: [Card; 2]) -> u64 {
    hand[0].rank() as u64 * 17
        + hand[0].suit() as u64 * 5
        + hand[1].rank() as u64 * 13
        + hand[1].suit() as u64
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
            0
        } else {
            (self.next_u64() as usize) % max
        }
    }

    fn gen_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

/// Диагностика EV колла BB (без FP). Фиксированные диапазоны по combo-share.
pub(crate) fn debug_bb_report(
    input: &SolverInput,
    cache: &EquityCache,
    hand_label: &str,
    btn_combo_share: f64,
    sb_combo_share: f64,
) -> Result<String, String> {
    use crate::equity_3way::icm_showdown_equity;
    use crate::equity_cache::combo_label;
    use std::fmt::Write;

    let hand_idx = (0..HAND_TYPES)
        .find(|&idx| combo_label(idx as u8).eq_ignore_ascii_case(hand_label))
        .ok_or_else(|| format!("unknown hand label: {hand_label}"))?;

    let ctx = ThreeMaxContext::new(input, ThreeWayCache::new())
        .ok_or_else(|| "failed to build ThreeMaxContext".to_string())?;

    let btn_push = top_combo_range(cache, btn_combo_share);
    let sb_call = top_combo_range(cache, sb_combo_share);

    let mut out = String::new();
    writeln!(
        out,
        "BB debug: hand={} (idx={})  stacks={:?}",
        hand_label, hand_idx, input.stacks
    )
    .unwrap();
    writeln!(
        out,
        "Fixed ranges: BTN push combo-share {:.1}%  SB call combo-share {:.1}%",
        combo_range_share(&btn_push) * 100.0,
        combo_range_share(&sb_call) * 100.0
    )
    .unwrap();
    writeln!(
        out,
        "3-way contested={:?}  uncalled={:?}",
        ctx.all_in_stacks, ctx.three_way_uncalled
    )
    .unwrap();
    writeln!(out).unwrap();

    let icm_win_hu = ctx.icm_hu_btn_bb_sb_fold_bb_win[ctx.bb];
    let icm_lose_hu = ctx.icm_hu_btn_bb_sb_fold_btn_win[ctx.bb];
    let ev_call_hu = ev_bb_call_vs_btn(hand_idx, &btn_push, &sb_call, &ctx, cache);
    let ev_fold_hu = ctx.icm_btn_takes_blinds[ctx.bb];
    let p_bb_win_hu = hu_equity_vs_range(hand_idx, &ctx.unblocked[hand_idx][0], &btn_push, cache);
    let ev_call_hu_formula = p_bb_win_hu * icm_win_hu + (1.0 - p_bb_win_hu) * icm_lose_hu;

    writeln!(out, "=== Card removal (avg over BB combos) ===").unwrap();
    writeln!(
        out,
        "{}",
        debug_card_removal(hand_idx, &btn_push, &sb_call, &ctx)
    )
    .unwrap();
    writeln!(out).unwrap();

    writeln!(out, "=== HU (SB fold) ===").unwrap();
    writeln!(out, "EV_call = {:.2}%", ev_call_hu * 100.0).unwrap();
    writeln!(out, "P(BB wins vs BTN push): {:.2}%", p_bb_win_hu * 100.0).unwrap();
    writeln!(
        out,
        "ICM: win={:.2}%  lose={:.2}%",
        icm_win_hu * 100.0,
        icm_lose_hu * 100.0
    )
    .unwrap();
    writeln!(
        out,
        "EV_call = {:.2}% × {:.2}% + {:.2}% × {:.2}% = {:.2}%",
        p_bb_win_hu * 100.0,
        icm_win_hu * 100.0,
        (1.0 - p_bb_win_hu) * 100.0,
        icm_lose_hu * 100.0,
        ev_call_hu_formula * 100.0
    )
    .unwrap();
    writeln!(out, "EV_fold = {:.2}%", ev_fold_hu * 100.0).unwrap();
    writeln!(out).unwrap();

    let ev_call_3w = ev_bb_call_vs_both(hand_idx, &btn_push, &sb_call, &ctx, cache);
    let ev_fold_3w = ev_bb_fold_vs_both(hand_idx, &btn_push, &sb_call, &ctx, cache);

    let breakdown = debug_bb_3way_scenarios(hand_idx, &btn_push, &sb_call, &ctx, THREE_WAY_BOARDS);

    writeln!(out, "=== 3-way (SB call) ===").unwrap();
    writeln!(out, "{}", breakdown.report).unwrap();
    writeln!(out, "EV_call (solver) = {:.2}%", ev_call_3w * 100.0).unwrap();
    writeln!(out, "EV_fold = {:.2}%", ev_fold_3w * 100.0).unwrap();
    writeln!(
        out,
        "Fold spectator ICM: btn_win={:.2}%  sb_win={:.2}%",
        ctx.icm_hu_btn_sb_bb_fold_btn_win[ctx.bb] * 100.0,
        ctx.icm_hu_btn_sb_bb_fold_sb_win[ctx.bb] * 100.0
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Comparison: EV_call_HU {:.2}% vs EV_call_3way {:.2}% (Δ {:.2}%)",
        ev_call_hu * 100.0,
        ev_call_3w * 100.0,
        (ev_call_3w - ev_call_hu) * 100.0
    )
    .unwrap();
    writeln!(out).unwrap();

    writeln!(out, "=== ICM terminals (fixed stacks) ===").unwrap();
    let contested = ctx.all_in_stacks;
    let uncalled = ctx.three_way_uncalled;
    let payouts = &ctx.payouts;
    let pre_showdown = [
        contested[0] + uncalled[0],
        contested[1] + uncalled[1],
        contested[2] + uncalled[2],
    ];
    let dummy_ranks = [
        debug_eval_rank(&["Ah", "Ad", "Ac", "As", "Kh", "2c", "3d"]),
        debug_eval_rank(&["Kh", "Kd", "Kc", "Qh", "Qd", "2s", "3s"]),
        debug_eval_rank(&["9h", "8d", "7c", "5s", "4h", "3c", "2d"]),
    ];
    let terminal_cases = [
        ("S1: BB wins all", [1100.0, 0.0, 2300.0]),
        ("S2: SB main, BB side", [1100.0, 1500.0, 800.0]),
        ("S3: SB main, BTN side (BB bust)", [1900.0, 1500.0, 0.0]),
        ("S4: BTN wins all (both bust)", [3400.0, 0.0, 0.0]),
    ];
    for (label, stacks) in terminal_cases {
        let icm = icm_showdown_equity(stacks, pre_showdown, dummy_ranks, payouts);
        writeln!(
            out,
            "{}: stacks={:?}  BB $EV={:.2}%  full={:?}",
            label,
            stacks,
            icm[2] * 100.0,
            icm
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "=== HRC comparison (fill in) ===").unwrap();
    writeln!(
        out,
        "HRC HU (SB fold) EV_call({}) = ?  ours {:.2}%",
        hand_label,
        ev_call_hu * 100.0
    )
    .unwrap();
    writeln!(
        out,
        "HRC 3-way (SB call) EV_call({}) = ?  ours {:.2}%",
        hand_label,
        ev_call_3w * 100.0
    )
    .unwrap();
    writeln!(out).unwrap();
    if ev_call_3w > ev_call_hu {
        writeln!(out, "OK: EV_call(3-way) > EV_call(HU)").unwrap();
    } else {
        writeln!(
            out,
            "ANOMALY: EV_call(3-way) {:.4} <= EV_call(HU) {:.4}",
            ev_call_3w, ev_call_hu
        )
        .unwrap();
    }
    if ev_call_3w > ev_fold_3w {
        writeln!(out, "OK: EV_call(3-way) > EV_fold(3-way) for {hand_label}").unwrap();
    } else {
        writeln!(
            out,
            "ANOMALY: EV_call(3-way) {:.4} <= EV_fold(3-way) {:.4}",
            ev_call_3w, ev_fold_3w
        )
        .unwrap();
    }

    Ok(out)
}

fn combo_range_share(range: &[f64; HAND_TYPES]) -> f64 {
    let mut used = 0.0;
    let total: f64 = (0..HAND_TYPES)
        .map(|idx| expand_combo(idx as u8).len() as f64)
        .sum();
    for (idx, &freq) in range.iter().enumerate() {
        if freq > 0.0 {
            used += expand_combo(idx as u8).len() as f64;
        }
    }
    if total > 0.0 {
        used / total
    } else {
        0.0
    }
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
    if total > 0.0 {
        weighted / total
    } else {
        0.5
    }
}

fn hu_equity_vs_range(
    hero_idx: usize,
    unblocked: &[u8; HAND_TYPES],
    villain_freq: &[f64; HAND_TYPES],
    cache: &EquityCache,
) -> f64 {
    let mut weight = 0.0;
    let mut equity_weight = 0.0;
    for (villain_idx, &count) in unblocked.iter().enumerate() {
        let w = count as f64 * villain_freq[villain_idx].clamp(0.0, 1.0);
        if w <= 0.0 {
            continue;
        }
        weight += w;
        equity_weight += w * cache.equity(hero_idx as u8, villain_idx as u8);
    }
    if weight <= 0.0 {
        0.5
    } else {
        equity_weight / weight
    }
}

fn debug_eval_rank(cards: &[&str; 7]) -> crate::hand_evaluator::HandRank {
    use crate::hand_evaluator::evaluate_hand;
    use std::str::FromStr;
    let parsed: [Card; 7] = cards
        .iter()
        .map(|s| Card::from_str(s).expect("card"))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    evaluate_hand(&parsed)
}

fn classify_bb_scenario(
    r_btn: crate::hand_evaluator::HandRank,
    r_sb: crate::hand_evaluator::HandRank,
    r_bb: crate::hand_evaluator::HandRank,
) -> u8 {
    if r_bb > r_sb && r_bb > r_btn {
        1
    } else if r_sb > r_bb && r_bb > r_btn {
        2
    } else if r_sb > r_btn && r_btn > r_bb {
        3
    } else if r_btn > r_sb && r_btn > r_bb {
        4
    } else {
        0
    }
}

struct ScenarioBreakdown {
    report: String,
}

fn debug_bb_3way_scenarios(
    hand_idx: usize,
    btn_push: &[f64; HAND_TYPES],
    sb_call: &[f64; HAND_TYPES],
    ctx: &ThreeMaxContext,
    boards_per_sample: u64,
) -> ScenarioBreakdown {
    use crate::equity_3way::{finalize_3way_stacks, icm_showdown_equity};
    use std::fmt::Write;

    let bb_combo = ctx.combos[hand_idx][0];
    let contested = ctx.all_in_stacks;
    let uncalled = ctx.three_way_uncalled;
    let pre_showdown = [
        contested[0] + uncalled[0],
        contested[1] + uncalled[1],
        contested[2] + uncalled[2],
    ];
    let payouts = &ctx.payouts;

    let mut count = [0u64; 5];
    let mut ev_sum = [0.0; 5];

    let (weights_btn, total_btn) = type_weights(btn_push, ctx);
    let (weights_sb, total_sb) = type_weights(sb_call, ctx);

    let mut rng = Rng::new(0xBB3A7D1A6 ^ (hand_idx as u64));

    for _ in 0..THREE_WAY_SAMPLES {
        let Some(btn_hand) = sample_from_weights(
            &weights_btn,
            total_btn,
            &[bb_combo[0], bb_combo[1]],
            ctx,
            &mut rng,
        ) else {
            continue;
        };
        let dead = [bb_combo[0], bb_combo[1], btn_hand[0], btn_hand[1]];
        let Some(sb_hand) = sample_from_weights(&weights_sb, total_sb, &dead, ctx, &mut rng) else {
            continue;
        };

        for _ in 0..boards_per_sample {
            let (ranks, _board) = sample_ranks(btn_hand, sb_hand, bb_combo, &mut rng);
            let stacks = finalize_3way_stacks(contested, uncalled, ranks);
            let icm = icm_showdown_equity(stacks, pre_showdown, ranks, payouts);
            let scenario = classify_bb_scenario(ranks[0], ranks[1], ranks[2]);
            let idx = scenario as usize;
            if idx < count.len() {
                count[idx] += 1;
                ev_sum[idx] += icm[2];
            }
        }
    }

    let labels = ["other/tie", "S1", "S2", "S3", "S4"];
    let total = count.iter().sum::<u64>().max(1) as f64;
    let mut weighted_ev = 0.0_f64;
    let mut out = String::new();
    for i in 1..5 {
        if count[i] > 0 {
            let p = count[i] as f64 / total;
            let ev = ev_sum[i] / count[i] as f64;
            weighted_ev += p * ev;
            writeln!(
                out,
                "P({}) = {:.1}%, $EV_BB(S{}) = {:.2}%",
                labels[i],
                p * 100.0,
                i,
                ev * 100.0
            )
            .unwrap();
        }
    }
    writeln!(
        out,
        "EV_call_3way = Σ P_i × $EV_i = {:.2}%",
        weighted_ev * 100.0
    )
    .unwrap();
    let _ = weighted_ev;
    ScenarioBreakdown { report: out }
}

fn debug_card_removal(
    hand_idx: usize,
    btn_push: &[f64; HAND_TYPES],
    sb_call: &[f64; HAND_TYPES],
    ctx: &ThreeMaxContext,
) -> String {
    use std::fmt::Write;

    let mut out = String::new();
    let n_combos = ctx.combos[hand_idx].len().max(1);
    let mut btn_kept_sum = 0u32;
    let mut btn_total_sum = 0u32;
    let mut sb_kept_sum = 0u32;
    let mut sb_total_sum = 0u32;

    for combo_idx in 0..ctx.combos[hand_idx].len() {
        let unblocked = &ctx.unblocked[hand_idx][combo_idx];
        let (btn_kept, btn_total) = range_combo_counts(unblocked, btn_push);
        let (sb_kept, sb_total) = range_combo_counts(unblocked, sb_call);
        btn_kept_sum += btn_kept;
        btn_total_sum += btn_total;
        sb_kept_sum += sb_kept;
        sb_total_sum += sb_total;
    }

    let btn_kept_avg = btn_kept_sum as f64 / n_combos as f64;
    let btn_total_avg = btn_total_sum as f64 / n_combos as f64;
    let sb_kept_avg = sb_kept_sum as f64 / n_combos as f64;
    let sb_total_avg = sb_total_sum as f64 / n_combos as f64;

    writeln!(
        out,
        "BB combos: {} (hand type idx={})",
        ctx.combos[hand_idx].len(),
        hand_idx
    )
    .unwrap();
    writeln!(
        out,
        "BTN push range: {:.0}/{:.0} combos kept ({:.1}%)",
        btn_kept_avg,
        btn_total_avg,
        pct(btn_kept_avg, btn_total_avg)
    )
    .unwrap();
    writeln!(
        out,
        "SB call range: {:.0}/{:.0} combos kept ({:.1}%)",
        sb_kept_avg,
        sb_total_avg,
        pct(sb_kept_avg, sb_total_avg)
    )
    .unwrap();
    writeln!(
        out,
        "Nominal range sizes: BTN push {:.0} combos, SB call {:.0} combos",
        btn_total_avg, sb_total_avg
    )
    .unwrap();
    out
}

fn range_combo_counts(unblocked: &[u8; HAND_TYPES], range: &[f64; HAND_TYPES]) -> (u32, u32) {
    let mut kept = 0u32;
    let mut total = 0u32;
    for (idx, &freq) in range.iter().enumerate() {
        if freq <= 0.0 {
            continue;
        }
        let combos = expand_combo(idx as u8).len() as u32;
        total += combos;
        kept += unblocked[idx] as u32;
    }
    (kept, total)
}

fn pct(kept: f64, total: f64) -> f64 {
    if total > 0.0 {
        100.0 * kept / total
    } else {
        0.0
    }
}

fn sample_ranks(
    btn_hand: [Card; 2],
    sb_hand: [Card; 2],
    bb_hand: [Card; 2],
    rng: &mut Rng,
) -> ([crate::hand_evaluator::HandRank; 3], [Card; 5]) {
    use crate::hand_evaluator::evaluate_hand;
    let dead = [
        btn_hand[0],
        btn_hand[1],
        sb_hand[0],
        sb_hand[1],
        bb_hand[0],
        bb_hand[1],
    ];
    let available = available_cards_debug(&dead);
    let board = sample_board_debug(&available, rng);
    let ranks = [
        evaluate_hand(&[
            btn_hand[0],
            btn_hand[1],
            board[0],
            board[1],
            board[2],
            board[3],
            board[4],
        ]),
        evaluate_hand(&[
            sb_hand[0], sb_hand[1], board[0], board[1], board[2], board[3], board[4],
        ]),
        evaluate_hand(&[
            bb_hand[0], bb_hand[1], board[0], board[1], board[2], board[3], board[4],
        ]),
    ];
    (ranks, board)
}

fn available_cards_debug(dead: &[Card]) -> Vec<Card> {
    let mut cards = Vec::new();
    for rank in 2..=14 {
        for suit in 0..4 {
            let c = Card::new(rank, suit);
            if !dead.iter().any(|d| *d == c) {
                cards.push(c);
            }
        }
    }
    cards
}

fn sample_board_debug(available: &[Card], rng: &mut Rng) -> [Card; 5] {
    let mut pool = available.to_vec();
    for i in 0..5 {
        let j = i + rng.gen_range(pool.len() - i);
        pool.swap(i, j);
    }
    [pool[0], pool[1], pool[2], pool[3], pool[4]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity_cache::combo_index;
    use crate::solver::SolverInput;
    use std::path::Path;
    use std::sync::OnceLock;

    fn test_cache() -> &'static EquityCache {
        static CACHE: OnceLock<EquityCache> = OnceLock::new();
        CACHE.get_or_init(|| {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("equity_cache.bin");
            EquityCache::load(&path).expect("equity_cache.bin")
        })
    }

    #[test]
    fn three_way_effective_stacks_used() {
        let input = SolverInput {
            stacks: vec![2000.0, 500.0, 900.0],
            payouts: vec![0.5, 0.3, 0.2],
            small_blind: 50.0,
            big_blind: 100.0,
            ante: 0.0,
            button_index: 0,
            max_iterations: 1,
            tolerance: 0.001,
            num_players: 3,
            verbose_convergence: false,
            profile: false,
            algorithm: crate::solver::Algorithm::FictitiousPlay,
        };
        let ctx = ThreeMaxContext::new(&input, ThreeWayCache::new()).expect("ctx");
        assert_eq!(ctx.all_in_stacks, [900.0, 500.0, 900.0]);
        assert_eq!(ctx.three_way_uncalled, [1100.0, 0.0, 0.0]);
    }

    #[test]
    fn bb_call_vs_both_aa_ev_beats_fold() {
        let input = SolverInput {
            stacks: vec![500.0, 1000.0, 5000.0],
            payouts: vec![0.5, 0.3, 0.2],
            small_blind: 50.0,
            big_blind: 100.0,
            ante: 0.0,
            button_index: 0,
            max_iterations: 1,
            tolerance: 0.001,
            num_players: 3,
            verbose_convergence: false,
            profile: false,
            algorithm: crate::solver::Algorithm::FictitiousPlay,
        };
        let ctx = ThreeMaxContext::new(&input, ThreeWayCache::new()).expect("ctx");
        let ones = [1.0; HAND_TYPES];
        let aa = combo_index([Card::new(14, 0), Card::new(14, 1)]) as usize;
        let ev_call = ev_bb_call_vs_both(aa, &ones, &ones, &ctx, test_cache());
        let ev_fold = ev_bb_fold_vs_both(aa, &ones, &ones, &ctx, test_cache());
        println!(
            "AA vs 100% ranges: call={ev_call:.4} fold={ev_fold:.4} diff={:.4}",
            ev_call - ev_fold
        );
        println!(
            "precomputed fold btn_win={} sb_win={}",
            ctx.icm_hu_btn_sb_bb_fold_btn_win[ctx.bb], ctx.icm_hu_btn_sb_bb_fold_sb_win[ctx.bb]
        );
        assert!(
            ev_call > ev_fold,
            "AA should call 3-way: call {ev_call} fold {ev_fold}"
        );
    }
}
