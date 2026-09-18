use rayon::prelude::*;

use crate::card::Card;
use crate::equity_3way::equity_3way_places;
use crate::equity_cache::{expand_combo, EquityCache};
use crate::solver::{
    best_response, build_unblocked, empty_output, tournament_equity, transfer, SolverInput,
    SolverOutput, ThreeMaxRanges, HAND_TYPES,
};

const THREE_WAY_SAMPLES: u32 = 64;
const THREE_WAY_BOARDS: u64 = 12;

pub(crate) fn solve_3max(input: &SolverInput, cache: &EquityCache) -> SolverOutput {
    let Some(ctx) = ThreeMaxContext::new(input) else {
        return empty_output(input.stacks.len());
    };

    let mut btn_push = [1.0; HAND_TYPES];
    let mut sb_call_vs_btn = [1.0; HAND_TYPES];
    let mut bb_call_vs_btn = [1.0; HAND_TYPES];
    let mut bb_call_vs_both = [1.0; HAND_TYPES];
    let mut sb_push = [1.0; HAND_TYPES];
    let mut bb_call_vs_sb = [1.0; HAND_TYPES];

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

        let br_btn_push: Vec<f64> = (0..HAND_TYPES)
            .into_par_iter()
            .map(|hand| {
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
            })
            .collect();

        let br_sb_call: Vec<f64> = (0..HAND_TYPES)
            .into_par_iter()
            .map(|hand| {
                let ev_call = ev_sb_call_vs_btn(hand, &btn_push, &bb_call_vs_both, &ctx, cache);
                let ev_fold = ev_sb_fold_vs_btn(hand, &btn_push, &bb_call_vs_btn, &ctx, cache);
                best_response(ev_call, ev_fold, sb_call_vs_btn[hand])
            })
            .collect();

        let br_bb_vs_btn: Vec<f64> = (0..HAND_TYPES)
            .into_par_iter()
            .map(|hand| {
                let ev_call = ev_bb_call_vs_btn(hand, &btn_push, &sb_call_vs_btn, &ctx, cache);
                let ev_fold = ctx.icm_btn_takes_blinds[ctx.bb];
                best_response(ev_call, ev_fold, bb_call_vs_btn[hand])
            })
            .collect();

        let br_bb_vs_both: Vec<f64> = (0..HAND_TYPES)
            .into_par_iter()
            .map(|hand| {
                let ev_call = ev_bb_call_vs_both(hand, &btn_push, &sb_call_vs_btn, &ctx, cache);
                let ev_fold = ev_bb_fold_vs_both(hand, &btn_push, &sb_call_vs_btn, &ctx, cache);
                best_response(ev_call, ev_fold, bb_call_vs_both[hand])
            })
            .collect();

        let br_sb_push: Vec<f64> = (0..HAND_TYPES)
            .into_par_iter()
            .map(|hand| {
                let ev_push = ev_sb_push_after_btn_fold(hand, &bb_call_vs_sb, &ctx, cache);
                best_response(ev_push, ctx.icm_bb_walks[ctx.sb], sb_push[hand])
            })
            .collect();

        let br_bb_vs_sb: Vec<f64> = (0..HAND_TYPES)
            .into_par_iter()
            .map(|hand| {
                let ev_call = ev_bb_call_vs_sb(hand, &sb_push, &ctx, cache);
                best_response(ev_call, ctx.icm_sb_walks[ctx.bb], bb_call_vs_sb[hand])
            })
            .collect();

        let t = iteration as f64;
        let mut change = 0.0;
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

            btn_push[hand] = next_btn;
            sb_call_vs_btn[hand] = next_sb_call;
            bb_call_vs_btn[hand] = next_bb_btn;
            bb_call_vs_both[hand] = next_bb_both;
            sb_push[hand] = next_sb_push;
            bb_call_vs_sb[hand] = next_bb_sb;
        }

        let mean_change = change / (6.0 * HAND_TYPES as f64);
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
}

impl ThreeMaxContext {
    fn new(input: &SolverInput) -> Option<Self> {
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
        })
    }
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
        let ordered = if hero_seat == ctx.btn {
            [hero, hand_a, hand_b]
        } else if hero_seat == ctx.sb {
            [hand_a, hero, hand_b]
        } else {
            [hand_a, hand_b, hero]
        };
        let places = equity_3way_places(
            ordered[0],
            ordered[1],
            ordered[2],
            &ctx.payouts,
            THREE_WAY_BOARDS,
        );
        ev += if hero_seat == ctx.btn {
            places[0]
        } else if hero_seat == ctx.sb {
            places[1]
        } else {
            places[2]
        };
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
