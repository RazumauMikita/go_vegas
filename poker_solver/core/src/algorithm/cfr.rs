use crate::algorithm::{run_loop, SolverAlgorithm};
use crate::equity_cache::EquityCache;
use crate::solver::{
    average_sb_equity, build_hu_output, empty_output, ev_call_bb, ev_push_sb, HuContext,
    SolverInput, SolverOutput, HAND_TYPES,
};

pub struct CfrSolver<'a> {
    input: SolverInput,
    cache: &'a EquityCache,
    ctx: HuContext,
    regret_sb: [[f64; 2]; HAND_TYPES],
    regret_bb: [[f64; 2]; HAND_TYPES],
    strategy_sb: [[f64; 2]; HAND_TYPES],
    strategy_bb: [[f64; 2]; HAND_TYPES],
    prev_push: [f64; HAND_TYPES],
    prev_call: [f64; HAND_TYPES],
    iterations_done: usize,
    has_converged: bool,
}

impl<'a> CfrSolver<'a> {
    pub fn init(input: &SolverInput, cache: &'a EquityCache) -> Option<Self> {
        let ctx = HuContext::new(input)?;
        Some(Self {
            input: input.clone(),
            cache,
            ctx,
            regret_sb: [[0.0; 2]; HAND_TYPES],
            regret_bb: [[0.0; 2]; HAND_TYPES],
            strategy_sb: [[0.5; 2]; HAND_TYPES],
            strategy_bb: [[0.5; 2]; HAND_TYPES],
            prev_push: [0.0; HAND_TYPES],
            prev_call: [0.0; HAND_TYPES],
            iterations_done: 0,
            has_converged: false,
        })
    }

    pub fn run(input: &SolverInput, cache: &'a EquityCache) -> SolverOutput {
        match Self::init(input, cache) {
            Some(solver) => run_loop(solver, input.max_iterations),
            None => empty_output(input.stacks.len()),
        }
    }

    fn recompute_strategy(&mut self) {
        for h in 0..HAND_TYPES {
            regret_match(&self.regret_sb[h], &mut self.strategy_sb[h]);
            regret_match(&self.regret_bb[h], &mut self.strategy_bb[h]);
        }
    }

    fn current_ranges(&self) -> ([f64; HAND_TYPES], [f64; HAND_TYPES]) {
        let mut push_range = [0.0; HAND_TYPES];
        let mut call_range = [0.0; HAND_TYPES];
        for h in 0..HAND_TYPES {
            push_range[h] = self.strategy_sb[h][0];
            call_range[h] = self.strategy_bb[h][0];
        }
        (push_range, call_range)
    }

    fn compute_change(
        &self,
        push_range: &[f64; HAND_TYPES],
        call_range: &[f64; HAND_TYPES],
    ) -> f64 {
        let mut change = 0.0;
        for h in 0..HAND_TYPES {
            change += (push_range[h] - self.prev_push[h]).abs();
            change += (call_range[h] - self.prev_call[h]).abs();
        }
        change
    }

    #[cfg(test)]
    fn max_instantaneous_regret(&self) -> f64 {
        let (push_range, call_range) = self.current_ranges();
        let mut max_regret: f64 = 0.0;
        for hand_sb in 0..HAND_TYPES {
            let ev_push = ev_push_sb(hand_sb, &call_range, &self.ctx, self.cache);
            let ev_fold = self.ctx.ev_sb_fold;
            let sigma = push_range[hand_sb].clamp(0.0, 1.0);
            let expected = sigma * ev_push + (1.0 - sigma) * ev_fold;
            max_regret = max_regret.max((ev_push - expected).max(0.0));
            max_regret = max_regret.max((ev_fold - expected).max(0.0));
        }
        for hand_bb in 0..HAND_TYPES {
            let ev_call = ev_call_bb(hand_bb, &push_range, &self.ctx, self.cache);
            let ev_fold = self.ctx.ev_bb_fold;
            let sigma = call_range[hand_bb].clamp(0.0, 1.0);
            let expected = sigma * ev_call + (1.0 - sigma) * ev_fold;
            max_regret = max_regret.max((ev_call - expected).max(0.0));
            max_regret = max_regret.max((ev_fold - expected).max(0.0));
        }
        max_regret
    }
}

impl SolverAlgorithm for CfrSolver<'_> {
    fn iterate(&mut self) {
        self.recompute_strategy();
        let (push_range, call_range) = self.current_ranges();

        for hand_sb in 0..HAND_TYPES {
            let ev_push = ev_push_sb(hand_sb, &call_range, &self.ctx, self.cache);
            let ev_fold = self.ctx.ev_sb_fold;
            let sigma_push = self.strategy_sb[hand_sb][0];
            let sigma_fold = self.strategy_sb[hand_sb][1];
            let expected = sigma_push * ev_push + sigma_fold * ev_fold;
            self.regret_sb[hand_sb][0] = (self.regret_sb[hand_sb][0] + ev_push - expected).max(0.0);
            self.regret_sb[hand_sb][1] = (self.regret_sb[hand_sb][1] + ev_fold - expected).max(0.0);
        }

        for hand_bb in 0..HAND_TYPES {
            let ev_call = ev_call_bb(hand_bb, &push_range, &self.ctx, self.cache);
            let ev_fold = self.ctx.ev_bb_fold;
            let p_push = reach_vs_range(hand_bb, &push_range, &self.ctx);
            let sigma_call = self.strategy_bb[hand_bb][0];
            let sigma_fold = self.strategy_bb[hand_bb][1];
            let expected = sigma_call * ev_call + sigma_fold * ev_fold;
            self.regret_bb[hand_bb][0] =
                (self.regret_bb[hand_bb][0] + p_push * (ev_call - expected)).max(0.0);
            self.regret_bb[hand_bb][1] =
                (self.regret_bb[hand_bb][1] + p_push * (ev_fold - expected)).max(0.0);
        }

        self.iterations_done += 1;
        let change = self.compute_change(&push_range, &call_range);
        self.prev_push = push_range;
        self.prev_call = call_range;
        let mean_change = change / (2.0 * HAND_TYPES as f64);
        if self.iterations_done >= 2
            && (change < self.input.tolerance || mean_change < self.input.tolerance)
        {
            self.has_converged = true;
        }
    }

    fn converged(&self) -> bool {
        self.has_converged
    }

    fn result(&self) -> SolverOutput {
        let mut push_range = [0.0; HAND_TYPES];
        let mut call_range = [0.0; HAND_TYPES];
        let mut strategy = [0.0; 2];
        for h in 0..HAND_TYPES {
            regret_match(&self.regret_sb[h], &mut strategy);
            push_range[h] = strategy[0];
            regret_match(&self.regret_bb[h], &mut strategy);
            call_range[h] = strategy[0];
        }
        let sb_equity = average_sb_equity(&push_range, &call_range, &self.ctx, self.cache);
        build_hu_output(
            &self.ctx,
            self.cache,
            &push_range,
            &call_range,
            sb_equity,
            self.iterations_done,
            self.has_converged,
        )
    }
}

pub(crate) fn regret_match(regret: &[f64; 2], strategy: &mut [f64; 2]) {
    let r0 = regret[0].max(0.0);
    let r1 = regret[1].max(0.0);
    let sum = r0 + r1;
    if sum > 0.0 {
        strategy[0] = r0 / sum;
        strategy[1] = r1 / sum;
    } else {
        strategy[0] = 0.5;
        strategy[1] = 0.5;
    }
}

fn reach_vs_range(hand_idx: usize, range: &[f64; HAND_TYPES], ctx: &HuContext) -> f64 {
    let combo_count = ctx.combos[hand_idx].len();
    if combo_count == 0 {
        return 0.0;
    }

    let mut total = 0.0;
    for combo_idx in 0..combo_count {
        let unblocked = &ctx.unblocked[hand_idx][combo_idx];
        let mut live = 0.0;
        let mut weight = 0.0;
        for (villain_idx, &live_count) in unblocked.iter().enumerate() {
            let live_f = live_count as f64;
            live += live_f;
            weight += live_f * range[villain_idx].clamp(0.0, 1.0);
        }
        if live > 0.0 {
            total += weight / live;
        }
    }
    total / combo_count as f64
}

#[cfg(test)]
pub(crate) fn cfr_max_instantaneous_regret(
    input: &SolverInput,
    cache: &EquityCache,
) -> Option<f64> {
    let mut solver = CfrSolver::init(input, cache)?;
    while solver.iterations_done < input.max_iterations {
        solver.iterate();
        if solver.converged() {
            break;
        }
    }
    solver.recompute_strategy();
    Some(solver.max_instantaneous_regret())
}
