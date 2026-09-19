use crate::algorithm::{run_loop, SolverAlgorithm};
use crate::equity_cache::EquityCache;
use crate::solver::{
    average_sb_equity, best_response, build_hu_output, empty_output, ev_call_bb, ev_push_sb,
    HuContext, SolverInput, SolverOutput, HAND_TYPES,
};

pub struct FictitiousPlay<'a> {
    input: SolverInput,
    cache: &'a EquityCache,
    ctx: HuContext,
    push_range: [f64; HAND_TYPES],
    call_range: [f64; HAND_TYPES],
    push_sum: [f64; HAND_TYPES],
    call_sum: [f64; HAND_TYPES],
    iterations_done: usize,
    has_converged: bool,
}

impl<'a> FictitiousPlay<'a> {
    pub fn init(input: &SolverInput, cache: &'a EquityCache) -> Option<Self> {
        let ctx = HuContext::new(input)?;
        Some(Self {
            input: input.clone(),
            cache,
            ctx,
            push_range: [1.0; HAND_TYPES],
            call_range: [1.0; HAND_TYPES],
            push_sum: [0.0; HAND_TYPES],
            call_sum: [0.0; HAND_TYPES],
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
}

impl SolverAlgorithm for FictitiousPlay<'_> {
    fn iterate(&mut self) {
        self.iterations_done += 1;

        let mut br_push = [0.0; HAND_TYPES];
        let mut br_call = [0.0; HAND_TYPES];

        for hand_idx in 0..HAND_TYPES {
            let ev_push = ev_push_sb(hand_idx, &self.call_range, &self.ctx, self.cache);
            br_push[hand_idx] =
                best_response(ev_push, self.ctx.ev_sb_fold, self.push_range[hand_idx]);
        }

        for hand_idx in 0..HAND_TYPES {
            let ev_call = ev_call_bb(hand_idx, &self.push_range, &self.ctx, self.cache);
            br_call[hand_idx] =
                best_response(ev_call, self.ctx.ev_bb_fold, self.call_range[hand_idx]);
        }

        let mut change = 0.0;
        let t = self.iterations_done as f64;
        for hand_idx in 0..HAND_TYPES {
            self.push_sum[hand_idx] += br_push[hand_idx];
            self.call_sum[hand_idx] += br_call[hand_idx];
            let next_push = self.push_sum[hand_idx] / t;
            let next_call = self.call_sum[hand_idx] / t;
            change += (next_push - self.push_range[hand_idx]).abs();
            change += (next_call - self.call_range[hand_idx]).abs();
            self.push_range[hand_idx] = next_push;
            self.call_range[hand_idx] = next_call;
        }

        let mean_change = change / (2.0 * HAND_TYPES as f64);
        if change < self.input.tolerance || mean_change < self.input.tolerance {
            self.has_converged = true;
        }
    }

    fn converged(&self) -> bool {
        self.has_converged
    }

    fn result(&self) -> SolverOutput {
        let sb_equity =
            average_sb_equity(&self.push_range, &self.call_range, &self.ctx, self.cache);
        build_hu_output(
            &self.ctx,
            self.cache,
            &self.push_range,
            &self.call_range,
            sb_equity,
            self.iterations_done,
            self.has_converged,
        )
    }
}
