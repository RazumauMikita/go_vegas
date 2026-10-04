use crate::solver::{SolverInput, SolverOutput, HAND_TYPES};

pub(crate) fn set_locked_strategy(range: &[f64; HAND_TYPES], dest: &mut [[f64; 2]; HAND_TYPES]) {
    for h in 0..HAND_TYPES {
        let p = range[h].clamp(0.0, 1.0);
        dest[h] = [p, 1.0 - p];
    }
}

pub(crate) fn apply_locked_strategies(
    input: &SolverInput,
    strategy: &mut [[[f64; 2]; HAND_TYPES]],
) {
    for (&id, range) in &input.locked_ranges {
        if let Some(node) = strategy.get_mut(id) {
            set_locked_strategy(range, node);
        }
    }
}

pub(crate) fn overlay_locked_freqs(input: &SolverInput, freq: &mut [[f64; HAND_TYPES]]) {
    for (&id, range) in &input.locked_ranges {
        if let Some(slot) = freq.get_mut(id) {
            *slot = *range;
        }
    }
}

pub(crate) fn initial_locked_range(input: &SolverInput, id: usize, fill: f64) -> [f64; HAND_TYPES] {
    input
        .locked_ranges
        .get(&id)
        .copied()
        .unwrap_or([fill; HAND_TYPES])
}

pub mod cfr;
pub mod cfr_3max;
pub mod cfr_4max;
pub mod cfr_5max;
pub mod cfr_6max;
pub mod cfr_7max;
pub mod cfr_8max;
pub mod cfr_9max;
pub mod fp;

pub use cfr::CfrSolver;
pub use cfr_3max::Cfr3Max;
pub use cfr_4max::Cfr4Max;
pub use cfr_5max::Cfr5Max;
pub use cfr_6max::Cfr6Max;
pub use cfr_7max::Cfr7Max;
pub use cfr_8max::Cfr8Max;
pub use cfr_9max::Cfr9Max;
pub use fp::FictitiousPlay;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Algorithm {
    #[default]
    FictitiousPlay,
    Cfr,
    Cfr3Max,
    Cfr4Max,
    Cfr5Max,
    Cfr6Max,
    Cfr7Max,
    Cfr8Max,
    Cfr9Max,
}

impl Algorithm {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FictitiousPlay => "fp",
            Self::Cfr => "cfr",
            Self::Cfr3Max => "cfr-3max",
            Self::Cfr4Max => "cfr-4max",
            Self::Cfr5Max => "cfr-5max",
            Self::Cfr6Max => "cfr-6max",
            Self::Cfr7Max => "cfr-7max",
            Self::Cfr8Max => "cfr-8max",
            Self::Cfr9Max => "cfr-9max",
        }
    }
}

impl std::str::FromStr for Algorithm {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "fp" => Ok(Self::FictitiousPlay),
            "cfr" => Ok(Self::Cfr),
            "cfr-3max" => Ok(Self::Cfr3Max),
            "cfr-4max" => Ok(Self::Cfr4Max),
            "cfr-5max" => Ok(Self::Cfr5Max),
            "cfr-6max" => Ok(Self::Cfr6Max),
            "cfr-7max" => Ok(Self::Cfr7Max),
            "cfr-8max" => Ok(Self::Cfr8Max),
            "cfr-9max" => Ok(Self::Cfr9Max),
            other => Err(format!(
                "unknown algorithm: {other} (expected fp, cfr, cfr-3max, cfr-4max, cfr-5max, cfr-6max, cfr-7max, cfr-8max, or cfr-9max)"
            )),
        }
    }
}

pub trait SolverAlgorithm {
    fn iterate(&mut self);
    fn converged(&self) -> bool;
    fn result(&self) -> SolverOutput;
}

pub fn run_loop<S: SolverAlgorithm>(mut solver: S, max_iterations: usize) -> SolverOutput {
    for _ in 0..max_iterations {
        solver.iterate();
        if solver.converged() {
            break;
        }
    }
    solver.result()
}

pub fn dispatch(input: &SolverInput, cache: &crate::equity_cache::EquityCache) -> SolverOutput {
    match input.algorithm {
        Algorithm::FictitiousPlay => FictitiousPlay::run(input, cache),
        Algorithm::Cfr
        | Algorithm::Cfr3Max
        | Algorithm::Cfr4Max
        | Algorithm::Cfr5Max
        | Algorithm::Cfr6Max
        | Algorithm::Cfr7Max
        | Algorithm::Cfr8Max
        | Algorithm::Cfr9Max => CfrSolver::run(input, cache),
    }
}
