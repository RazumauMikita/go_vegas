use crate::solver::{SolverInput, SolverOutput};

pub mod cfr;
pub mod cfr_3max;
pub mod cfr_4max;
pub mod fp;

pub use cfr::CfrSolver;
pub use cfr_3max::Cfr3Max;
pub use cfr_4max::Cfr4Max;
pub use fp::FictitiousPlay;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Algorithm {
    #[default]
    FictitiousPlay,
    Cfr,
    Cfr3Max,
    Cfr4Max,
}

impl Algorithm {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FictitiousPlay => "fp",
            Self::Cfr => "cfr",
            Self::Cfr3Max => "cfr-3max",
            Self::Cfr4Max => "cfr-4max",
        }
    }
}

impl std::str::FromStr for Algorithm {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "fp" => Ok(Self::FictitiousPlay),
            "cfr" => Ok(Self::Cfr),
            "cfr-3max" => Ok(Self::Cfr3Max),
            "cfr-4max" => Ok(Self::Cfr4Max),
            other => Err(format!(
                "unknown algorithm: {other} (expected fp, cfr, cfr-3max, or cfr-4max)"
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
        Algorithm::Cfr | Algorithm::Cfr3Max | Algorithm::Cfr4Max => CfrSolver::run(input, cache),
    }
}
