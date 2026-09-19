use crate::solver::{SolverInput, SolverOutput};

pub mod cfr;
pub mod cfr_3max;
pub mod fp;

pub use cfr::CfrSolver;
pub use cfr_3max::Cfr3Max;
pub use fp::FictitiousPlay;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Algorithm {
    #[default]
    FictitiousPlay,
    Cfr,
    Cfr3Max,
}

impl Algorithm {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FictitiousPlay => "fp",
            Self::Cfr => "cfr",
            Self::Cfr3Max => "cfr-3max",
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
            other => Err(format!(
                "unknown algorithm: {other} (expected fp, cfr, or cfr-3max)"
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
        Algorithm::Cfr | Algorithm::Cfr3Max => CfrSolver::run(input, cache),
    }
}
