use crate::solver::{SolverInput, SolverOutput};

pub mod cfr;
pub mod fp;

pub use cfr::CfrSolver;
pub use fp::FictitiousPlay;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Algorithm {
    #[default]
    FictitiousPlay,
    Cfr,
}

impl Algorithm {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FictitiousPlay => "fp",
            Self::Cfr => "cfr",
        }
    }
}

impl std::str::FromStr for Algorithm {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "fp" => Ok(Self::FictitiousPlay),
            "cfr" => Ok(Self::Cfr),
            other => Err(format!("unknown algorithm: {other} (expected fp or cfr)")),
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
        Algorithm::Cfr => CfrSolver::run(input, cache),
    }
}
