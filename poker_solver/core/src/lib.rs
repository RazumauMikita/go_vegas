#![forbid(unsafe_code)]

pub mod card;
pub mod equity;
pub mod equity_3way;
pub mod equity_cache;
pub mod hand_evaluator;
pub mod icm;
pub mod solver;

pub use card::{Card, CardParseError, RANK_CHARS, SUIT_CHARS};
pub use equity::{equity_exact, equity_monte_carlo, parse_range, EquityResult, HandRange};
pub use equity_3way::{equity_3way, equity_3way_places, equity_3way_shares};
pub use equity_cache::{
    combo_index, combo_label, expand_combo, index_to_ranks, EquityCache, UNIQUE_PAIR_COUNT,
};
pub use hand_evaluator::{evaluate_hand, HandCategory, HandRank};
pub use icm::{icm_equity, icm_equity_for_player};
pub use solver::{solve, SolverInput, SolverOutput, ThreeMaxHandEvs, ThreeMaxRanges};
