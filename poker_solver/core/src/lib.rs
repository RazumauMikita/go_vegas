#![forbid(unsafe_code)]

pub mod algorithm;
pub mod card;
pub mod equity;
pub mod equity_3way;
pub mod equity_cache;
pub mod hand_evaluator;
pub mod hh_parser;
pub mod icm;
pub mod solver;
pub mod three_way_rank_cache;

pub use card::{Card, CardParseError, RANK_CHARS, SUIT_CHARS};
pub use equity::{equity_exact, equity_monte_carlo, parse_range, EquityResult, HandRange};
pub use equity_3way::{
    apply_permutation, equity_3way, equity_3way_icm, equity_3way_places, equity_3way_shares,
    finalize_3way_stacks, icm_showdown_equity, rank_distribution_3way, resolve_3way_showdown,
};
pub use equity_cache::{
    combo_index, combo_label, expand_combo, index_to_ranks, EquityCache, EquityCacheError,
    UNIQUE_PAIR_COUNT,
};
pub use hand_evaluator::{evaluate_hand, HandCategory, HandRank};
pub use hh_parser::{
    is_non_push_fold_situation, parse_hand_history, position_label as hh_position_label,
    solver_position_index, ParseError, ParsedHand, ParsedPlayer, Position,
};
pub use icm::{icm_equity, icm_equity_for_player};
pub use solver::{
    debug_bb_report, solve, Algorithm, SolverInput, SolverOutput, ThreeMaxHandEvs, ThreeMaxRanges,
};
pub use three_way_rank_cache::ThreeWayRankCache;
