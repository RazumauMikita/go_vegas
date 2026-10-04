#![forbid(unsafe_code)]

pub mod algorithm;
pub mod bucketing;
pub mod card;
pub mod equity;
pub mod equity_3way;
pub mod equity_cache;
pub mod four_way_rank_cache;
pub mod hand_evaluator;
pub mod hh_parser;
pub mod icm;
pub mod solver;
pub mod three_way_rank_cache;

pub use bucketing::{compute_bucket_map, Bucketing, NUM_BUCKETS};
pub use card::{Card, CardParseError, RANK_CHARS, SUIT_CHARS};
pub use equity::{equity_exact, equity_monte_carlo, parse_range, EquityResult, HandRange};
pub use equity_3way::{
    apply_permutation, equity_3way, equity_3way_icm, equity_3way_places, equity_3way_shares,
    finalize_3way_stacks, icm_showdown_equity, rank_distribution_3way,
    rank_distribution_3way_serial, resolve_3way_showdown,
};
pub use equity_cache::{
    combo_index, combo_label, expand_combo, index_to_ranks, EquityCache, EquityCacheError,
    UNIQUE_PAIR_COUNT,
};
pub use four_way_rank_cache::{
    FourWayRankCache, CACHE_SIZE as FOUR_WAY_CACHE_SIZE, FILE_BYTES as FOUR_WAY_FILE_BYTES,
    NUM_PERMS as FOUR_WAY_NUM_PERMS, STORED_PERMS,
};
pub use hand_evaluator::{evaluate_hand, HandCategory, HandRank};
pub use hh_parser::{
    is_non_push_fold_situation, parse_hand_history, position_label as hh_position_label,
    solver_position_index, ParseError, ParsedHand, ParsedPlayer, Position,
};
pub use icm::{icm_equity, icm_equity_for_player};
pub use solver::{
    debug_bb_report, five_node, six_node, solve, Algorithm, FiveMaxRanges, FourMaxRanges,
    SixMaxRanges, SolverInput, SolverOutput, ThreeMaxHandEvs, ThreeMaxRanges, FIVE_MAX_NODES,
    SIX_MAX_NODES,
};
pub use three_way_rank_cache::{
    RankCacheFillStats, ThreeWayRankCache, THREE_WAY_RANK_CACHE_BYTES, THREE_WAY_RANK_CACHE_SIZE,
};
