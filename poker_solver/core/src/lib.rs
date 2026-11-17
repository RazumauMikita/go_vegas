#![forbid(unsafe_code)]

use std::path::PathBuf;

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
    is_non_push_fold_situation, parse_hand_histories, parse_hand_history, parse_imported_hand,
    position_label as hh_position_label, solver_position_index, solver_seat_label,
    split_hand_histories, table_position_label, HeroActionKind, ImportedAction, ImportedHand,
    ParseError, ParsedHand, ParsedPlayer, Position,
};
pub use icm::{icm_equity, icm_equity_for_player};
pub use solver::{
    debug_bb_report, eight_node, five_node, nine_node, seven_node, six_node, solve, Algorithm,
    EightMaxRanges, FiveMaxRanges, FourMaxRanges, NineMaxRanges, SevenMaxRanges, SixMaxRanges,
    SolverInput, SolverOutput, ThreeMaxHandEvs, ThreeMaxRanges, EIGHT_MAX_NODES, FIVE_MAX_NODES,
    NINE_MAX_NODES, SEVEN_MAX_NODES, SIX_MAX_NODES,
};
pub use three_way_rank_cache::{
    RankCacheFillStats, ThreeWayRankCache, THREE_WAY_RANK_CACHE_BYTES, THREE_WAY_RANK_CACHE_SIZE,
};

pub fn find_cache_file(name: &str) -> Option<PathBuf> {
    let mut candidates = Vec::with_capacity(8);
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(name));
            if let Some(target) = dir.parent() {
                candidates.push(target.join(name));
                if let Some(root) = target.parent() {
                    candidates.push(root.join(name));
                    candidates.push(root.join("poker_ui").join("assets").join(name));
                }
            }
        }
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    candidates.push(manifest.join(name));
    candidates.push(manifest.join("..").join(name));
    candidates.push(
        manifest
            .join("..")
            .join("poker_ui")
            .join("assets")
            .join(name),
    );
    candidates.push(PathBuf::from(name));
    candidates.into_iter().find(|path| path.is_file())
}

#[cfg(test)]
mod cache_path_tests {
    #[test]
    fn finds_project_solver_caches() {
        for name in [
            "equity_cache.bin",
            "bucketing.bin",
            "three_way_rank_cache.bin",
            "four_way_rank_cache.bin",
        ] {
            let path = super::find_cache_file(name).unwrap_or_else(|| panic!("{name}"));
            assert!(path.is_file(), "{name}");
        }
    }
}
