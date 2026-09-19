use std::sync::atomic::{AtomicU64, Ordering};

static ICM_EQUITY_CALLS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn icm_equity_call_count() -> u64 {
    ICM_EQUITY_CALLS.load(Ordering::Relaxed)
}

pub(crate) fn reset_icm_equity_call_count() {
    ICM_EQUITY_CALLS.store(0, Ordering::Relaxed);
}

type IcmKey = (i64, i64, i64, u8);

/// In-memory ICM memoization for one solve run.
/// Prefill once, then lock-free lookups. Showdown ICM has only a handful of
/// unique stack triples, so a shared map would contend under rayon.
pub struct IcmCache {
    entries: Vec<(IcmKey, Vec<f64>)>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl IcmCache {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    fn key(stacks: &[f64]) -> IcmKey {
        let round = |index: usize| {
            stacks
                .get(index)
                .map(|&stack| (stack * 100.0).round() as i64)
                .unwrap_or(i64::MIN)
        };
        (round(0), round(1), round(2), stacks.len() as u8)
    }

    pub fn insert(&mut self, stacks: &[f64], payouts: &[f64]) {
        let key = Self::key(stacks);
        if self.entries.iter().any(|(existing, _)| *existing == key) {
            return;
        }
        self.entries.push((key, icm_equity(stacks, payouts)));
    }

    pub fn remember_showdown(&mut self, new_stacks: [f64; 3], payouts: &[f64]) {
        let alive: Vec<usize> = (0..3).filter(|&i| new_stacks[i] > 0.0).collect();
        if alive.len() == 3 {
            self.insert(&new_stacks, payouts);
            return;
        }
        if alive.len() != 2 {
            return;
        }
        let remaining: Vec<f64> = (0..alive.len())
            .map(|place| payouts.get(place).copied().unwrap_or(0.0))
            .collect();
        let remaining_sum: f64 = remaining.iter().sum();
        if remaining_sum <= 0.0 {
            return;
        }
        let alive_stacks: Vec<f64> = alive.iter().map(|&i| new_stacks[i]).collect();
        let normalized: Vec<f64> = remaining
            .iter()
            .map(|payout| payout / remaining_sum)
            .collect();
        self.insert(&alive_stacks, &normalized);
    }

    pub fn lookup(&self, stacks: &[f64], payouts: &[f64]) -> Vec<f64> {
        let key = Self::key(stacks);
        for (existing, value) in &self.entries {
            if *existing == key {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return value.clone();
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        icm_equity(stacks, payouts)
    }

    pub fn get(&self, stacks: [f64; 3], payouts: &[f64], player_idx: usize) -> f64 {
        self.lookup(&stacks, payouts)[player_idx]
    }

    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }
}

impl Default for IcmCache {
    fn default() -> Self {
        Self::new()
    }
}

/// ICM-калькулятор по модели Malmuth-Harville.
///
/// Вероятность занять k-е место для игрока i считается рекурсивно:
/// P(i, 1) = s_i / Σs
/// P(i, k) = Σ_{j≠i} P(j, 1) · P(i, k-1 | j выбыл)
pub fn icm_equity(stacks: &[f64], payouts: &[f64]) -> Vec<f64> {
    ICM_EQUITY_CALLS.fetch_add(1, Ordering::Relaxed);
    if stacks.is_empty() {
        return Vec::new();
    }

    assert!(
        stacks.iter().all(|&stack| stack >= 0.0),
        "stacks must be non-negative"
    );
    assert!(
        stacks.iter().any(|&stack| stack > 0.0),
        "at least one stack must be positive"
    );
    assert!(
        !payouts.is_empty(),
        "payouts must contain at least one place"
    );

    let payout_sum: f64 = payouts.iter().sum();
    assert!(
        (payout_sum - 1.0).abs() < 1e-9,
        "payouts must sum to 1.0, got {payout_sum}"
    );

    let mut equity = vec![0.0; stacks.len()];

    let active_players: Vec<usize> = stacks
        .iter()
        .enumerate()
        .filter(|(_, &stack)| stack > 0.0)
        .map(|(index, _)| index)
        .collect();

    let place_count = active_players.len().min(payouts.len());
    let active_payout_sum: f64 = payouts[..place_count].iter().sum();
    let active_payouts: Vec<f64> = payouts[..place_count]
        .iter()
        .map(|payout| payout / active_payout_sum)
        .collect();

    for &player in &active_players {
        for (place_index, &payout) in active_payouts.iter().enumerate() {
            let rank = place_index + 1;
            let probability = finish_probability(stacks, player, rank, &active_players);
            equity[player] += probability * payout;
        }
    }

    equity
}

/// $EV одного игрока по индексу.
pub fn icm_equity_for_player(stacks: &[f64], payouts: &[f64], player_idx: usize) -> f64 {
    assert!(
        player_idx < stacks.len(),
        "player_idx out of bounds: {player_idx}"
    );

    icm_equity(stacks, payouts)[player_idx]
}

fn finish_probability(stacks: &[f64], player: usize, rank: usize, alive: &[usize]) -> f64 {
    if rank == 0 || rank > alive.len() {
        return 0.0;
    }

    debug_assert!(alive.contains(&player));

    if rank == 1 {
        let total: f64 = alive.iter().map(|&index| stacks[index]).sum();
        debug_assert!(total > 0.0);
        return stacks[player] / total;
    }

    let total: f64 = alive.iter().map(|&index| stacks[index]).sum();
    let mut probability = 0.0;

    for &first in alive {
        if first == player {
            continue;
        }

        let first_probability = stacks[first] / total;
        let remaining: Vec<usize> = alive
            .iter()
            .copied()
            .filter(|&index| index != first)
            .collect();
        probability += first_probability * finish_probability(stacks, player, rank - 1, &remaining);
    }

    probability
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() < tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn sum_of_ev_equals_one() {
        let cases = [
            (
                [5000.0, 3000.0, 2000.0].as_slice(),
                [0.5, 0.3, 0.2].as_slice(),
            ),
            ([1200.0, 800.0].as_slice(), [0.65, 0.35].as_slice()),
            (
                [
                    1000.0, 900.0, 800.0, 700.0, 600.0, 500.0, 400.0, 300.0, 200.0,
                ]
                .as_slice(),
                [0.5, 0.3, 0.2].as_slice(),
            ),
        ];

        for (stacks, payouts) in cases {
            let evs = icm_equity(stacks, payouts);
            let sum: f64 = evs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "expected sum 1.0 for stacks {stacks:?}, got {sum}"
            );
        }
    }

    #[test]
    fn zero_stack_has_zero_ev() {
        let stacks = [1000.0, 0.0, 1000.0];
        let payouts = [0.5, 0.3, 0.2];
        let evs = icm_equity(&stacks, &payouts);

        assert_eq!(evs[1], 0.0);
        assert!((evs[0] + evs[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn payout_structure_matters() {
        let stacks = [5000.0, 3000.0, 2000.0];
        let payouts = [1.0];
        let evs = icm_equity(&stacks, &payouts);

        assert!((evs[0] - 0.5).abs() < 1e-9);
        assert!((evs[1] - 0.3).abs() < 1e-9);
        assert!((evs[2] - 0.2).abs() < 1e-9);
    }

    #[test]
    fn heads_up_equal_stacks_split_prize_pool() {
        let stacks = [1000.0, 1000.0];
        let payouts = [1.0];
        let equity = icm_equity(&stacks, &payouts);

        assert_close(equity[0], 0.5, 1e-9);
        assert_close(equity[1], 0.5, 1e-9);
        assert_close(equity.iter().sum(), 1.0, 1e-9);
    }

    #[test]
    fn three_way_equal_stacks_use_full_payout_ladder() {
        let stacks = [1000.0, 1000.0, 1000.0];
        let payouts = [0.5, 0.3, 0.2];
        let equity = icm_equity(&stacks, &payouts);

        for value in &equity {
            assert_close(*value, 1.0 / 3.0, 1e-9);
        }
        assert_close(equity.iter().sum(), 1.0, 1e-9);
    }

    #[test]
    fn icm_favors_big_stack_on_bubble() {
        let stacks = [5000.0, 3000.0, 2000.0];
        let payouts = [0.65, 0.35];
        let equity = icm_equity(&stacks, &payouts);

        assert!(equity[0] > equity[1]);
        assert!(equity[1] > equity[2]);
        assert_close(equity.iter().sum(), 1.0, 1e-9);
    }

    #[test]
    fn icm_equity_for_player_matches_vector_entry() {
        let stacks = [4000.0, 3500.0, 2500.0];
        let payouts = [0.5, 0.3, 0.2];
        let equity = icm_equity(&stacks, &payouts);

        assert_close(
            icm_equity_for_player(&stacks, &payouts, 1),
            equity[1],
            1e-12,
        );
    }

    #[test]
    fn icm_cache_matches_direct_and_counts_hits() {
        let mut cache = IcmCache::new();
        let stacks = [2000.0, 500.0, 900.0];
        let payouts = [0.5, 0.3, 0.2];
        let direct = icm_equity(&stacks, &payouts);
        cache.insert(&stacks, &payouts);
        let first = cache.get(stacks, &payouts, 1);
        let second = cache.get(stacks, &payouts, 1);

        assert_close(first, direct[1], 1e-12);
        assert_close(second, direct[1], 1e-12);
        assert_eq!(cache.misses(), 0);
        assert_eq!(cache.hits(), 2);
    }

    #[test]
    fn more_players_than_payouts_only_top_places_are_paid() {
        let stacks = [1000.0, 1000.0, 1000.0, 1000.0];
        let payouts = [0.6, 0.4];
        let equity = icm_equity(&stacks, &payouts);

        assert_close(equity.iter().sum(), 1.0, 1e-9);
        for value in &equity {
            assert!(*value > 0.0);
        }
    }
}
