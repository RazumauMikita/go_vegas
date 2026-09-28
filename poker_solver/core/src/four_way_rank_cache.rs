//! Кэш 4-way rank-distribution по бакетам: top-5 перестановок + rest.
//! На диске packed `u64` (combinadic 5 индексов + 6×u8 вероятностей).

use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::time::Instant;

use rayon::prelude::*;

use crate::bucketing::{Bucketing, NUM_BUCKETS};
use crate::card::Card;
use crate::equity_cache::expand_combo;
use crate::hand_evaluator::{evaluate_hand, HandRank};
use crate::three_way_rank_cache::RankCacheFillStats;

pub const NUM_PERMS: usize = 24;
pub const STORED_PERMS: usize = 6;
pub const TOP_PERMS: usize = 5;
pub const CACHE_SIZE: usize = NUM_BUCKETS * NUM_BUCKETS * NUM_BUCKETS * NUM_BUCKETS;
pub const CELL_BYTES: usize = 8;
pub const FILE_BYTES: usize = CACHE_SIZE * CELL_BYTES;

const SENTINEL: u64 = u64::MAX;
const DECK_AFTER_EIGHT: usize = 44;
const HOLE_ATTEMPTS: usize = 64;
const UNIFORM: [f64; NUM_PERMS] = [1.0 / NUM_PERMS as f64; NUM_PERMS];
const REST_COUNT: f64 = (NUM_PERMS - TOP_PERMS) as f64;

const RANK_PERMUTATIONS: [[usize; 4]; NUM_PERMS] = [
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [1, 2, 0, 3],
    [1, 2, 3, 0],
    [1, 3, 0, 2],
    [1, 3, 2, 0],
    [2, 0, 1, 3],
    [2, 0, 3, 1],
    [2, 1, 0, 3],
    [2, 1, 3, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
    [3, 0, 1, 2],
    [3, 0, 2, 1],
    [3, 1, 0, 2],
    [3, 1, 2, 0],
    [3, 2, 0, 1],
    [3, 2, 1, 0],
];

pub struct FourWayRankCache {
    values: Vec<AtomicU64>,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl FourWayRankCache {
    pub fn new() -> Self {
        let mut values = Vec::with_capacity(CACHE_SIZE);
        values.resize_with(CACHE_SIZE, || AtomicU64::new(SENTINEL));
        Self {
            values,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    pub fn lookup(&self, utg_b: u8, btn_b: u8, sb_b: u8, bb_b: u8) -> [f64; NUM_PERMS] {
        let packed = self.values[index(utg_b, btn_b, sb_b, bb_b)].load(AtomicOrdering::Acquire);
        if packed == SENTINEL {
            self.misses.fetch_add(1, AtomicOrdering::Relaxed);
            return UNIFORM;
        }
        self.hits.fetch_add(1, AtomicOrdering::Relaxed);
        decompress(packed)
    }

    pub fn store(&self, utg_b: u8, btn_b: u8, sb_b: u8, bb_b: u8, value: [f64; NUM_PERMS]) {
        self.values[index(utg_b, btn_b, sb_b, bb_b)]
            .store(compress(value), AtomicOrdering::Release);
    }

    pub fn is_filled(&self, utg_b: u8, btn_b: u8, sb_b: u8, bb_b: u8) -> bool {
        self.values[index(utg_b, btn_b, sb_b, bb_b)].load(AtomicOrdering::Relaxed) != SENTINEL
    }

    pub fn hit_count(&self) -> usize {
        self.hits.load(AtomicOrdering::Relaxed)
    }

    pub fn miss_count(&self) -> usize {
        self.misses.load(AtomicOrdering::Relaxed)
    }

    pub fn fill_with_progress<F>(
        &self,
        bucketing: &Bucketing,
        samples: u64,
        limit: usize,
        progress: F,
    ) -> RankCacheFillStats
    where
        F: Fn(u64, Instant) + Sync,
    {
        let total = if limit == 0 {
            CACHE_SIZE
        } else {
            limit.min(CACHE_SIZE)
        };
        let combos = bucket_combos(&bucketing.map);
        let valid = AtomicUsize::new(0);
        let invalid = AtomicUsize::new(0);
        let done = AtomicUsize::new(0);
        let started = Instant::now();
        progress(0, started);

        (0..total).into_par_iter().for_each(|idx| {
            let (utg, btn, sb, bb) = decode_index(idx);
            let lists = [
                &combos[utg as usize][..],
                &combos[btn as usize][..],
                &combos[sb as usize][..],
                &combos[bb as usize][..],
            ];
            let seed = 0x4A1C_E5B0_9D27_F813 ^ (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            match rank_distribution_4way(lists, samples, seed) {
                Some(dist) => {
                    self.values[idx].store(compress(dist), AtomicOrdering::Relaxed);
                    valid.fetch_add(1, AtomicOrdering::Relaxed);
                }
                None => {
                    invalid.fetch_add(1, AtomicOrdering::Relaxed);
                }
            }
            let completed = done.fetch_add(1, AtomicOrdering::Relaxed) + 1;
            let step = (total / 20).max(1);
            if completed == total || completed.is_multiple_of(step) {
                progress(completed as u64, started);
            }
        });

        RankCacheFillStats {
            total,
            valid: valid.load(AtomicOrdering::Relaxed),
            invalid: invalid.load(AtomicOrdering::Relaxed),
        }
    }

    pub fn generate_with_progress<F>(
        bucketing: &Bucketing,
        samples: u64,
        limit: usize,
        progress: F,
    ) -> (Self, RankCacheFillStats)
    where
        F: Fn(u64, Instant) + Sync,
    {
        let cache = Self::new();
        let stats = cache.fill_with_progress(bucketing, samples, limit, progress);
        (cache, stats)
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut file = File::create(path)?;
        let mut buf = [0_u8; 4096];
        let mut offset = 0;
        for cell in 0..CACHE_SIZE {
            let bytes = self.values[cell]
                .load(AtomicOrdering::Relaxed)
                .to_le_bytes();
            if offset + CELL_BYTES > buf.len() {
                file.write_all(&buf[..offset])?;
                offset = 0;
            }
            buf[offset..offset + CELL_BYTES].copy_from_slice(&bytes);
            offset += CELL_BYTES;
        }
        if offset > 0 {
            file.write_all(&buf[..offset])?;
        }
        Ok(())
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() != FILE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid 4-way rank cache size: expected {FILE_BYTES} bytes, got {}",
                    bytes.len()
                ),
            ));
        }

        let mut values = Vec::with_capacity(CACHE_SIZE);
        for chunk in bytes.chunks_exact(CELL_BYTES) {
            let encoded: [u8; CELL_BYTES] = chunk.try_into().expect("chunks_exact(8)");
            values.push(AtomicU64::new(u64::from_le_bytes(encoded)));
        }

        Ok(Self {
            values,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        })
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        Self::from_bytes(&bytes)
    }
}

impl Default for FourWayRankCache {
    fn default() -> Self {
        Self::new()
    }
}

fn index(utg: u8, btn: u8, sb: u8, bb: u8) -> usize {
    ((((utg as usize) * NUM_BUCKETS + btn as usize) * NUM_BUCKETS) + sb as usize) * NUM_BUCKETS
        + bb as usize
}

fn decode_index(idx: usize) -> (u8, u8, u8, u8) {
    let bb = idx % NUM_BUCKETS;
    let rem = idx / NUM_BUCKETS;
    let sb = rem % NUM_BUCKETS;
    let rem = rem / NUM_BUCKETS;
    let btn = rem % NUM_BUCKETS;
    let utg = rem / NUM_BUCKETS;
    (utg as u8, btn as u8, sb as u8, bb as u8)
}

struct TypedCombo {
    cards: [Card; 2],
    mask: u64,
}

fn bucket_combos(map: &[u8; 169]) -> Vec<Vec<TypedCombo>> {
    let mut out = Vec::with_capacity(NUM_BUCKETS);
    out.resize_with(NUM_BUCKETS, Vec::new);
    for (hand, &bucket) in map.iter().enumerate() {
        for cards in expand_combo(hand as u8) {
            out[bucket as usize].push(TypedCombo {
                mask: hand_mask(cards),
                cards,
            });
        }
    }
    out
}

fn rank_distribution_4way(
    lists: [&[TypedCombo]; 4],
    samples: u64,
    seed: u64,
) -> Option<[f64; NUM_PERMS]> {
    if samples == 0 {
        return Some(UNIFORM);
    }

    let mut rng = Rng::new(seed);
    let mut counts = [0.0; NUM_PERMS];
    let mut hole_ok = 0_u64;
    let mut fallback = None;

    for _ in 0..samples {
        let holes = match sample_quad(lists, &mut rng) {
            Some(holes) => holes,
            None => {
                if fallback.is_none() {
                    fallback = first_valid_quad(lists);
                }
                match fallback {
                    Some(holes) => holes,
                    None => continue,
                }
            }
        };
        hole_ok += 1;
        let dead = holes[0].mask | holes[1].mask | holes[2].mask | holes[3].mask;
        let available = available_cards(dead);
        let mut indices = [0_u8; DECK_AFTER_EIGHT];
        for (slot, index) in indices.iter_mut().enumerate() {
            *index = slot as u8;
        }
        for _ in 0..samples {
            let board = sample_board(&available, &mut indices, &mut rng);
            let ranks = [
                evaluate_seven(holes[0].cards, board),
                evaluate_seven(holes[1].cards, board),
                evaluate_seven(holes[2].cards, board),
                evaluate_seven(holes[3].cards, board),
            ];
            let weights = rank_permutation_weights(ranks);
            for i in 0..NUM_PERMS {
                counts[i] += weights[i];
            }
        }
    }

    if hole_ok == 0 {
        return None;
    }

    let total = (hole_ok * samples) as f64;
    Some(counts.map(|count| count / total))
}

fn sample_quad<'a>(lists: [&'a [TypedCombo]; 4], rng: &mut Rng) -> Option<[&'a TypedCombo; 4]> {
    if lists.iter().any(|list| list.is_empty()) {
        return None;
    }
    for _ in 0..HOLE_ATTEMPTS {
        let a = &lists[0][rng.gen_range(lists[0].len())];
        let b = &lists[1][rng.gen_range(lists[1].len())];
        if a.mask & b.mask != 0 {
            continue;
        }
        let ab = a.mask | b.mask;
        let c = &lists[2][rng.gen_range(lists[2].len())];
        if ab & c.mask != 0 {
            continue;
        }
        let abc = ab | c.mask;
        let d = &lists[3][rng.gen_range(lists[3].len())];
        if abc & d.mask == 0 {
            return Some([a, b, c, d]);
        }
    }
    None
}

fn first_valid_quad(lists: [&[TypedCombo]; 4]) -> Option<[&TypedCombo; 4]> {
    for a in lists[0] {
        for b in lists[1] {
            if a.mask & b.mask != 0 {
                continue;
            }
            let ab = a.mask | b.mask;
            for c in lists[2] {
                if ab & c.mask != 0 {
                    continue;
                }
                let abc = ab | c.mask;
                for d in lists[3] {
                    if abc & d.mask == 0 {
                        return Some([a, b, c, d]);
                    }
                }
            }
        }
    }
    None
}

fn rank_permutation_weights(ranks: [HandRank; 4]) -> [f64; NUM_PERMS] {
    let mut ok = [false; NUM_PERMS];
    let mut valid = 0.0;
    for (i, order) in RANK_PERMUTATIONS.iter().enumerate() {
        if ranks[order[0]] >= ranks[order[1]]
            && ranks[order[1]] >= ranks[order[2]]
            && ranks[order[2]] >= ranks[order[3]]
        {
            ok[i] = true;
            valid += 1.0;
        }
    }
    if valid == 0.0 {
        return UNIFORM;
    }
    let share = 1.0 / valid;
    let mut out = [0.0; NUM_PERMS];
    for i in 0..NUM_PERMS {
        if ok[i] {
            out[i] = share;
        }
    }
    out
}

fn compress(probs: [f64; NUM_PERMS]) -> u64 {
    let mut order: [usize; NUM_PERMS] = std::array::from_fn(|i| i);
    order.sort_by(|&left, &right| {
        probs[right]
            .partial_cmp(&probs[left])
            .unwrap_or(Ordering::Equal)
            .then(left.cmp(&right))
    });

    let mut top = [0_u8; TOP_PERMS];
    for i in 0..TOP_PERMS {
        top[i] = order[i] as u8;
    }
    top.sort_unstable();

    let mut six = [0.0; STORED_PERMS];
    let mut used = [false; NUM_PERMS];
    for i in 0..TOP_PERMS {
        let perm = top[i] as usize;
        six[i] = probs[perm].max(0.0);
        used[perm] = true;
    }
    for (perm, &keep) in used.iter().enumerate() {
        if !keep {
            six[TOP_PERMS] += probs[perm].max(0.0);
        }
    }

    pack(combination_index(top), encode_probs(six))
}

fn decompress(packed: u64) -> [f64; NUM_PERMS] {
    let (combo, encoded) = unpack(packed);
    let top = combination_from_index(combo);
    let mut used = [false; NUM_PERMS];
    let mut out = [0.0; NUM_PERMS];
    for i in 0..TOP_PERMS {
        let perm = top[i] as usize;
        used[perm] = true;
        out[perm] = f64::from(encoded[i]) / 255.0;
    }
    let each = f64::from(encoded[TOP_PERMS]) / 255.0 / REST_COUNT;
    for (perm, used) in used.iter().enumerate() {
        if !used {
            out[perm] = each;
        }
    }
    out
}

fn encode_probs(probs: [f64; STORED_PERMS]) -> [u8; STORED_PERMS] {
    let mut p = [0.0; STORED_PERMS];
    let mut sum = 0.0;
    for i in 0..STORED_PERMS {
        p[i] = probs[i].max(0.0);
        sum += p[i];
    }
    if sum <= 0.0 {
        return [43, 42, 43, 42, 43, 42];
    }

    let mut scaled = [0.0; STORED_PERMS];
    let mut floors = [0_u8; STORED_PERMS];
    let mut used = 0_u32;
    for i in 0..STORED_PERMS {
        scaled[i] = p[i] / sum * 255.0;
        floors[i] = scaled[i].floor() as u8;
        used += u32::from(floors[i]);
    }

    let mut order = [0_usize, 1, 2, 3, 4, 5];
    order.sort_by(|&a, &b| {
        let ra = scaled[a] - f64::from(floors[a]);
        let rb = scaled[b] - f64::from(floors[b]);
        rb.partial_cmp(&ra)
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(&b))
    });

    let mut remain = 255_u32.saturating_sub(used);
    for idx in order {
        if remain == 0 {
            break;
        }
        floors[idx] = floors[idx].saturating_add(1);
        remain -= 1;
    }
    floors
}

fn pack(combo: u16, probs: [u8; STORED_PERMS]) -> u64 {
    let mut value = u64::from(combo);
    for (i, &prob) in probs.iter().enumerate() {
        value |= u64::from(prob) << (16 + i * 8);
    }
    value
}

fn unpack(packed: u64) -> (u16, [u8; STORED_PERMS]) {
    let combo = packed as u16;
    let mut probs = [0_u8; STORED_PERMS];
    for (i, slot) in probs.iter_mut().enumerate() {
        *slot = (packed >> (16 + i * 8)) as u8;
    }
    (combo, probs)
}

fn combination_index(combo: [u8; TOP_PERMS]) -> u16 {
    let mut idx = 0_u32;
    for (i, &value) in combo.iter().enumerate() {
        idx += binom(u32::from(value), (i + 1) as u32);
    }
    idx as u16
}

fn combination_from_index(idx: u16) -> [u8; TOP_PERMS] {
    let mut combo = [0_u8; TOP_PERMS];
    let mut max = NUM_PERMS as u32;
    let mut remaining = u32::from(idx);
    for slot in (0..TOP_PERMS).rev() {
        let k = (slot + 1) as u32;
        let mut x = max;
        loop {
            x -= 1;
            let b = binom(x, k);
            if remaining >= b {
                remaining -= b;
                combo[slot] = x as u8;
                max = x;
                break;
            }
        }
    }
    combo
}

fn binom(n: u32, k: u32) -> u32 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result = 1_u32;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

fn hand_mask(cards: [Card; 2]) -> u64 {
    card_bit(cards[0]) | card_bit(cards[1])
}

fn card_bit(card: Card) -> u64 {
    1_u64 << ((u32::from(card.rank() - 2) * 4) + u32::from(card.suit()))
}

fn available_cards(dead: u64) -> [Card; DECK_AFTER_EIGHT] {
    let mut available = [Card::new(2, 0); DECK_AFTER_EIGHT];
    let mut slot = 0;
    for suit in 0..4 {
        for rank in Card::MIN_RANK..=Card::MAX_RANK {
            let card = Card::new(rank, suit);
            if dead & card_bit(card) != 0 {
                continue;
            }
            available[slot] = card;
            slot += 1;
        }
    }
    available
}

fn sample_board(
    available: &[Card; DECK_AFTER_EIGHT],
    indices: &mut [u8; DECK_AFTER_EIGHT],
    rng: &mut Rng,
) -> [Card; 5] {
    for slot in 0..5 {
        let remaining = DECK_AFTER_EIGHT - slot;
        let pick = rng.gen_range(remaining) + slot;
        indices.swap(slot, pick);
    }
    [
        available[indices[0] as usize],
        available[indices[1] as usize],
        available[indices[2] as usize],
        available[indices[3] as usize],
        available[indices[4] as usize],
    ]
}

fn evaluate_seven(hole: [Card; 2], board: [Card; 5]) -> HandRank {
    evaluate_hand(&[
        hole[0], hole[1], board[0], board[1], board[2], board[3], board[4],
    ])
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn gen_range(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_u64() as usize) % max
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::Card;
    use crate::equity_cache::{combo_index, EquityCache};
    use std::sync::OnceLock;

    fn test_cache() -> &'static EquityCache {
        static CACHE: OnceLock<EquityCache> = OnceLock::new();
        CACHE.get_or_init(|| {
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
            let candidates = [
                root.join("equity_cache.bin"),
                root.join("poker_ui")
                    .join("assets")
                    .join("equity_cache.bin"),
            ];
            for path in &candidates {
                if let Ok(cache) = EquityCache::load(path) {
                    return cache;
                }
            }
            panic!("equity_cache.bin must exist next to the crate or in poker_ui/assets")
        })
    }

    fn test_bucketing() -> Bucketing {
        Bucketing::new(test_cache())
    }

    fn aa_bucket() -> u8 {
        let aa = combo_index([Card::new(14, 0), Card::new(14, 1)]);
        test_bucketing().bucket_of(aa)
    }

    #[test]
    fn cache_size_correct() {
        assert_eq!(CACHE_SIZE, 6_250_000);
        assert_eq!(NUM_PERMS, 24);
        assert_eq!(STORED_PERMS, 6);
    }

    #[test]
    fn combinadic_roundtrip() {
        let mut combo = [0_u8, 1, 2, 3, 4];
        loop {
            let idx = combination_index(combo);
            assert_eq!(combination_from_index(idx), combo);
            if !next_combination(&mut combo, NUM_PERMS as u8) {
                break;
            }
        }
    }

    fn next_combination(combo: &mut [u8; 5], n: u8) -> bool {
        for i in (0..5).rev() {
            let max = n - (5 - i as u8);
            if combo[i] < max {
                combo[i] += 1;
                for j in i + 1..5 {
                    combo[j] = combo[j - 1] + 1;
                }
                return true;
            }
        }
        false
    }

    #[test]
    fn aa_aa_aa_aa_symmetric() {
        let cache = FourWayRankCache::new();
        let bucket = aa_bucket();
        cache.store(bucket, bucket, bucket, bucket, UNIFORM);
        let dist = cache.lookup(bucket, bucket, bucket, bucket);
        for &prob in &dist {
            assert!(
                (prob - UNIFORM[0]).abs() < 0.01,
                "expected ~{:.4}, got {prob}",
                UNIFORM[0]
            );
        }
    }

    #[test]
    fn sum_probabilities() {
        let cache = FourWayRankCache::new();
        cache.store(
            1,
            2,
            3,
            4,
            [
                0.4, 0.2, 0.1, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.004, 0.003, 0.002, 0.001,
            ],
        );
        let empty = cache.lookup(0, 0, 0, 0);
        let stored = cache.lookup(1, 2, 3, 4);
        for dist in [empty, stored] {
            let sum: f64 = dist.iter().sum();
            assert!((sum - 1.0).abs() <= 0.01, "sum {sum}");
        }
    }

    #[test]
    fn lookup_returns_valid() {
        let cache = FourWayRankCache::new();
        cache.store(
            7,
            8,
            9,
            10,
            [
                0.7, 0.1, 0.05, 0.05, 0.04, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        );
        for dist in [cache.lookup(0, 1, 2, 3), cache.lookup(7, 8, 9, 10)] {
            for &prob in &dist {
                assert!((0.0..=1.0).contains(&prob), "prob {prob}");
            }
        }
    }

    #[test]
    fn save_load_roundtrip() {
        let cache = FourWayRankCache::new();
        let cells = [
            (
                1_u8,
                2_u8,
                3_u8,
                4_u8,
                [
                    0.5, 0.2, 0.1, 0.05, 0.05, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            (10, 20, 30, 40, UNIFORM),
        ];
        let mut stored = Vec::new();
        for &(utg, btn, sb, bb, value) in &cells {
            cache.store(utg, btn, sb, bb, value);
            stored.push(cache.lookup(utg, btn, sb, bb));
        }

        let path = std::env::temp_dir().join("poker_four_way_rank_cache_roundtrip.bin");
        cache.save(&path).expect("save 4-way rank cache");
        let loaded = FourWayRankCache::load(&path).expect("load 4-way rank cache");
        let meta = fs::metadata(&path).expect("cache metadata");
        assert_eq!(meta.len(), FILE_BYTES as u64);
        let _ = fs::remove_file(&path);

        for (idx, &(utg, btn, sb, bb, _)) in cells.iter().enumerate() {
            let got = loaded.lookup(utg, btn, sb, bb);
            for i in 0..NUM_PERMS {
                assert!(
                    (got[i] - stored[idx][i]).abs() < 1e-12,
                    "cell ({utg},{btn},{sb},{bb})[{i}]"
                );
            }
        }
    }

    #[test]
    fn encode_six_sums_to_255() {
        let encoded = encode_probs([0.4, 0.2, 0.15, 0.1, 0.1, 0.05]);
        let sum: u32 = encoded.iter().map(|&x| u32::from(x)).sum();
        assert_eq!(sum, 255);
    }
}
