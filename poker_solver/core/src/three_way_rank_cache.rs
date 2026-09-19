//! Универсальный кэш 3-way rank-distribution: 6 вероятностей перестановок на тройку типов рук.
//! Не зависит от стеков и payouts.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use rayon::prelude::*;

use crate::card::Card;
use crate::equity_3way::rank_distribution_3way_serial;
use crate::equity_cache::expand_combo;

const SENTINEL: u32 = u32::MAX;
const NUM_HANDS: usize = 169;
const NUM_PERMS: usize = 6;
const CACHE_SIZE: usize = NUM_HANDS * NUM_HANDS * NUM_HANDS;
const SLOT_COUNT: usize = CACHE_SIZE * 2;
const CACHE_FILE_BYTES: usize = CACHE_SIZE * NUM_PERMS;
const EMPTY_BYTES: [u8; NUM_PERMS] = [0xFF; NUM_PERMS];

/// Число ячеек кэша (169³).
pub const THREE_WAY_RANK_CACHE_SIZE: usize = CACHE_SIZE;
/// Размер файла кэша в байтах (169³ × 6).
pub const THREE_WAY_RANK_CACHE_BYTES: usize = CACHE_FILE_BYTES;

static EMBEDDED_BYTES: OnceLock<&'static [u8]> = OnceLock::new();

/// Статистика офлайн-заполнения кэша.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankCacheFillStats {
    pub total: usize,
    pub valid: usize,
    pub invalid: usize,
}

/// Кэш 6 вероятностей перестановок: индекс `btn * 169² + sb * 169 + bb`.
pub struct ThreeWayRankCache {
    /// Два `u32` на тройку: `[p0,p1,p2,p3]` и `[p4,p5,0,0]`. Первый `u32::MAX` = пусто.
    values: Vec<AtomicU32>,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl ThreeWayRankCache {
    pub fn new() -> Self {
        let mut values = Vec::with_capacity(SLOT_COUNT);
        values.resize_with(SLOT_COUNT, || AtomicU32::new(SENTINEL));
        Self {
            values,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    /// Встроить байты кэша (например `include_bytes!`) как fallback, если файла нет.
    pub fn install_embedded(bytes: &'static [u8]) {
        let _ = EMBEDDED_BYTES.set(bytes);
    }

    /// Lookup: `Some` если ячейка заполнена. SENTINEL → `None`, miss не считается.
    pub fn get(&self, btn_hand: u8, sb_hand: u8, bb_hand: u8) -> Option<[f64; NUM_PERMS]> {
        let base = index(btn_hand, sb_hand, bb_hand) * 2;
        let cached = self.values[base].load(Ordering::Acquire);
        if cached == SENTINEL {
            return None;
        }
        self.hits.fetch_add(1, Ordering::Relaxed);
        let hi = self.values[base + 1].load(Ordering::Acquire);
        Some(decode(unpack(cached, hi)))
    }

    /// Вероятности P1..P6 для тройки типов рук. Miss считает сразу все шесть и кладёт в кэш.
    pub fn get_or_compute<F>(
        &self,
        btn_hand: u8,
        sb_hand: u8,
        bb_hand: u8,
        compute_fn: F,
    ) -> [f64; NUM_PERMS]
    where
        F: FnOnce() -> [f64; NUM_PERMS],
    {
        match self.get(btn_hand, sb_hand, bb_hand) {
            Some(value) => value,
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                let value = compute_fn();
                self.store_if_absent(btn_hand, sb_hand, bb_hand, value)
            }
        }
    }

    /// Read-only lookup. Strict: SENTINEL → ошибка. Иначе lazy `compute_fn`.
    pub fn lookup<F>(
        &self,
        btn_hand: u8,
        sb_hand: u8,
        bb_hand: u8,
        strict: bool,
        compute_fn: F,
    ) -> [f64; NUM_PERMS]
    where
        F: FnOnce() -> [f64; NUM_PERMS],
    {
        if let Some(value) = self.get(btn_hand, sb_hand, bb_hand) {
            return value;
        }
        if strict {
            panic!(
                "three_way_rank_cache missing entry ({btn_hand},{sb_hand},{bb_hand}); rebuild with build_rank_cache"
            );
        }
        self.get_or_compute(btn_hand, sb_hand, bb_hand, compute_fn)
    }

    /// Записать вероятности в ячейку (lock-free). Невалидные тройки не трогать — SENTINEL.
    pub fn store(&self, btn_hand: u8, sb_hand: u8, bb_hand: u8, value: [f64; NUM_PERMS]) {
        let encoded = encode_probs(value);
        let lo = pack_lo(encoded);
        if lo == SENTINEL {
            return;
        }
        let hi = pack_hi(encoded);
        let base = index(btn_hand, sb_hand, bb_hand) * 2;
        self.values[base + 1].store(hi, Ordering::Relaxed);
        self.values[base].store(lo, Ordering::Release);
    }

    fn store_if_absent(
        &self,
        btn_hand: u8,
        sb_hand: u8,
        bb_hand: u8,
        value: [f64; NUM_PERMS],
    ) -> [f64; NUM_PERMS] {
        let encoded = encode_probs(value);
        let lo = pack_lo(encoded);
        if lo == SENTINEL {
            return decode(encoded);
        }
        let hi = pack_hi(encoded);
        let base = index(btn_hand, sb_hand, bb_hand) * 2;
        self.values[base + 1].store(hi, Ordering::Relaxed);
        match self.values[base].compare_exchange(SENTINEL, lo, Ordering::Release, Ordering::Acquire)
        {
            Ok(_) => decode(encoded),
            Err(current) => {
                let hi = self.values[base + 1].load(Ordering::Acquire);
                decode(unpack(current, hi))
            }
        }
    }

    pub fn is_filled(&self, btn_hand: u8, sb_hand: u8, bb_hand: u8) -> bool {
        let base = index(btn_hand, sb_hand, bb_hand) * 2;
        self.values[base].load(Ordering::Relaxed) != SENTINEL
    }

    pub fn filled_count(&self) -> usize {
        self.filled_in_prefix(CACHE_SIZE)
    }

    pub fn filled_in_prefix(&self, limit: usize) -> usize {
        let limit = limit.min(CACHE_SIZE);
        let mut filled = 0;
        for cell in 0..limit {
            if self.values[cell * 2].load(Ordering::Relaxed) != SENTINEL {
                filled += 1;
            }
        }
        filled
    }

    pub fn hit_count(&self) -> usize {
        self.hits.load(Ordering::Relaxed)
    }

    pub fn miss_count(&self) -> usize {
        self.misses.load(Ordering::Relaxed)
    }

    /// Заполнить первые `limit` троек (или все, если `limit == 0`).
    /// Параллелизм только по индексам; MC samples — sequential.
    pub fn fill_with_progress<F>(
        &self,
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
        let combos = type_combos();
        let valid = AtomicUsize::new(0);
        let invalid = AtomicUsize::new(0);
        let done = AtomicUsize::new(0);
        let started = Instant::now();
        progress(0, started);

        (0..total).into_par_iter().for_each(|idx| {
            let (btn, sb, bb) = decode_index(idx);
            match first_valid_triple(btn, sb, bb, &combos) {
                Some((h1, h2, h3)) => {
                    let dist = rank_distribution_3way_serial(h1, h2, h3, samples);
                    self.store(btn, sb, bb, dist);
                    valid.fetch_add(1, Ordering::Relaxed);
                }
                None => {
                    invalid.fetch_add(1, Ordering::Relaxed);
                }
            }
            let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
            let step = (total / 20).max(1);
            if completed == total || completed.is_multiple_of(step) {
                progress(completed as u64, started);
            }
        });

        RankCacheFillStats {
            total,
            valid: valid.load(Ordering::Relaxed),
            invalid: invalid.load(Ordering::Relaxed),
        }
    }

    pub fn generate_with_progress<F>(
        samples: u64,
        limit: usize,
        progress: F,
    ) -> (Self, RankCacheFillStats)
    where
        F: Fn(u64, Instant) + Sync,
    {
        let cache = Self::new();
        let stats = cache.fill_with_progress(samples, limit, progress);
        (cache, stats)
    }

    /// Сохранить в файл (169³ × 6 u8 little-endian, ~28 MB).
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut file = File::create(path)?;
        let mut buf = [0_u8; 4092];
        let mut offset = 0;
        for cell in 0..CACHE_SIZE {
            let lo = self.values[cell * 2].load(Ordering::Relaxed);
            let bytes = if lo == SENTINEL {
                EMPTY_BYTES
            } else {
                let hi = self.values[cell * 2 + 1].load(Ordering::Relaxed);
                unpack(lo, hi)
            };
            if offset + NUM_PERMS > buf.len() {
                file.write_all(&buf[..offset])?;
                offset = 0;
            }
            buf[offset..offset + NUM_PERMS].copy_from_slice(&bytes);
            offset += NUM_PERMS;
        }
        if offset > 0 {
            file.write_all(&buf[..offset])?;
        }
        Ok(())
    }

    /// Загрузить из байтового слайса (например, через `include_bytes!`).
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() != CACHE_FILE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid 3-way rank cache size: expected {CACHE_FILE_BYTES} bytes, got {}",
                    bytes.len()
                ),
            ));
        }

        let mut values = Vec::with_capacity(SLOT_COUNT);
        for chunk in bytes.chunks_exact(NUM_PERMS) {
            let encoded: [u8; NUM_PERMS] = chunk.try_into().expect("chunks_exact(6)");
            if encoded == EMPTY_BYTES {
                values.push(AtomicU32::new(SENTINEL));
                values.push(AtomicU32::new(SENTINEL));
            } else {
                values.push(AtomicU32::new(pack_lo(encoded)));
                values.push(AtomicU32::new(pack_hi(encoded)));
            }
        }

        Ok(Self {
            values,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        })
    }

    /// Загрузить из файла.
    pub fn load(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Fallback на `install_embedded`, если файла нет.
    pub fn load_or_embedded(path: &Path) -> io::Result<Self> {
        match Self::load(path) {
            Ok(cache) => Ok(cache),
            Err(error) => match EMBEDDED_BYTES.get() {
                Some(bytes) => Self::from_bytes(bytes),
                None => Err(error),
            },
        }
    }
}

impl Default for ThreeWayRankCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Первая конкретная тройка карт данных типов без card-removal конфликта.
pub fn first_disjoint_combos(
    btn_hand: u8,
    sb_hand: u8,
    bb_hand: u8,
) -> Option<([Card; 2], [Card; 2], [Card; 2])> {
    first_valid_triple(btn_hand, sb_hand, bb_hand, &type_combos())
}

fn index(btn: u8, sb: u8, bb: u8) -> usize {
    (btn as usize) * NUM_HANDS * NUM_HANDS + (sb as usize) * NUM_HANDS + (bb as usize)
}

fn decode_index(idx: usize) -> (u8, u8, u8) {
    let btn = idx / (NUM_HANDS * NUM_HANDS);
    let rem = idx % (NUM_HANDS * NUM_HANDS);
    let sb = rem / NUM_HANDS;
    let bb = rem % NUM_HANDS;
    (btn as u8, sb as u8, bb as u8)
}

struct TypedCombo {
    cards: [Card; 2],
    mask: u64,
}

fn type_combos() -> Vec<Vec<TypedCombo>> {
    (0..NUM_HANDS as u8)
        .map(|idx| {
            expand_combo(idx)
                .into_iter()
                .map(|cards| TypedCombo {
                    mask: hand_mask(cards),
                    cards,
                })
                .collect()
        })
        .collect()
}

fn first_valid_triple(
    btn: u8,
    sb: u8,
    bb: u8,
    combos: &[Vec<TypedCombo>],
) -> Option<([Card; 2], [Card; 2], [Card; 2])> {
    let a_list = &combos[btn as usize];
    let b_list = &combos[sb as usize];
    let c_list = &combos[bb as usize];
    for a in a_list {
        for b in b_list {
            if a.mask & b.mask != 0 {
                continue;
            }
            let ab = a.mask | b.mask;
            for c in c_list {
                if ab & c.mask == 0 {
                    return Some((a.cards, b.cards, c.cards));
                }
            }
        }
    }
    None
}

fn hand_mask(cards: [Card; 2]) -> u64 {
    card_bit(cards[0]) | card_bit(cards[1])
}

fn card_bit(card: Card) -> u64 {
    1_u64 << ((u32::from(card.rank() - 2) * 4) + u32::from(card.suit()))
}

fn encode_probs(probs: [f64; NUM_PERMS]) -> [u8; NUM_PERMS] {
    let mut p = [0.0; NUM_PERMS];
    let mut sum = 0.0;
    for i in 0..NUM_PERMS {
        p[i] = probs[i].max(0.0);
        sum += p[i];
    }
    if sum <= 0.0 {
        return [43, 42, 43, 42, 43, 42];
    }

    let mut scaled = [0.0; NUM_PERMS];
    let mut floors = [0_u8; NUM_PERMS];
    let mut used = 0_u32;
    for i in 0..NUM_PERMS {
        scaled[i] = p[i] / sum * 255.0;
        floors[i] = scaled[i].floor() as u8;
        used += u32::from(floors[i]);
    }

    let mut order = [0_usize, 1, 2, 3, 4, 5];
    order.sort_by(|&a, &b| {
        let ra = scaled[a] - f64::from(floors[a]);
        let rb = scaled[b] - f64::from(floors[b]);
        rb.partial_cmp(&ra)
            .unwrap_or(std::cmp::Ordering::Equal)
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

fn decode(bytes: [u8; NUM_PERMS]) -> [f64; NUM_PERMS] {
    bytes.map(|byte| f64::from(byte) / 255.0)
}

fn pack_lo(bytes: [u8; NUM_PERMS]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn pack_hi(bytes: [u8; NUM_PERMS]) -> u32 {
    u32::from_le_bytes([bytes[4], bytes[5], 0, 0])
}

fn unpack(lo: u32, hi: u32) -> [u8; NUM_PERMS] {
    let lo = lo.to_le_bytes();
    let hi = hi.to_le_bytes();
    [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn encode_sums_to_255() {
        let cases = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0 / 6.0; 6],
            [0.5, 0.25, 0.125, 0.125, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        for probs in cases {
            let encoded = encode_probs(probs);
            let sum: u32 = encoded.iter().map(|&x| u32::from(x)).sum();
            assert_eq!(sum, 255, "encoded {encoded:?} from {probs:?}");
            let decoded = decode(encoded);
            let decoded_sum: f64 = decoded.iter().sum();
            assert!(
                (decoded_sum - 1.0).abs() < 1e-12,
                "decoded sum {decoded_sum}"
            );
            for i in 0..6 {
                assert!(
                    (decoded[i] - probs[i]).abs() < 0.005,
                    "component {i}: expected {}, got {}",
                    probs[i],
                    decoded[i]
                );
            }
        }
    }

    #[test]
    fn cache_get_or_compute_returns_same() {
        let cache = ThreeWayRankCache::new();
        let calls = AtomicUsize::new(0);
        let first = cache.get_or_compute(3, 7, 11, || {
            calls.fetch_add(1, Ordering::Relaxed);
            [0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
        });
        let second = cache.get_or_compute(3, 7, 11, || {
            calls.fetch_add(1, Ordering::Relaxed);
            [1.0; 6]
        });

        let first_sum: f64 = first.iter().sum();
        assert!((first_sum - 1.0).abs() < 1e-12, "sum {first_sum}");
        for i in 0..6 {
            assert!(
                (second[i] - first[i]).abs() < 1e-12,
                "second {second:?} != first {first:?}"
            );
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(cache.hit_count(), 1);
        assert_eq!(cache.miss_count(), 1);
    }

    #[test]
    fn cache_save_load_roundtrip() {
        let cache = ThreeWayRankCache::new();
        let cells = [
            (0_u8, 1_u8, 2_u8, [0.5, 0.2, 0.1, 0.1, 0.05, 0.05]),
            (5, 9, 14, [0.1, 0.2, 0.3, 0.15, 0.15, 0.1]),
            (20, 40, 80, [1.0 / 6.0; 6]),
            (168, 0, 42, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ];
        let mut stored = Vec::new();
        for &(btn, sb, bb, value) in &cells {
            stored.push(cache.get_or_compute(btn, sb, bb, || value));
        }

        let path = std::env::temp_dir().join("poker_three_way_rank_cache_roundtrip.bin");
        cache.save(&path).expect("save 3-way rank cache");
        let bytes = fs::read(&path).expect("read cache bytes");
        let loaded = ThreeWayRankCache::from_bytes(&bytes).expect("from_bytes");
        let meta = fs::metadata(&path).expect("cache metadata");
        assert_eq!(meta.len(), CACHE_FILE_BYTES as u64);
        let _ = fs::remove_file(&path);

        for (idx, &(btn, sb, bb, _)) in cells.iter().enumerate() {
            let calls = AtomicUsize::new(0);
            let got = loaded.get_or_compute(btn, sb, bb, || {
                calls.fetch_add(1, Ordering::Relaxed);
                [0.0; 6]
            });
            assert_eq!(calls.load(Ordering::Relaxed), 0, "loaded cell must hit");
            for i in 0..6 {
                assert!(
                    (got[i] - stored[idx][i]).abs() < 1e-12,
                    "cell ({btn},{sb},{bb})[{i}]: expected {}, got {}",
                    stored[idx][i],
                    got[i]
                );
            }
        }
        assert_eq!(loaded.hit_count(), cells.len());
        assert_eq!(loaded.miss_count(), 0);
    }

    #[test]
    fn aa_aa_aa_is_invalid_aa_kk_qq_is_valid() {
        assert!(first_disjoint_combos(0, 0, 0).is_none());
        assert!(first_disjoint_combos(0, 1, 2).is_some());
    }

    #[test]
    fn build_rank_cache_completes() {
        let limit = 1000;
        let cache = ThreeWayRankCache::new();
        let stats = cache.fill_with_progress(2, limit, |_, _| {});
        assert_eq!(stats.total, limit);
        assert_eq!(stats.valid + stats.invalid, limit);
        assert_eq!(cache.filled_in_prefix(limit), stats.valid);
        assert!(
            stats.valid > 0,
            "expected some valid triples in first {limit}"
        );

        let combos = type_combos();
        for idx in 0..limit {
            let (btn, sb, bb) = decode_index(idx);
            let expected = first_valid_triple(btn, sb, bb, &combos).is_some();
            assert_eq!(
                cache.is_filled(btn, sb, bb),
                expected,
                "index {idx} ({btn},{sb},{bb})"
            );
        }
    }

    #[test]
    fn full_cache_read_only() {
        let limit = 64;
        let (cache, stats) = ThreeWayRankCache::generate_with_progress(2, limit, |_, _| {});
        assert!(stats.valid > 0);
        let calls = AtomicUsize::new(0);
        for idx in 0..limit {
            let (btn, sb, bb) = decode_index(idx);
            if !cache.is_filled(btn, sb, bb) {
                continue;
            }
            let _ = cache.get_or_compute(btn, sb, bb, || {
                calls.fetch_add(1, Ordering::Relaxed);
                [0.0; 6]
            });
        }
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(cache.miss_count(), 0);
    }

    #[test]
    #[should_panic(expected = "three_way_rank_cache missing entry")]
    fn strict_lookup_panics_on_sentinel() {
        let cache = ThreeWayRankCache::new();
        let _ = cache.lookup(0, 1, 2, true, || [1.0 / 6.0; 6]);
    }
}
