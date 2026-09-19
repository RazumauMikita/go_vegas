//! In-memory кэш 3-way equity для всех 169³ троек типов рук.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

const SENTINEL: u32 = u32::MAX;
const NUM_HANDS: usize = 169;
const NUM_PLAYERS: usize = 3;
const CACHE_SIZE: usize = NUM_HANDS * NUM_HANDS * NUM_HANDS;
const SLOT_COUNT: usize = CACHE_SIZE * NUM_PLAYERS;
const CACHE_FILE_BYTES: usize = SLOT_COUNT * 4;

/// Кэш 3-way $EV: индекс `(btn * 169² + sb * 169 + bb) * 3 + player`.
pub struct ThreeWayCache {
    values: Vec<AtomicU32>,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl ThreeWayCache {
    pub fn new() -> Self {
        let mut values = Vec::with_capacity(SLOT_COUNT);
        values.resize_with(SLOT_COUNT, || AtomicU32::new(SENTINEL));
        Self {
            values,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    /// $EV игрока `player` (0=BTN, 1=SB, 2=BB) для тройки типов рук.
    /// Miss считает сразу всех троих и кладёт в кэш.
    pub fn get_or_compute<F>(
        &self,
        btn_hand: u8,
        sb_hand: u8,
        bb_hand: u8,
        player: usize,
        compute_fn: F,
    ) -> f64
    where
        F: FnOnce() -> [f64; 3],
    {
        debug_assert!(player < NUM_PLAYERS);
        let base = index(btn_hand, sb_hand, bb_hand) * NUM_PLAYERS;
        let cached = self.values[base].load(Ordering::Acquire);
        if cached != SENTINEL {
            self.hits.fetch_add(1, Ordering::Relaxed);
            let bits = self.values[base + player].load(Ordering::Acquire);
            return f32::from_bits(bits) as f64;
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        let value = compute_fn();
        let bits = [
            (value[0] as f32).to_bits(),
            (value[1] as f32).to_bits(),
            (value[2] as f32).to_bits(),
        ];
        if bits[0] == SENTINEL {
            return value[player];
        }

        self.values[base + 1].store(bits[1], Ordering::Relaxed);
        self.values[base + 2].store(bits[2], Ordering::Relaxed);
        match self.values[base].compare_exchange(
            SENTINEL,
            bits[0],
            Ordering::Release,
            Ordering::Acquire,
        ) {
            Ok(_) => value[player],
            Err(_) => {
                let bits = self.values[base + player].load(Ordering::Acquire);
                f32::from_bits(bits) as f64
            }
        }
    }

    pub fn hit_count(&self) -> usize {
        self.hits.load(Ordering::Relaxed)
    }

    pub fn miss_count(&self) -> usize {
        self.misses.load(Ordering::Relaxed)
    }

    /// Сохранить в файл (4.8M × 3 × u32 little-endian, ~57 MB).
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut file = File::create(path)?;
        let mut buf = [0_u8; 4096];
        let mut offset = 0;
        for cell in &self.values {
            if offset + 4 > buf.len() {
                file.write_all(&buf[..offset])?;
                offset = 0;
            }
            buf[offset..offset + 4].copy_from_slice(&cell.load(Ordering::Relaxed).to_le_bytes());
            offset += 4;
        }
        if offset > 0 {
            file.write_all(&buf[..offset])?;
        }
        Ok(())
    }

    /// Загрузить из файла.
    pub fn load(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.len() != CACHE_FILE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid 3-way cache size: expected {CACHE_FILE_BYTES} bytes, got {}",
                    bytes.len()
                ),
            ));
        }

        let mut values = Vec::with_capacity(SLOT_COUNT);
        for chunk in bytes.chunks_exact(4) {
            let bits = u32::from_le_bytes(chunk.try_into().expect("chunks_exact(4)"));
            values.push(AtomicU32::new(bits));
        }

        Ok(Self {
            values,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        })
    }
}

impl Default for ThreeWayCache {
    fn default() -> Self {
        Self::new()
    }
}

fn index(btn: u8, sb: u8, bb: u8) -> usize {
    (btn as usize) * NUM_HANDS * NUM_HANDS + (sb as usize) * NUM_HANDS + (bb as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn cache_get_or_compute_returns_same() {
        let cache = ThreeWayCache::new();
        let calls = AtomicUsize::new(0);
        let first = cache.get_or_compute(3, 7, 11, 0, || {
            calls.fetch_add(1, Ordering::Relaxed);
            [0.42, 0.33, 0.25]
        });
        let second = cache.get_or_compute(3, 7, 11, 0, || {
            calls.fetch_add(1, Ordering::Relaxed);
            [0.99, 0.99, 0.99]
        });
        let sb = cache.get_or_compute(3, 7, 11, 1, || {
            calls.fetch_add(1, Ordering::Relaxed);
            [0.99, 0.99, 0.99]
        });

        assert!((first - 0.42).abs() < 1e-6, "first {first}");
        assert!(
            (second - first).abs() < 1e-6,
            "second {second} != first {first}"
        );
        assert!((sb - 0.33).abs() < 1e-6, "sb {sb}");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(cache.hit_count(), 2);
        assert_eq!(cache.miss_count(), 1);
    }

    #[test]
    fn cache_save_load_roundtrip() {
        let cache = ThreeWayCache::new();
        let cells = [
            (0_u8, 1_u8, 2_u8, [0.11_f64, 0.41, 0.48]),
            (5, 9, 14, [0.37, 0.22, 0.41]),
            (20, 40, 80, [0.58, 0.19, 0.23]),
            (168, 0, 42, [0.73, 0.12, 0.15]),
        ];
        for &(btn, sb, bb, value) in &cells {
            let got = cache.get_or_compute(btn, sb, bb, 0, || value);
            assert!((got - value[0]).abs() < 1e-6);
        }

        let path = std::env::temp_dir().join("poker_three_way_cache_roundtrip.bin");
        cache.save(&path).expect("save 3-way cache");
        let loaded = ThreeWayCache::load(&path).expect("load 3-way cache");
        let meta = fs::metadata(&path).expect("cache metadata");
        assert_eq!(meta.len(), CACHE_FILE_BYTES as u64);
        let _ = fs::remove_file(&path);

        for &(btn, sb, bb, value) in &cells {
            for player in 0..3 {
                let calls = AtomicUsize::new(0);
                let got = loaded.get_or_compute(btn, sb, bb, player, || {
                    calls.fetch_add(1, Ordering::Relaxed);
                    [0.0, 0.0, 0.0]
                });
                assert_eq!(calls.load(Ordering::Relaxed), 0, "loaded cell must hit");
                assert!(
                    (got - value[player]).abs() < 1e-5,
                    "cell ({btn},{sb},{bb},{player}): expected {}, got {got}",
                    value[player]
                );
            }
        }
        assert_eq!(loaded.hit_count(), cells.len() * 3);
        assert_eq!(loaded.miss_count(), 0);
    }
}
