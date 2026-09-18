//! In-memory кэш 3-way equity для всех 169³ троек типов рук.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

const SENTINEL: u32 = u32::MAX;
const NUM_HANDS: usize = 169;
const CACHE_SIZE: usize = NUM_HANDS * NUM_HANDS * NUM_HANDS;
const CACHE_FILE_BYTES: usize = CACHE_SIZE * 4;

/// Кэш 3-way equity: индекс `btn * 169² + sb * 169 + bb`.
pub struct ThreeWayCache {
    values: Vec<AtomicU32>,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl ThreeWayCache {
    pub fn new() -> Self {
        let mut values = Vec::with_capacity(CACHE_SIZE);
        values.resize_with(CACHE_SIZE, || AtomicU32::new(SENTINEL));
        Self {
            values,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    /// Получить 3-way equity для тройки рук.
    /// Если в кэше нет — вычислить через `compute_fn` и сохранить.
    pub fn get_or_compute<F>(&self, btn_hand: u8, sb_hand: u8, bb_hand: u8, compute_fn: F) -> f64
    where
        F: FnOnce() -> f64,
    {
        let idx = index(btn_hand, sb_hand, bb_hand);
        let cached = self.values[idx].load(Ordering::Acquire);
        if cached != SENTINEL {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return f32::from_bits(cached) as f64;
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        let value = compute_fn();
        let bits = (value as f32).to_bits();
        if bits != SENTINEL {
            match self.values[idx].compare_exchange(
                SENTINEL,
                bits,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => value,
                Err(existing) => f32::from_bits(existing) as f64,
            }
        } else {
            value
        }
    }

    pub fn hit_count(&self) -> usize {
        self.hits.load(Ordering::Relaxed)
    }

    pub fn miss_count(&self) -> usize {
        self.misses.load(Ordering::Relaxed)
    }

    /// Сохранить в файл (4.8M × u32 little-endian, ~19 MB).
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

        let mut values = Vec::with_capacity(CACHE_SIZE);
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
        let first = cache.get_or_compute(3, 7, 11, || {
            calls.fetch_add(1, Ordering::Relaxed);
            0.42
        });
        let second = cache.get_or_compute(3, 7, 11, || {
            calls.fetch_add(1, Ordering::Relaxed);
            0.99
        });

        assert!((first - 0.42).abs() < 1e-6, "first {first}");
        assert!(
            (second - first).abs() < 1e-6,
            "second {second} != first {first}"
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(cache.hit_count(), 1);
        assert_eq!(cache.miss_count(), 1);
    }

    #[test]
    fn cache_save_load_roundtrip() {
        let cache = ThreeWayCache::new();
        let cells = [
            (0_u8, 1_u8, 2_u8, 0.11_f64),
            (5, 9, 14, 0.37),
            (20, 40, 80, 0.58),
            (168, 0, 42, 0.73),
        ];
        for &(btn, sb, bb, value) in &cells {
            let got = cache.get_or_compute(btn, sb, bb, || value);
            assert!((got - value).abs() < 1e-6);
        }

        let path = std::env::temp_dir().join("poker_three_way_cache_roundtrip.bin");
        cache.save(&path).expect("save 3-way cache");
        let loaded = ThreeWayCache::load(&path).expect("load 3-way cache");
        let meta = fs::metadata(&path).expect("cache metadata");
        assert_eq!(meta.len(), CACHE_FILE_BYTES as u64);
        let _ = fs::remove_file(&path);

        for &(btn, sb, bb, value) in &cells {
            let calls = AtomicUsize::new(0);
            let got = loaded.get_or_compute(btn, sb, bb, || {
                calls.fetch_add(1, Ordering::Relaxed);
                0.0
            });
            assert_eq!(calls.load(Ordering::Relaxed), 0, "loaded cell must hit");
            assert!(
                (got - value).abs() < 1e-5,
                "cell ({btn},{sb},{bb}): expected {value}, got {got}"
            );
        }
        assert_eq!(loaded.hit_count(), cells.len());
        assert_eq!(loaded.miss_count(), 0);
    }
}
