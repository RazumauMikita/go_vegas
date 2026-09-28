//! Сжатие 169 типов рук в 50 бакетов по среднему equity vs random.

use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use crate::equity_cache::EquityCache;

pub const NUM_BUCKETS: usize = 50;

const HAND_COUNT: usize = 169;
const OPPONENTS: f64 = (HAND_COUNT - 1) as f64;
const FILE_BYTES: usize = HAND_COUNT + NUM_BUCKETS + NUM_BUCKETS * 4;

pub fn compute_bucket_map(cache: &EquityCache) -> [u8; HAND_COUNT] {
    assign_buckets(&avg_equity_table(cache))
}

#[derive(Debug, Clone, PartialEq)]
pub struct Bucketing {
    pub map: [u8; HAND_COUNT],
    pub bucket_sizes: [u8; NUM_BUCKETS],
    pub bucket_avg_equity: [f32; NUM_BUCKETS],
}

impl Bucketing {
    pub fn new(cache: &EquityCache) -> Self {
        let hand_eq = avg_equity_table(cache);
        let map = assign_buckets(&hand_eq);
        let mut bucket_sizes = [0_u8; NUM_BUCKETS];
        let mut eq_sum = [0.0_f64; NUM_BUCKETS];

        for (hand, &bucket) in map.iter().enumerate() {
            let bucket = bucket as usize;
            bucket_sizes[bucket] += 1;
            eq_sum[bucket] += hand_eq[hand];
        }

        let mut bucket_avg_equity = [0.0_f32; NUM_BUCKETS];
        for bucket in 0..NUM_BUCKETS {
            let size = bucket_sizes[bucket];
            if size > 0 {
                bucket_avg_equity[bucket] = (eq_sum[bucket] / f64::from(size)) as f32;
            }
        }

        Self {
            map,
            bucket_sizes,
            bucket_avg_equity,
        }
    }

    pub fn bucket_of(&self, hand: u8) -> u8 {
        self.map[hand as usize]
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut file = File::create(path)?;
        file.write_all(&self.map)?;
        file.write_all(&self.bucket_sizes)?;
        for value in &self.bucket_avg_equity {
            file.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.len() != FILE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid bucketing size: expected {FILE_BYTES} bytes, got {}",
                    bytes.len()
                ),
            ));
        }

        let mut map = [0_u8; HAND_COUNT];
        map.copy_from_slice(&bytes[..HAND_COUNT]);
        if map.iter().any(|&bucket| bucket as usize >= NUM_BUCKETS) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bucket id out of range",
            ));
        }

        let mut bucket_sizes = [0_u8; NUM_BUCKETS];
        bucket_sizes.copy_from_slice(&bytes[HAND_COUNT..HAND_COUNT + NUM_BUCKETS]);

        let mut computed_sizes = [0_u8; NUM_BUCKETS];
        for &bucket in &map {
            computed_sizes[bucket as usize] += 1;
        }
        if computed_sizes != bucket_sizes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bucket_sizes does not match map",
            ));
        }

        let mut bucket_avg_equity = [0.0_f32; NUM_BUCKETS];
        let equity_bytes = &bytes[HAND_COUNT + NUM_BUCKETS..];
        for (bucket, chunk) in equity_bytes.chunks_exact(4).enumerate() {
            let encoded: [u8; 4] = chunk.try_into().expect("chunks_exact(4)");
            bucket_avg_equity[bucket] = f32::from_le_bytes(encoded);
        }

        Ok(Self {
            map,
            bucket_sizes,
            bucket_avg_equity,
        })
    }
}

fn avg_equity_table(cache: &EquityCache) -> [f64; HAND_COUNT] {
    let mut avg = [0.0_f64; HAND_COUNT];
    for hero in 0..HAND_COUNT as u8 {
        let mut sum = 0.0;
        for opponent in 0..HAND_COUNT as u8 {
            if opponent != hero {
                sum += cache.equity(hero, opponent);
            }
        }
        avg[hero as usize] = sum / OPPONENTS;
    }
    avg
}

fn assign_buckets(hand_eq: &[f64; HAND_COUNT]) -> [u8; HAND_COUNT] {
    let mut ranked: Vec<u8> = (0..HAND_COUNT as u8).collect();
    ranked.sort_by(|&left, &right| {
        hand_eq[left as usize]
            .partial_cmp(&hand_eq[right as usize])
            .unwrap_or(Ordering::Equal)
            .then(left.cmp(&right))
    });

    let mut map = [0_u8; HAND_COUNT];
    for bucket in 0..NUM_BUCKETS {
        let start = bucket * HAND_COUNT / NUM_BUCKETS;
        let end = (bucket + 1) * HAND_COUNT / NUM_BUCKETS;
        for &hand in &ranked[start..end] {
            map[hand as usize] = bucket as u8;
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::Card;
    use crate::equity_cache::{combo_index, combo_label};
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

    fn fixture() -> Bucketing {
        Bucketing::new(test_cache())
    }

    #[test]
    fn buckets_cover_all_hands() {
        let bucketing = fixture();
        let sum: usize = bucketing
            .bucket_sizes
            .iter()
            .map(|&size| size as usize)
            .sum();
        assert_eq!(sum, HAND_COUNT);
        assert!(bucketing
            .bucket_sizes
            .iter()
            .all(|&size| size == 3 || size == 4));
    }

    #[test]
    fn monotonic_equity() {
        let bucketing = fixture();
        for bucket in 0..NUM_BUCKETS - 1 {
            assert!(
                bucketing.bucket_avg_equity[bucket] < bucketing.bucket_avg_equity[bucket + 1],
                "bucket {bucket} eq {} >= bucket {} eq {}",
                bucketing.bucket_avg_equity[bucket],
                bucket + 1,
                bucketing.bucket_avg_equity[bucket + 1]
            );
        }
    }

    #[test]
    fn save_load_roundtrip() {
        let original = fixture();
        let path = std::env::temp_dir().join("poker_bucketing_roundtrip.bin");
        original.save(&path).expect("save bucketing");
        let loaded = Bucketing::load(&path).expect("load bucketing");
        assert_eq!(original, loaded);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn aa_in_top_bucket() {
        let aa = combo_index([Card::new(14, 0), Card::new(14, 1)]);
        assert_eq!(combo_label(aa), "AA");
        assert_eq!(fixture().bucket_of(aa), (NUM_BUCKETS - 1) as u8);
    }

    #[test]
    fn weakest_hand_in_bucket_zero() {
        let trey_deuce = combo_index([Card::new(3, 0), Card::new(2, 1)]);
        assert_eq!(combo_label(trey_deuce), "32o");
        assert_eq!(fixture().bucket_of(trey_deuce), 0);
    }
}
