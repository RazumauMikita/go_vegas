use crate::card::Card;
use crate::hand_evaluator::{evaluate_hand, HandRank};

const DECK_AFTER_SIX: usize = 46;

/// Chip-share игрока `hand1` в 3-way all-in: P(выигрыш) + 0.5·P(тай на двоих) + 1/3·P(тай на троих).
pub fn equity_3way(hand1: [Card; 2], hand2: [Card; 2], hand3: [Card; 2], iterations: u64) -> f64 {
    equity_3way_shares(hand1, hand2, hand3, iterations)[0]
}

/// Chip-share всех трёх игроков.
pub fn equity_3way_shares(
    hand1: [Card; 2],
    hand2: [Card; 2],
    hand3: [Card; 2],
    iterations: u64,
) -> [f64; 3] {
    if iterations == 0 {
        return [1.0 / 3.0; 3];
    }

    let dead = [hand1[0], hand1[1], hand2[0], hand2[1], hand3[0], hand3[1]];
    if has_duplicate(&dead) {
        return [1.0 / 3.0; 3];
    }

    let available = available_cards(dead);
    let mut rng = Rng::new(0xC3A5_9B17_F4D2_81E9 ^ hash_six(dead));
    let mut indices = [0_u8; DECK_AFTER_SIX];
    for (slot, index) in indices.iter_mut().enumerate() {
        *index = slot as u8;
    }

    let mut shares = [0.0; 3];
    for _ in 0..iterations {
        let board = sample_board(&available, &mut indices, &mut rng);
        let ranks = [
            evaluate_seven(hand1, board),
            evaluate_seven(hand2, board),
            evaluate_seven(hand3, board),
        ];
        let sample = chip_shares(ranks);
        shares[0] += sample[0];
        shares[1] += sample[1];
        shares[2] += sample[2];
    }

    let total = iterations as f64;
    [shares[0] / total, shares[1] / total, shares[2] / total]
}

/// $EV трёх игроков при равных стеках (места 1/2/3 по силе руки).
pub fn equity_3way_places(
    hand1: [Card; 2],
    hand2: [Card; 2],
    hand3: [Card; 2],
    payouts: &[f64],
    iterations: u64,
) -> [f64; 3] {
    if iterations == 0 {
        return split_all(payouts);
    }

    let dead = [hand1[0], hand1[1], hand2[0], hand2[1], hand3[0], hand3[1]];
    if has_duplicate(&dead) {
        return split_all(payouts);
    }

    let available = available_cards(dead);
    let mut rng = Rng::new(0x91D3_E70A_22BC_44F1 ^ hash_six(dead));
    let mut indices = [0_u8; DECK_AFTER_SIX];
    for (slot, index) in indices.iter_mut().enumerate() {
        *index = slot as u8;
    }

    let mut ev = [0.0; 3];
    for _ in 0..iterations {
        let board = sample_board(&available, &mut indices, &mut rng);
        let ranks = [
            evaluate_seven(hand1, board),
            evaluate_seven(hand2, board),
            evaluate_seven(hand3, board),
        ];
        let sample = place_payouts(ranks, payouts);
        ev[0] += sample[0];
        ev[1] += sample[1];
        ev[2] += sample[2];
    }

    let total = iterations as f64;
    [ev[0] / total, ev[1] / total, ev[2] / total]
}

fn split_all(payouts: &[f64]) -> [f64; 3] {
    let sum: f64 = (0..3).map(|i| payouts.get(i).copied().unwrap_or(0.0)).sum();
    [sum / 3.0, sum / 3.0, sum / 3.0]
}

fn chip_shares(ranks: [HandRank; 3]) -> [f64; 3] {
    let best = ranks[0].max(ranks[1]).max(ranks[2]);
    let winners = ranks.iter().filter(|rank| **rank == best).count() as f64;
    let mut shares = [0.0; 3];
    for (index, rank) in ranks.iter().enumerate() {
        if *rank == best {
            shares[index] = 1.0 / winners;
        }
    }
    shares
}

fn place_payouts(ranks: [HandRank; 3], payouts: &[f64]) -> [f64; 3] {
    let mut ev = [0.0; 3];
    for player in 0..3 {
        let better = ranks.iter().filter(|rank| **rank > ranks[player]).count();
        let tied = ranks.iter().filter(|rank| **rank == ranks[player]).count();
        let mut prize = 0.0;
        for offset in 0..tied {
            prize += payouts.get(better + offset).copied().unwrap_or(0.0);
        }
        ev[player] = prize / tied as f64;
    }
    ev
}

fn evaluate_seven(hole: [Card; 2], board: [Card; 5]) -> HandRank {
    evaluate_hand(&[
        hole[0], hole[1], board[0], board[1], board[2], board[3], board[4],
    ])
}

fn has_duplicate(cards: &[Card; 6]) -> bool {
    for i in 0..6 {
        for j in (i + 1)..6 {
            if cards[i] == cards[j] {
                return true;
            }
        }
    }
    false
}

fn available_cards(dead: [Card; 6]) -> [Card; DECK_AFTER_SIX] {
    let mut available = [Card::new(2, 0); DECK_AFTER_SIX];
    let mut slot = 0;
    for suit in 0..4 {
        for rank in Card::MIN_RANK..=Card::MAX_RANK {
            let card = Card::new(rank, suit);
            if dead.contains(&card) {
                continue;
            }
            available[slot] = card;
            slot += 1;
        }
    }
    available
}

fn sample_board(
    available: &[Card; DECK_AFTER_SIX],
    indices: &mut [u8; DECK_AFTER_SIX],
    rng: &mut Rng,
) -> [Card; 5] {
    for slot in 0..5 {
        let remaining = DECK_AFTER_SIX - slot;
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

fn hash_six(cards: [Card; 6]) -> u64 {
    let mut hash = 0_u64;
    for card in cards {
        hash = hash.wrapping_mul(53).wrapping_add(card.rank() as u64);
        hash = hash.wrapping_mul(53).wrapping_add(card.suit() as u64);
    }
    hash
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

    fn hand(a: &str, b: &str) -> [Card; 2] {
        [a.parse().expect("card"), b.parse().expect("card")]
    }

    #[test]
    fn aa_is_ahead_in_three_way() {
        let aa = hand("Ah", "Ad");
        let kk = hand("Kh", "Kd");
        let qq = hand("Qh", "Qd");
        let eq = equity_3way(aa, kk, qq, 4_000);
        assert!(eq > 0.45, "AA 3-way equity {eq}");
        assert!(eq < 0.75, "AA 3-way equity {eq}");
    }

    #[test]
    fn shares_sum_to_one() {
        let shares =
            equity_3way_shares(hand("Ah", "Kd"), hand("Qc", "Js"), hand("7h", "7d"), 3_000);
        let sum = shares[0] + shares[1] + shares[2];
        assert!((sum - 1.0).abs() < 1e-9, "sum {sum}");
    }
}
