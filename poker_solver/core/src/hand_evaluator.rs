use crate::card::Card;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HandRank(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandCategory {
    HighCard = 0,
    OnePair = 1,
    TwoPair = 2,
    ThreeOfAKind = 3,
    Straight = 4,
    Flush = 5,
    FullHouse = 6,
    FourOfAKind = 7,
    StraightFlush = 8,
}

impl HandRank {
    pub fn category(self) -> HandCategory {
        match self.0 >> 52 {
            0 => HandCategory::HighCard,
            1 => HandCategory::OnePair,
            2 => HandCategory::TwoPair,
            3 => HandCategory::ThreeOfAKind,
            4 => HandCategory::Straight,
            5 => HandCategory::Flush,
            6 => HandCategory::FullHouse,
            7 => HandCategory::FourOfAKind,
            8 => HandCategory::StraightFlush,
            _ => unreachable!("invalid hand category encoding"),
        }
    }

    pub fn value(self) -> u64 {
        self.0
    }
}

pub fn evaluate_hand(cards: &[Card; 7]) -> HandRank {
    let mut best = HandRank(0);

    for combo in five_card_combinations(cards) {
        let rank = evaluate_five(&combo);
        if rank > best {
            best = rank;
        }
    }

    best
}

fn evaluate_five(cards: &[Card; 5]) -> HandRank {
    let mut ranks = [0u8; 5];
    let mut suits = [0u8; 5];

    for (index, card) in cards.iter().enumerate() {
        ranks[index] = card.rank();
        suits[index] = card.suit();
    }

    ranks.sort_unstable_by(|a, b| b.cmp(a));

    let is_flush = suits[0] == suits[1]
        && suits[1] == suits[2]
        && suits[2] == suits[3]
        && suits[3] == suits[4];

    let straight_high = straight_high_rank(&ranks);
    let is_straight = straight_high.is_some();

    if is_flush && is_straight {
        return encode(HandCategory::StraightFlush, &[straight_high.unwrap()]);
    }

    let mut counts = [0u8; 15];
    for rank in ranks {
        counts[rank as usize] += 1;
    }

    let mut groups: Vec<(u8, u8)> = (2..=14)
        .filter(|&rank| counts[rank as usize] > 0)
        .map(|rank| (counts[rank as usize], rank))
        .collect();

    groups.sort_unstable_by(|a, b| b.cmp(a).then_with(|| b.1.cmp(&a.1)));

    let pattern: Vec<u8> = groups.iter().map(|(count, _)| *count).collect();
    let ordered_ranks: Vec<u8> = groups.iter().map(|(_, rank)| *rank).collect();

    match pattern.as_slice() {
        [4, 1] => encode(HandCategory::FourOfAKind, &ordered_ranks),
        [3, 2] => encode(HandCategory::FullHouse, &ordered_ranks),
        [3, 1, 1] => encode(HandCategory::ThreeOfAKind, &ordered_ranks),
        [2, 2, 1] => encode(HandCategory::TwoPair, &ordered_ranks),
        [2, 1, 1, 1] => encode(HandCategory::OnePair, &ordered_ranks),
        _ if is_flush => encode(HandCategory::Flush, &ranks),
        _ if is_straight => encode(HandCategory::Straight, &[straight_high.unwrap()]),
        _ => encode(HandCategory::HighCard, &ranks),
    }
}

fn straight_high_rank(sorted_desc: &[u8; 5]) -> Option<u8> {
    let unique: Vec<u8> = sorted_desc
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .rev()
        .collect();

    if unique.len() < 5 {
        return None;
    }

    for window in unique.windows(5) {
        if window[0] - window[4] == 4 {
            return Some(window[0]);
        }
    }

    if unique.ends_with(&[5, 4, 3, 2]) && unique.contains(&14) {
        return Some(5);
    }

    None
}

fn encode(category: HandCategory, kickers: &[u8]) -> HandRank {
    let mut value = (category as u64) << 52;

    for (index, kicker) in kickers.iter().take(5).enumerate() {
        value |= (*kicker as u64) << (8 * (4 - index));
    }

    HandRank(value)
}

fn five_card_combinations(cards: &[Card; 7]) -> [[Card; 5]; 21] {
    let mut combos = [[Card::new(2, 0); 5]; 21];
    let mut index = 0;

    for i in 0..7 {
        for j in (i + 1)..7 {
            let mut combo_index = 0;
            for (source_index, card) in cards.iter().enumerate() {
                if source_index == i || source_index == j {
                    continue;
                }
                combos[index][combo_index] = *card;
                combo_index += 1;
            }
            index += 1;
        }
    }

    combos
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn cards<const N: usize>(input: &str) -> [Card; N] {
        let parsed: Vec<Card> = input
            .split_whitespace()
            .map(|token| Card::from_str(token).unwrap())
            .collect();

        parsed
            .try_into()
            .expect("invalid card count for test input")
    }

    #[test]
    fn royal_flush_beats_straight_flush() {
        let royal = cards("As Ks Qs Js Ts 2c 3d");
        let straight_flush = cards("9h 8h 7h 6h 5h Ac Kd");

        assert!(evaluate_hand(&royal) > evaluate_hand(&straight_flush));
        assert_eq!(
            evaluate_hand(&royal).category(),
            HandCategory::StraightFlush
        );
    }

    #[test]
    fn four_of_a_kind_beats_full_house() {
        let quads = cards("Ah Ad Ac As Kd Qc 2h");
        let full_house = cards("Kh Kd Kc 9s 9h 2c 3d");

        assert!(evaluate_hand(&quads) > evaluate_hand(&full_house));
        assert_eq!(evaluate_hand(&quads).category(), HandCategory::FourOfAKind);
        assert_eq!(
            evaluate_hand(&full_house).category(),
            HandCategory::FullHouse
        );
    }

    #[test]
    fn flush_beats_straight() {
        let flush = cards("Ah Kh 9h 7h 2h Qc Jd");
        let straight = cards("9c 8d 7h 6s 5c Ah Kd");

        assert!(evaluate_hand(&flush) > evaluate_hand(&straight));
        assert_eq!(evaluate_hand(&flush).category(), HandCategory::Flush);
        assert_eq!(evaluate_hand(&straight).category(), HandCategory::Straight);
    }

    #[test]
    fn wheel_straight_is_detected() {
        let wheel = cards("Ac 2d 3h 4s 5c Kd Qh");

        assert_eq!(evaluate_hand(&wheel).category(), HandCategory::Straight);
    }

    #[test]
    fn picks_best_five_from_seven() {
        let hand = cards("Ah Kh Qh Jh 2h 3c 4d");

        assert_eq!(evaluate_hand(&hand).category(), HandCategory::Flush);
    }

    #[test]
    fn pair_kickers_are_ordered() {
        let top_pair = cards("Ah Ad Kc Qs 9h 2d 3c");
        let weaker_pair = cards("Ah Ad Qc Js 9h 2d 3c");

        assert!(evaluate_hand(&top_pair) > evaluate_hand(&weaker_pair));
        assert_eq!(evaluate_hand(&top_pair).category(), HandCategory::OnePair);
    }

    #[test]
    fn two_pair_ordering() {
        let aces_and_kings = cards("Ah Ad Kc Ks 9h 2d 3c");
        let aces_and_queens = cards("Ah Ad Qc Qs 9h 2d 3c");

        assert!(evaluate_hand(&aces_and_kings) > evaluate_hand(&aces_and_queens));
    }

    #[test]
    fn high_card_ordering() {
        let ace_high = cards("Ah Kd Qc 9s 7h 2d 3c");
        let king_high = cards("Kh Qd Jc 9s 7h 2d 3c");

        assert!(evaluate_hand(&ace_high) > evaluate_hand(&king_high));
        assert_eq!(evaluate_hand(&ace_high).category(), HandCategory::HighCard);
    }
}
