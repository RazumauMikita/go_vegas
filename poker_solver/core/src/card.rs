use std::fmt;
use std::str::FromStr;

pub const RANK_CHARS: &str = "23456789TJQKA";
pub const SUIT_CHARS: &str = "cdhs";

const RANK_ARRAY: [u8; 13] = *b"23456789TJQKA";
const SUIT_ARRAY: [u8; 4] = *b"cdhs";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Card {
    rank: u8,
    suit: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardParseError {
    InvalidLength,
    InvalidRank,
    InvalidSuit,
}

impl Card {
    pub const MIN_RANK: u8 = 2;
    pub const MAX_RANK: u8 = 14;

    /// Внутренний конструктор. Паникует при невалидных входах.
    /// Использовать только там, где корректность гарантирована.
    pub const fn new(rank: u8, suit: u8) -> Self {
        assert!(
            rank >= Self::MIN_RANK && rank <= Self::MAX_RANK,
            "rank out of range"
        );
        assert!(suit <= 3, "suit out of range");
        Self { rank, suit }
    }

    /// Внешний конструктор для пользовательского ввода.
    pub fn try_new(rank: u8, suit: u8) -> Result<Self, CardParseError> {
        if !(Self::MIN_RANK..=Self::MAX_RANK).contains(&rank) {
            return Err(CardParseError::InvalidRank);
        }
        if suit > 3 {
            return Err(CardParseError::InvalidSuit);
        }
        Ok(Self::new(rank, suit))
    }

    pub fn rank(self) -> u8 {
        self.rank
    }

    pub fn suit(self) -> u8 {
        self.suit
    }

    pub fn rank_char(self) -> char {
        char::from(RANK_ARRAY[(self.rank - Self::MIN_RANK) as usize])
    }

    pub fn suit_char(self) -> char {
        char::from(SUIT_ARRAY[self.suit as usize])
    }

    fn from_parts(rank_char: char, suit_char: char) -> Result<Self, CardParseError> {
        let rank = rank_char_to_u8(rank_char).ok_or(CardParseError::InvalidRank)?;
        let suit = suit_char_to_u8(suit_char).ok_or(CardParseError::InvalidSuit)?;

        Self::try_new(rank, suit)
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank_char(), self.suit_char())
    }
}

impl FromStr for Card {
    type Err = CardParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.len() != 2 {
            return Err(CardParseError::InvalidLength);
        }

        let mut chars = s.chars();
        let rank_char = chars.next().expect("length checked");
        let suit_char = chars.next().expect("length checked");

        Self::from_parts(rank_char, suit_char)
    }
}

fn rank_char_to_u8(c: char) -> Option<u8> {
    RANK_CHARS
        .chars()
        .position(|rank| rank == c || rank == c.to_ascii_uppercase())
        .map(|index| index as u8 + Card::MIN_RANK)
}

fn suit_char_to_u8(c: char) -> Option<u8> {
    SUIT_CHARS
        .chars()
        .position(|suit| suit == c || suit == c.to_ascii_lowercase())
        .map(|index| index as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_standard_notation() {
        let card = Card::from_str("As").unwrap();
        assert_eq!(card.rank(), 14);
        assert_eq!(card.suit(), 3);
        assert_eq!(card.to_string(), "As");
    }

    #[test]
    fn parses_lowercase() {
        let card = Card::from_str("td").unwrap();
        assert_eq!(card.rank(), 10);
        assert_eq!(card.suit(), 1);
    }

    #[test]
    fn rejects_invalid_card() {
        assert_eq!(Card::from_str("1x"), Err(CardParseError::InvalidRank));
        assert_eq!(Card::from_str("Az"), Err(CardParseError::InvalidSuit));
        assert_eq!(Card::from_str("A"), Err(CardParseError::InvalidLength));
    }
}
