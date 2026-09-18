use std::time::Instant;

use egui::{RichText, Ui};
use poker_core::{evaluate_hand, HandCategory};

use crate::util::{format_duration, parse_cards};

struct EvalCalculation {
    cards_text: String,
    category: HandCategory,
    value: u64,
    duration: std::time::Duration,
}

pub struct EvalTab {
    cards_text: String,
    error: Option<String>,
    result: Option<EvalCalculation>,
}

impl Default for EvalTab {
    fn default() -> Self {
        Self {
            cards_text: "As Ks Qs Js Ts 2c 3d".to_string(),
            error: None,
            result: None,
        }
    }
}

impl EvalTab {
    pub fn ui(&mut self, ui: &mut Ui) {
        ui.label("7 карт:");
        ui.text_edit_singleline(&mut self.cards_text);

        if ui.button("Считать").clicked() {
            self.calculate();
        }

        if let Some(error) = &self.error {
            ui.colored_label(egui::Color32::RED, error);
        }

        if let Some(result) = &self.result {
            ui.add_space(8.0);
            self.show_result(ui, result);
        }
    }

    fn calculate(&mut self) {
        self.error = None;
        self.result = None;

        let cards_vec = match parse_cards(&self.cards_text, 7) {
            Ok(cards) => cards,
            Err(message) => {
                self.error = Some(message);
                return;
            }
        };

        let cards: [poker_core::Card; 7] = match cards_vec.try_into() {
            Ok(cards) => cards,
            Err(_) => {
                self.error = Some("ожидается 7 карт".to_string());
                return;
            }
        };

        let started = Instant::now();
        let hand_rank = evaluate_hand(&cards);
        let duration = started.elapsed();

        self.result = Some(EvalCalculation {
            cards_text: self.cards_text.clone(),
            category: hand_rank.category(),
            value: hand_rank.value(),
            duration,
        });
    }

    fn show_result(&self, ui: &mut Ui, result: &EvalCalculation) {
        ui.label(
            RichText::new(format!("Время расчёта: {}", format_duration(result.duration)))
                .strong(),
        );
        ui.label(format!("Карты: {}", result.cards_text));
        ui.label(format!("Категория: {}", category_name(result.category)));
        ui.label(format!("HandRank: {}", result.value));
    }
}

fn category_name(category: HandCategory) -> &'static str {
    match category {
        HandCategory::HighCard => "High Card",
        HandCategory::OnePair => "One Pair",
        HandCategory::TwoPair => "Two Pair",
        HandCategory::ThreeOfAKind => "Three of a Kind",
        HandCategory::Straight => "Straight",
        HandCategory::Flush => "Flush",
        HandCategory::FullHouse => "Full House",
        HandCategory::FourOfAKind => "Four of a Kind",
        HandCategory::StraightFlush => "Straight Flush",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn royal_flush_detection() {
        let mut tab = EvalTab::default();
        tab.calculate();
        let result = tab.result.expect("calculation should succeed");
        assert_eq!(result.category, HandCategory::StraightFlush);
    }
}
