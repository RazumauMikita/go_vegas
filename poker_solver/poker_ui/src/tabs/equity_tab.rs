use std::time::Instant;

use egui::{RichText, Ui};
use poker_core::{equity_exact, equity_monte_carlo, EquityResult, HandRange};

use crate::util::{format_duration, hands_overlap, parse_board, parse_hand};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EquityMethod {
    Exact,
    MonteCarlo,
}

struct EquityPlayerResult {
    label: String,
    result: EquityResult,
}

struct EquityCalculation {
    hero_label: String,
    villain_label: String,
    method: EquityMethod,
    iterations: u64,
    players: Vec<EquityPlayerResult>,
    duration: std::time::Duration,
}

pub struct EquityTab {
    hand1: String,
    hand2: String,
    board: String,
    iterations: String,
    method: EquityMethod,
    error: Option<String>,
    result: Option<EquityCalculation>,
}

impl Default for EquityTab {
    fn default() -> Self {
        Self {
            hand1: "AhAd".to_string(),
            hand2: "KhKd".to_string(),
            board: String::new(),
            iterations: "100000".to_string(),
            method: EquityMethod::Exact,
            error: None,
            result: None,
        }
    }
}

impl EquityTab {
    pub fn ui(&mut self, ui: &mut Ui) {
        ui.label("Рука 1:");
        ui.text_edit_singleline(&mut self.hand1);
        ui.label("Рука 2:");
        ui.text_edit_singleline(&mut self.hand2);
        ui.label("Board (опционально):");
        ui.text_edit_singleline(&mut self.board);

        ui.horizontal(|ui| {
            ui.label("Iterations:");
            ui.text_edit_singleline(&mut self.iterations);
        });

        ui.horizontal(|ui| {
            ui.radio_value(&mut self.method, EquityMethod::Exact, "Exact");
            ui.radio_value(&mut self.method, EquityMethod::MonteCarlo, "Monte Carlo");
        });

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

        let hero_hand = match parse_hand(&self.hand1) {
            Ok(hand) => hand,
            Err(message) => {
                self.error = Some(message);
                return;
            }
        };
        let villain_hand = match parse_hand(&self.hand2) {
            Ok(hand) => hand,
            Err(message) => {
                self.error = Some(message);
                return;
            }
        };

        if hands_overlap(hero_hand, villain_hand) {
            self.error = Some("руки содержат пересекающиеся карты".to_string());
            return;
        }

        let board = match parse_board(&self.board) {
            Ok(board) => board,
            Err(message) => {
                self.error = Some(message);
                return;
            }
        };

        let iterations = if self.method == EquityMethod::MonteCarlo {
            match self.iterations.trim().parse::<u64>() {
                Ok(value) if value > 0 => value,
                _ => {
                    self.error = Some(format!("некорректные iterations: {}", self.iterations));
                    return;
                }
            }
        } else {
            0
        };

        let hero_range: HandRange = vec![hero_hand];
        let villain_range: HandRange = vec![villain_hand];
        let ranges = [hero_range, villain_range];

        let started = Instant::now();
        let results = if self.method == EquityMethod::Exact {
            equity_exact(&ranges, &board)
        } else {
            equity_monte_carlo(&ranges, &board, iterations)
        };
        let duration = started.elapsed();

        self.result = Some(EquityCalculation {
            hero_label: self.hand1.clone(),
            villain_label: self.hand2.clone(),
            method: self.method,
            iterations,
            players: vec![
                EquityPlayerResult {
                    label: "Hero".to_string(),
                    result: results[0],
                },
                EquityPlayerResult {
                    label: "Villain".to_string(),
                    result: results[1],
                },
            ],
            duration,
        });
    }

    fn show_result(&self, ui: &mut Ui, result: &EquityCalculation) {
        ui.label(
            RichText::new(format!(
                "Время расчёта: {}",
                format_duration(result.duration)
            ))
            .strong(),
        );
        ui.label(format!("Hero: {}", result.hero_label));
        ui.label(format!("Villain: {}", result.villain_label));
        if self.board.trim().is_empty() {
            ui.label("Board: (preflop)");
        } else {
            ui.label(format!("Board: {}", self.board));
        }
        ui.label(match result.method {
            EquityMethod::Exact => "Метод: Exact".to_string(),
            EquityMethod::MonteCarlo => {
                format!("Метод: Monte Carlo ({} iterations)", result.iterations)
            }
        });
        ui.add_space(8.0);

        egui::Grid::new("equity_results")
            .num_columns(5)
            .striped(true)
            .show(ui, |ui| {
                ui.label("Игрок");
                ui.label("Win");
                ui.label("Tie");
                ui.label("Lose");
                ui.label("Equity");
                ui.end_row();

                for player in &result.players {
                    ui.label(&player.label);
                    ui.label(format!("{:.2}%", player.result.win * 100.0));
                    ui.label(format!("{:.2}%", player.result.tie * 100.0));
                    ui.label(format!("{:.2}%", player.result.lose * 100.0));
                    ui.label(format!("{:.2}%", player.result.equity() * 100.0));
                    ui.end_row();
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aa_vs_kk_preflop_equity() {
        let mut tab = EquityTab::default();
        tab.calculate();
        let result = tab.result.expect("calculation should succeed");
        let hero_equity = result.players[0].result.equity();
        assert!(
            hero_equity > 0.78 && hero_equity < 0.85,
            "expected ~0.81, got {hero_equity}"
        );
    }
}
