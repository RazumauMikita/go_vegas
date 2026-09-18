use std::time::Instant;

use egui::{RichText, Ui};
use poker_core::icm_equity;

use crate::util::{format_duration, parse_number_list};

struct IcmCalculation {
    stacks: Vec<f64>,
    payouts: Vec<f64>,
    equities: Vec<f64>,
    duration: std::time::Duration,
}

pub struct IcmTab {
    stacks_text: String,
    payouts_text: String,
    error: Option<String>,
    result: Option<IcmCalculation>,
}

impl Default for IcmTab {
    fn default() -> Self {
        Self {
            stacks_text: "1000,1000,1000".to_string(),
            payouts_text: "0.5,0.3,0.2".to_string(),
            error: None,
            result: None,
        }
    }
}

impl IcmTab {
    pub fn ui(&mut self, ui: &mut Ui) {
        ui.label("Стеки:");
        ui.text_edit_singleline(&mut self.stacks_text);
        ui.label("Payouts:");
        ui.text_edit_singleline(&mut self.payouts_text);

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

        let stacks = match parse_number_list(&self.stacks_text, "стека") {
            Ok(values) => values,
            Err(message) => {
                self.error = Some(message);
                return;
            }
        };
        let payouts = match parse_number_list(&self.payouts_text, "payout") {
            Ok(values) => values,
            Err(message) => {
                self.error = Some(message);
                return;
            }
        };

        if stacks.is_empty() {
            self.error = Some("список стеков не может быть пустым".to_string());
            return;
        }
        if payouts.is_empty() {
            self.error = Some("список payouts не может быть пустым".to_string());
            return;
        }
        if stacks.iter().any(|&stack| stack < 0.0) {
            self.error = Some("стеки должны быть неотрицательными".to_string());
            return;
        }
        if !stacks.iter().any(|&stack| stack > 0.0) {
            self.error = Some("хотя бы один стек должен быть положительным".to_string());
            return;
        }

        let payout_sum: f64 = payouts.iter().sum();
        if (payout_sum - 1.0).abs() > 1e-9 {
            self.error = Some(format!("payouts должны суммироваться в 1.0, получено {payout_sum}"));
            return;
        }

        let started = Instant::now();
        let equities = icm_equity(&stacks, &payouts);
        let duration = started.elapsed();

        self.result = Some(IcmCalculation {
            stacks,
            payouts,
            equities,
            duration,
        });
    }

    fn show_result(&self, ui: &mut Ui, result: &IcmCalculation) {
        ui.label(
            RichText::new(format!("Время расчёта: {}", format_duration(result.duration)))
                .strong(),
        );
        ui.label(format!(
            "Payouts: {}",
            result
                .payouts
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        ui.add_space(8.0);

        egui::Grid::new("icm_results")
            .num_columns(3)
            .striped(true)
            .show(ui, |ui| {
                ui.label("Игрок");
                ui.label("Стек");
                ui.label("$EV");
                ui.end_row();

                for (index, (&stack, &equity)) in result.stacks.iter().zip(result.equities.iter()).enumerate() {
                    ui.label(format!("Player {}", index + 1));
                    ui.label(format!("{stack:.0}"));
                    ui.label(format!("{equity:.4}"));
                    ui.end_row();
                }
            });

        let total_ev: f64 = result.equities.iter().sum();
        ui.add_space(8.0);
        ui.label(format!("Сумма $EV: {total_ev:.4}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_stacks_equal_ev() {
        let mut tab = IcmTab::default();
        tab.calculate();
        let result = tab.result.expect("calculation should succeed");
        for equity in &result.equities {
            assert!(
                (*equity - 1.0 / 3.0).abs() < 0.01,
                "expected ~0.333, got {equity}"
            );
        }
    }
}
