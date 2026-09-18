use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use egui::{Context, RichText, Ui};
use poker_core::{index_to_ranks, solve, EquityCache, SolverInput, SolverOutput, ThreeMaxRanges};

use crate::util::{format_duration, parse_number_list};
use crate::widgets::range_matrix_ui;

struct SolverResult {
    output: SolverOutput,
    stacks_len: usize,
    duration: Duration,
}

enum SolverWorkerMessage {
    Done(SolverResult),
}

pub struct SolverTab {
    stacks_text: String,
    small_blind: String,
    big_blind: String,
    payouts_text: String,
    cache_path: String,
    max_iterations: String,
    error: Option<String>,
    cached_equity: Option<EquityCache>,
    cached_equity_path: Option<PathBuf>,
    worker_rx: Option<mpsc::Receiver<SolverWorkerMessage>>,
    computing: bool,
    result: Option<SolverResult>,
}

impl Default for SolverTab {
    fn default() -> Self {
        Self {
            stacks_text: "1000,1000,1000".to_string(),
            small_blind: "50".to_string(),
            big_blind: "100".to_string(),
            payouts_text: "0.5,0.3,0.2".to_string(),
            cache_path: "equity_cache.bin".to_string(),
            max_iterations: "100".to_string(),
            error: None,
            cached_equity: None,
            cached_equity_path: None,
            worker_rx: None,
            computing: false,
            result: None,
        }
    }
}

impl SolverTab {
    pub fn ui(&mut self, ui: &mut Ui, ctx: &Context) {
        ui.label("Стеки (через запятую):");
        ui.text_edit_singleline(&mut self.stacks_text);

        ui.horizontal(|ui| {
            ui.label("SB:");
            ui.text_edit_singleline(&mut self.small_blind);
            ui.label("BB:");
            ui.text_edit_singleline(&mut self.big_blind);
        });

        ui.label("Payouts:");
        ui.text_edit_singleline(&mut self.payouts_text);

        ui.horizontal(|ui| {
            ui.label("Cache path:");
            ui.text_edit_singleline(&mut self.cache_path);
            if ui.button("Обзор...").clicked() {
                if let Some(path) = pick_cache_file() {
                    self.cache_path = path;
                }
            }
        });

        ui.horizontal(|ui| {
            ui.label("Max iterations:");
            ui.text_edit_singleline(&mut self.max_iterations);
        });

        let can_run = !self.computing;
        if ui
            .add_enabled(can_run, egui::Button::new("Рассчитать"))
            .clicked()
        {
            self.start_calculation(ctx);
        }

        if self.computing {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Расчёт...");
            });
        }

        if let Some(error) = &self.error {
            ui.colored_label(egui::Color32::RED, error);
        }

        if let Some(result) = &self.result {
            ui.add_space(8.0);
            self.show_result(ui, result);
        }
    }

    pub fn poll(&mut self, ctx: &Context) {
        if let Some(rx) = &self.worker_rx {
            match rx.try_recv() {
                Ok(SolverWorkerMessage::Done(result)) => {
                    self.result = Some(result);
                    self.error = None;
                    self.computing = false;
                    self.worker_rx = None;
                    ctx.request_repaint();
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    if self.computing {
                        self.error = Some("поток расчёта завершился неожиданно".to_string());
                        self.computing = false;
                    }
                    self.worker_rx = None;
                    ctx.request_repaint();
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
    }

    fn start_calculation(&mut self, ctx: &Context) {
        self.error = None;

        let input = match self.parse_input() {
            Ok(input) => input,
            Err(message) => {
                self.error = Some(message);
                return;
            }
        };

        let cache = match self.load_cache() {
            Ok(cache) => cache,
            Err(message) => {
                self.error = Some(message);
                return;
            }
        };

        let stacks_len = input.stacks.len();
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.computing = true;
        self.result = None;
        ctx.request_repaint();

        thread::spawn(move || {
            let started = Instant::now();
            let output = solve(&input, &cache);
            let duration = started.elapsed();
            let _ = tx.send(SolverWorkerMessage::Done(SolverResult {
                output,
                stacks_len,
                duration,
            }));
        });
    }

    fn parse_input(&self) -> Result<SolverInput, String> {
        let stacks = parse_number_list(&self.stacks_text, "стека")?;
        if stacks.len() != 2 && stacks.len() != 3 {
            return Err("солвер поддерживает 2 или 3 стека".to_string());
        }

        let payouts = parse_number_list(&self.payouts_text, "payout")?;
        let small_blind = self
            .small_blind
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("некорректный SB: {}", self.small_blind))?;
        let big_blind = self
            .big_blind
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("некорректный BB: {}", self.big_blind))?;
        let max_iterations = self
            .max_iterations
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("некорректные итерации: {}", self.max_iterations))?;
        if max_iterations == 0 {
            return Err("max iterations должен быть больше 0".to_string());
        }

        Ok(SolverInput {
            stacks,
            payouts,
            small_blind,
            big_blind,
            ante: 0.0,
            button_index: 0,
            max_iterations,
            tolerance: 0.001,
            num_players: 0,
        })
    }

    fn load_cache(&mut self) -> Result<EquityCache, String> {
        let path = PathBuf::from(self.cache_path.trim());
        if self.cached_equity_path.as_ref() == Some(&path) {
            if let Some(cache) = &self.cached_equity {
                return Ok(cache.clone());
            }
        }

        if !path.exists() {
            return Err(format!(
                "файл кэша не найден: {}. Укажите путь к equity_cache.bin",
                path.display()
            ));
        }

        let cache = EquityCache::load(&path).map_err(|error| {
            format!("не удалось загрузить кэш {}: {error}", path.display())
        })?;

        self.cached_equity = Some(cache.clone());
        self.cached_equity_path = Some(path);
        Ok(cache)
    }

    fn show_result(&self, ui: &mut Ui, result: &SolverResult) {
        ui.label(
            RichText::new(format!(
                "Время расчёта: {}",
                format_duration(result.duration)
            ))
            .strong(),
        );
        ui.label(format!(
            "Итерации: {}  converged={}",
            result.output.iterations_used, result.output.converged
        ));
        ui.add_space(8.0);

        ui.label(RichText::new("$EV игроков").strong());
        egui::Grid::new("solver_ev_table")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                ui.label("Игрок");
                ui.label("$EV");
                ui.end_row();
                for (index, equity) in result.output.equities.iter().enumerate() {
                    ui.label(format!("Player {}", index + 1));
                    ui.label(format!("{:.2}%", equity * 100.0));
                    ui.end_row();
                }
            });

        ui.add_space(12.0);

        if result.stacks_len == 3 {
            if let Some(ranges) = result.output.three_max.as_ref() {
                self.show_3max_ranges(ui, ranges);
            }
        } else {
            self.show_hu_ranges(ui, &result.output);
        }
    }

    fn show_hu_ranges(&self, ui: &mut Ui, output: &SolverOutput) {
        let sb = 0;
        let bb = 1;

        show_range_share(ui, "SB push", &output.push_ranges[sb]);
        show_range_share(ui, "BB call", &output.call_ranges[bb]);
        ui.add_space(8.0);
        range_matrix_ui(ui, &output.push_ranges[sb], "SB push");
        ui.add_space(12.0);
        range_matrix_ui(ui, &output.call_ranges[bb], "BB call");
    }

    fn show_3max_ranges(&self, ui: &mut Ui, ranges: &ThreeMaxRanges) {
        let range_sets = [
            ("BTN push", &ranges.btn_push),
            ("SB call vs BTN push", &ranges.sb_call_vs_btn),
            ("BB call vs BTN push (SB fold)", &ranges.bb_call_vs_btn),
            ("BB call vs BTN push (SB call)", &ranges.bb_call_vs_btn_and_sb),
            ("SB push (BTN fold)", &ranges.sb_push),
            ("BB call vs SB push", &ranges.bb_call_vs_sb),
        ];

        for (label, range) in range_sets {
            show_range_share(ui, label, range);
            ui.add_space(8.0);
            range_matrix_ui(ui, range, label);
            ui.add_space(12.0);
        }
    }
}

fn show_range_share(ui: &mut Ui, label: &str, range: &[f64; 169]) {
    let combo_share = combo_share(range);
    ui.label(format!("{label}: combo-share {:.1}%", combo_share * 100.0));
}

fn combo_share(range: &[f64; 169]) -> f64 {
    const TOTAL_COMBOS: f64 = 1326.0;
    let weighted: f64 = range
        .iter()
        .enumerate()
        .map(|(idx, &freq)| freq.clamp(0.0, 1.0) * combo_weight(idx as u8) as f64)
        .sum();
    weighted / TOTAL_COMBOS
}

fn pick_cache_file() -> Option<String> {
    rfd::FileDialog::new()
        .add_filter("Equity cache", &["bin"])
        .pick_file()
        .map(|path| path.display().to_string())
}

fn combo_weight(idx: u8) -> u8 {
    let (high, low, suited) = index_to_ranks(idx);
    if high == low {
        6
    } else if suited {
        4
    } else {
        12
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_solver_input_defaults() {
        let tab = SolverTab::default();
        let input = tab.parse_input().expect("default input should parse");
        assert_eq!(input.stacks, vec![1000.0, 1000.0, 1000.0]);
        assert_eq!(input.small_blind, 50.0);
        assert_eq!(input.big_blind, 100.0);
        assert_eq!(input.max_iterations, 100);
    }

    #[test]
    fn combo_share_bounds() {
        let range = [1.0; 169];
        let share = combo_share(&range);
        assert!(share > 0.99);
    }
}
