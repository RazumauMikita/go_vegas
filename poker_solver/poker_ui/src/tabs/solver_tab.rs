use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use egui::{Context, RichText, Ui};
use poker_core::{icm_equity, solve, EquityCache, SolverInput, SolverOutput};

use crate::tabs::strategy_tree::{
    build_strategy_tree, draw_strategy_tree, node_at_path, NodePath, TreeNode,
};
use crate::util::format_duration;
use crate::widgets::{combo_share, range_matrix_ui, MatrixMode};

#[derive(Clone)]
struct SolverResult {
    output: SolverOutput,
    input: SolverInput,
    eq_pre: Vec<f64>,
    duration: Duration,
}

enum SolverWorkerMessage {
    Done(SolverResult),
}

pub struct SolverTab {
    player_count: usize,
    stack_chips: Vec<String>,
    prize_percents: Vec<String>,
    small_blind: String,
    big_blind: String,
    cache_path: String,
    max_iterations: String,
    matrix_modes: HashMap<String, MatrixMode>,
    error: Option<String>,
    cached_equity: Option<EquityCache>,
    cached_equity_path: Option<PathBuf>,
    worker_rx: Option<mpsc::Receiver<SolverWorkerMessage>>,
    computing: bool,
    result: Option<SolverResult>,
    tree: Vec<TreeNode>,
    selected_path: Option<NodePath>,
}

impl Default for SolverTab {
    fn default() -> Self {
        Self {
            player_count: 3,
            stack_chips: vec!["1000".to_string(), "1000".to_string(), "1000".to_string()],
            prize_percents: vec!["50".to_string(), "30".to_string(), "20".to_string()],
            small_blind: "50".to_string(),
            big_blind: "100".to_string(),
            cache_path: "equity_cache.bin".to_string(),
            max_iterations: "100".to_string(),
            matrix_modes: HashMap::new(),
            error: None,
            cached_equity: None,
            cached_equity_path: None,
            worker_rx: None,
            computing: false,
            result: None,
            tree: Vec::new(),
            selected_path: None,
        }
    }
}

impl SolverTab {
    pub fn ui(&mut self, ctx: &Context) {
        egui::TopBottomPanel::top("solver_input_panel")
            .resizable(false)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                egui::CollapsingHeader::new("Настройки")
                    .default_open(true)
                    .show(ui, |ui| {
                        self.draw_input_form(ui, ctx);
                    });
                ui.add_space(4.0);
            });

        if self.result.is_some() && !self.tree.is_empty() {
            egui::SidePanel::left("strategy_tree_panel")
                .resizable(true)
                .default_width(300.0)
                .width_range(280.0..=360.0)
                .show(ctx, |ui| {
                    ui.heading("Strategy Tree");
                    ui.separator();
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            draw_strategy_tree(ui, &mut self.tree, &mut self.selected_path, &[], 0);
                        });
                });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(result) = self.result.clone() {
                self.draw_result_panels(ui, &result);
            } else {
                ui.vertical_centered(|ui| {
                    ui.add_space(40.0);
                    ui.label("Введите параметры и нажмите «Рассчитать».");
                });
            }
        });
    }

    pub fn poll(&mut self, ctx: &Context) {
        if let Some(rx) = &self.worker_rx {
            match rx.try_recv() {
                Ok(SolverWorkerMessage::Done(result)) => {
                    self.tree = build_strategy_tree(&result.output);
                    self.selected_path = if self.tree.is_empty() {
                        None
                    } else {
                        Some(vec![0])
                    };
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

    fn draw_input_form(&mut self, ui: &mut Ui, ctx: &Context) {
        ui.horizontal(|ui| {
            ui.label("Игроков:");
            if ui.selectable_label(self.player_count == 2, "2").clicked() {
                self.set_player_count(2);
            }
            if ui.selectable_label(self.player_count == 3, "3").clicked() {
                self.set_player_count(3);
            }
        });

        let bb = self.big_blind.trim().parse::<f64>().unwrap_or(100.0).max(1e-9);

        egui::Grid::new("stack_input_table")
            .num_columns(3)
            .striped(true)
            .show(ui, |ui| {
                ui.label("Position");
                ui.label("Stack (chips)");
                ui.label("Stack (BB)");
                ui.end_row();

                for index in 0..self.player_count {
                    ui.label(position_label(self.player_count, index));
                    ui.text_edit_singleline(&mut self.stack_chips[index]);
                    let stack_bb = self
                        .stack_chips
                        .get(index)
                        .and_then(|s| s.trim().parse::<f64>().ok())
                        .map(|stack| stack / bb)
                        .unwrap_or(0.0);
                    ui.label(format!("{stack_bb:.2}"));
                    ui.end_row();
                }
            });

        ui.add_space(4.0);

        egui::Grid::new("prize_input_table")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                ui.label("Place");
                ui.label("Prize (%)");
                ui.end_row();

                for index in 0..self.prize_percents.len() {
                    ui.label(format!("{}", index + 1));
                    ui.text_edit_singleline(&mut self.prize_percents[index]);
                    ui.end_row();
                }
            });

        ui.horizontal(|ui| {
            if ui.button("+").clicked() && self.prize_percents.len() < self.player_count {
                self.prize_percents.push("0".to_string());
            }
            if ui.button("−").clicked() && self.prize_percents.len() > 1 {
                self.prize_percents.pop();
            }
        });

        ui.add_space(6.0);
        draw_solver_params(ui, self);

        ui.horizontal(|ui| {
            let can_run = !self.computing;
            if ui
                .add_enabled(can_run, egui::Button::new("Рассчитать"))
                .clicked()
            {
                self.start_calculation(ctx);
            }
            if self.computing {
                ui.spinner();
                ui.label("Расчёт...");
            }
            if let Some(error) = &self.error {
                ui.colored_label(egui::Color32::RED, error);
            }
        });
    }

    fn draw_result_panels(&mut self, ui: &mut Ui, result: &SolverResult) {
        ui.label(
            RichText::new(format!(
                "Время: {} | Итерации: {} | converged={}",
                format_duration(result.duration),
                result.output.iterations_used,
                result.output.converged
            ))
            .small(),
        );
        ui.add_space(8.0);

        let path = self
            .selected_path
            .clone()
            .or_else(|| if self.tree.is_empty() { None } else { Some(vec![0]) });

        if let Some(path) = path {
            let node_data = node_at_path(&self.tree, &path).map(|node| {
                (
                    node.label.clone(),
                    node.range,
                    node.ev_range,
                )
            });

            if let Some((label, range, ev_range)) = node_data {
                {
                    let mode = self
                        .matrix_modes
                        .entry(label.clone())
                        .or_insert(MatrixMode::Frequency);
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            range_matrix_ui(ui, &range, ev_range.as_ref(), mode, &label);
                        });
                }

                ui.add_space(12.0);
                egui::CollapsingHeader::new("Outline")
                    .default_open(false)
                    .show(ui, |ui| {
                        self.show_outline_table(ui, result);
                    });
                return;
            }
        }

        ui.label("Выберите узел в дереве слева.");
    }

    fn set_player_count(&mut self, count: usize) {
        self.player_count = count;
        while self.stack_chips.len() < count {
            self.stack_chips.push("1000".to_string());
        }
        self.stack_chips.truncate(count);

        while self.prize_percents.len() < count {
            let default = if count == 2 {
                vec!["65".to_string(), "35".to_string()]
            } else {
                vec!["50".to_string(), "30".to_string(), "20".to_string()]
            };
            if self.prize_percents.is_empty() {
                self.prize_percents = default;
            } else {
                self.prize_percents.push("0".to_string());
            }
        }
        self.prize_percents.truncate(count);
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

        let payouts = normalize_payouts(&input.payouts);
        let eq_pre = icm_equity(&input.stacks, &payouts);

        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.computing = true;
        self.result = None;
        self.tree.clear();
        self.selected_path = None;
        ctx.request_repaint();

        thread::spawn(move || {
            let started = Instant::now();
            let output = solve(&input, &cache);
            let duration = started.elapsed();
            let _ = tx.send(SolverWorkerMessage::Done(SolverResult {
                output,
                input,
                eq_pre,
                duration,
            }));
        });
    }

    fn parse_input(&self) -> Result<SolverInput, String> {
        if self.stack_chips.len() != self.player_count {
            return Err("количество стеков не совпадает с числом игроков".to_string());
        }

        let stacks: Vec<f64> = self
            .stack_chips
            .iter()
            .enumerate()
            .map(|(index, text)| {
                text.trim()
                    .parse::<f64>()
                    .map_err(|_| format!("некорректный стек для позиции {}", index + 1))
            })
            .collect::<Result<Vec<_>, _>>()?;

        if stacks.len() != 2 && stacks.len() != 3 {
            return Err("солвер поддерживает 2 или 3 игрока".to_string());
        }

        let mut payouts = Vec::with_capacity(self.prize_percents.len());
        for (index, text) in self.prize_percents.iter().enumerate() {
            let value = text
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("некорректный приз для места {}", index + 1))?;
            payouts.push(value / 100.0);
        }

        let prize_sum: f64 = payouts.iter().sum();
        if (prize_sum - 1.0).abs() > 1e-6 {
            return Err(format!(
                "призы должны суммироваться в 100%, получено {:.1}%",
                prize_sum * 100.0
            ));
        }

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
            num_players: self.player_count,
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

    fn show_outline_table(&self, ui: &mut Ui, result: &SolverResult) {
        let bb = result.input.big_blind.max(1e-9);
        let players = result.input.stacks.len();
        let range_shares = outline_range_shares(&result.output, players);

        egui::Grid::new("solver_outline_table")
            .num_columns(6)
            .striped(true)
            .show(ui, |ui| {
                ui.label("Pos");
                ui.label("Stack");
                right_label(ui, "EQPre%");
                right_label(ui, "EQPost%");
                right_label(ui, "EQDiff%");
                right_label(ui, "Range%");
                ui.end_row();

                for index in 0..players {
                    ui.label(position_label(players, index));
                    let stack_bb = result.input.stacks[index] / bb;
                    right_label(ui, &format!("{stack_bb:.2}"));
                    let eq_pre = result.eq_pre[index] * 100.0;
                    let eq_post = result.output.equities[index] * 100.0;
                    let eq_diff = eq_post - eq_pre;
                    right_label(ui, &format!("{eq_pre:.2}"));
                    right_label(ui, &format!("{eq_post:.2}"));
                    right_label(ui, &format!("{:+.2}", eq_diff));
                    right_label(ui, &format!("{:.1}", range_shares[index] * 100.0));
                    ui.end_row();
                }
            });
    }
}

const PARAM_LABEL_WIDTH: f32 = 80.0;
const PARAM_FIELD_WIDTH: f32 = 200.0;
const PARAM_ROW_SPACING: f32 = 6.0;

fn draw_solver_params(ui: &mut Ui, tab: &mut SolverTab) {
    param_text_row(ui, "SB:", &mut tab.small_blind);
    param_text_row(ui, "BB:", &mut tab.big_blind);

    ui.horizontal(|ui| {
        ui.add_sized(
            [PARAM_LABEL_WIDTH, ui.spacing().interact_size.y],
            egui::Label::new("Cache:"),
        );
        ui.add(
            egui::TextEdit::singleline(&mut tab.cache_path).desired_width(PARAM_FIELD_WIDTH),
        );
        if ui.button("Обзор...").clicked() {
            if let Some(path) = pick_cache_file() {
                tab.cache_path = path;
            }
        }
    });
    ui.add_space(PARAM_ROW_SPACING);

    param_text_row(ui, "Iter:", &mut tab.max_iterations);
}

fn param_text_row(ui: &mut Ui, label: &str, value: &mut String) {
    ui.horizontal(|ui| {
        ui.add_sized(
            [PARAM_LABEL_WIDTH, ui.spacing().interact_size.y],
            egui::Label::new(label),
        );
        ui.add(egui::TextEdit::singleline(value).desired_width(PARAM_FIELD_WIDTH));
    });
    ui.add_space(PARAM_ROW_SPACING);
}

fn right_label(ui: &mut Ui, text: &str) {
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        ui.label(text);
    });
}

fn position_label(players: usize, index: usize) -> &'static str {
    if players == 2 {
        match index {
            0 => "SB",
            _ => "BB",
        }
    } else {
        match index {
            0 => "BTN",
            1 => "SB",
            _ => "BB",
        }
    }
}

fn normalize_payouts(payouts: &[f64]) -> Vec<f64> {
    let sum: f64 = payouts.iter().sum();
    if sum <= 0.0 {
        return payouts.to_vec();
    }
    payouts.iter().map(|p| *p / sum).collect()
}

fn outline_range_shares(output: &SolverOutput, players: usize) -> Vec<f64> {
    if players == 3 {
        if let Some(ranges) = output.three_max.as_ref() {
            return vec![
                combo_share(&ranges.btn_push),
                combo_share(&ranges.sb_push),
                combo_share(&ranges.bb_call_vs_btn),
            ];
        }
    }

    if players == 2 {
        return vec![
            combo_share(&output.push_ranges[0]),
            combo_share(&output.call_ranges[1]),
        ];
    }

    vec![0.0; players]
}

fn pick_cache_file() -> Option<String> {
    rfd::FileDialog::new()
        .add_filter("Equity cache", &["bin"])
        .pick_file()
        .map(|path| path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_solver_input_defaults() {
        let tab = SolverTab::default();
        let input = tab.parse_input().expect("default input should parse");
        assert_eq!(input.stacks, vec![1000.0, 1000.0, 1000.0]);
        assert_eq!(input.payouts, vec![0.5, 0.3, 0.2]);
        assert_eq!(input.small_blind, 50.0);
        assert_eq!(input.big_blind, 100.0);
        assert_eq!(input.max_iterations, 100);
    }

    #[test]
    fn outline_range_share_for_btn() {
        let tab = SolverTab::default();
        let input = tab.parse_input().expect("parse");
        let path = PathBuf::from("equity_cache.bin");
        if !path.exists() {
            return;
        }
        let cache = EquityCache::load(&path).expect("cache");
        let output = solve(&input, &cache);
        let shares = outline_range_shares(&output, 3);
        assert!(
            shares[0] > 0.20 && shares[0] < 0.32,
            "BTN range share expected ~26%, got {:.1}%",
            shares[0] * 100.0
        );
    }
}
