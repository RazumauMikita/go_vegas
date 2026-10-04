use std::collections::HashMap;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use egui::{Context, RichText, Ui};
use poker_core::{
    icm_equity, parse_hand_history, solve, solver_position_index, Algorithm, EquityCache,
    SolverInput, SolverOutput,
};

use crate::tabs::strategy_tree::{
    build_strategy_tree, draw_strategy_tree, draw_strategy_tree_header, node_at_path,
    node_at_path_mut, Action, NodePath, TreeNode,
};
use crate::util::format_duration;
use crate::widgets::{
    combo_share, fill_range_to_share, range_matrix_ui, rank_hands_for_slider, MatrixMode,
};

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

struct RangeEditor {
    range_id: usize,
    path: NodePath,
    label: String,
    range: [f64; 169],
    ev_range: Option<[f64; 169]>,
    slider_pct: f64,
    ranking: Vec<usize>,
    lock: bool,
    matrix_mode: MatrixMode,
    is_raise: bool,
}

const CACHE_BYTES: &[u8] = include_bytes!("../../assets/equity_cache.bin");

pub struct SolverTab {
    player_count: usize,
    stack_chips: Vec<String>,
    prize_percents: Vec<String>,
    small_blind: String,
    big_blind: String,
    ante: String,
    max_iterations: String,
    tolerance: String,
    algorithm: Algorithm,
    equity_cache: EquityCache,
    matrix_modes: HashMap<String, MatrixMode>,
    error: Option<String>,
    worker_rx: Option<mpsc::Receiver<SolverWorkerMessage>>,
    computing: bool,
    result: Option<SolverResult>,
    tree: Vec<TreeNode>,
    selected_path: Option<NodePath>,
    locked_ranges: HashMap<usize, [f64; 169]>,
    range_editor: Option<RangeEditor>,
}

impl Default for SolverTab {
    fn default() -> Self {
        Self {
            player_count: 3,
            stack_chips: vec!["1000".to_string(), "1000".to_string(), "1000".to_string()],
            prize_percents: vec!["50".to_string(), "30".to_string(), "20".to_string()],
            small_blind: "50".to_string(),
            big_blind: "100".to_string(),
            ante: "0".to_string(),
            max_iterations: "200".to_string(),
            tolerance: "0.001".to_string(),
            algorithm: Algorithm::FictitiousPlay,
            equity_cache: EquityCache::from_bytes(CACHE_BYTES).expect("embedded cache corrupted"),
            matrix_modes: HashMap::new(),
            error: None,
            worker_rx: None,
            computing: false,
            result: None,
            tree: Vec::new(),
            selected_path: None,
            locked_ranges: HashMap::new(),
            range_editor: None,
        }
    }
}

impl SolverTab {
    pub fn ui(&mut self, ctx: &Context) {
        egui::TopBottomPanel::top("solver_input_panel")
            .resizable(false)
            .show(ctx, |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);
                egui::CollapsingHeader::new("Настройки")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);
                        self.draw_settings(ui);
                    });
                self.draw_action_row(ui, ctx);
            });

        if self.result.is_some() && !self.tree.is_empty() {
            let (default_width, min_width, max_width) = (560.0, 420.0, 820.0);
            egui::SidePanel::left("strategy_tree_panel")
                .resizable(true)
                .default_width(default_width)
                .width_range(min_width..=max_width)
                .frame(
                    egui::Frame::none()
                        .fill(egui::Color32::WHITE)
                        .inner_margin(egui::Margin::same(4.0)),
                )
                .show(ctx, |ui| {
                    ui.visuals_mut().override_text_color =
                        Some(egui::Color32::from_rgb(32, 32, 32));
                    ui.visuals_mut().panel_fill = egui::Color32::WHITE;
                    draw_strategy_tree_header(ui);
                    let stacks_bb = self.tree_stack_bbs();
                    let mut editor_path = None;
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            draw_strategy_tree(
                                ui,
                                &mut self.tree,
                                &mut self.selected_path,
                                &mut editor_path,
                                &self.locked_ranges,
                                &stacks_bb,
                                &[],
                                0,
                            );
                        });
                    if let Some(path) = editor_path {
                        self.open_range_editor(path);
                    }
                });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(result) = self.result.clone() {
                egui::ScrollArea::both()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        self.draw_result_panels(ui, &result);
                    });
            } else {
                ui.vertical_centered(|ui| {
                    ui.add_space(40.0);
                    ui.label("Введите параметры и нажмите «Рассчитать».");
                });
            }
        });

        self.draw_range_editor(ctx);
    }

    pub fn poll(&mut self, ctx: &Context) {
        if let Some(rx) = &self.worker_rx {
            match rx.try_recv() {
                Ok(SolverWorkerMessage::Done(result)) => {
                    let keep_path = self.selected_path.clone();
                    self.tree = build_strategy_tree(&result.output, result.input.button_index);
                    self.selected_path =
                        keep_path.filter(|path| node_at_path(&self.tree, path).is_some());
                    if self.selected_path.is_none() && !self.tree.is_empty() {
                        self.selected_path = Some(vec![0]);
                    }
                    self.result = Some(result);
                    self.range_editor = None;
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

    fn import_hand_from_clipboard(&mut self) {
        self.error = None;

        let text = match arboard::Clipboard::new().and_then(|mut cb| cb.get_text()) {
            Ok(text) if !text.trim().is_empty() => text,
            Ok(_) => {
                self.error = Some("буфер обмена пуст".to_string());
                return;
            }
            Err(_) => {
                self.error = Some("не удалось прочитать буфер обмена".to_string());
                return;
            }
        };

        match parse_hand_history(&text) {
            Ok(hand) => {
                let num_players = hand.players.len();
                if num_players < 2 || num_players > 9 {
                    self.error = Some(format!(
                        "поддерживаются 2–9 игроков (найдено {num_players})"
                    ));
                    return;
                }

                self.set_player_count(num_players);
                self.small_blind = format_stack(hand.small_blind);
                self.big_blind = format_stack(hand.big_blind);
                self.ante = format_stack(hand.ante);

                let mut stacks = vec!["0".to_string(); num_players];
                for player in &hand.players {
                    if let Some(index) = solver_position_index(player.position, num_players) {
                        stacks[index] = format_stack(player.stack);
                    }
                }
                self.stack_chips = stacks;
            }
            Err(err) => {
                self.error = Some(err.to_string());
            }
        }
    }

    fn draw_settings(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Игроков:");
            if ui.selectable_label(self.player_count == 2, "2").clicked() {
                self.set_player_count(2);
            }
            if ui.selectable_label(self.player_count == 3, "3").clicked() {
                self.set_player_count(3);
            }
            if ui.selectable_label(self.player_count == 4, "4").clicked() {
                self.set_player_count(4);
            }
            if ui.selectable_label(self.player_count == 5, "5").clicked() {
                self.set_player_count(5);
            }
            if ui.selectable_label(self.player_count == 6, "6").clicked() {
                self.set_player_count(6);
            }
            if ui.selectable_label(self.player_count == 7, "7").clicked() {
                self.set_player_count(7);
            }
            if ui.selectable_label(self.player_count == 8, "8").clicked() {
                self.set_player_count(8);
            }
            if ui.selectable_label(self.player_count == 9, "9").clicked() {
                self.set_player_count(9);
            }
        });

        let bb = self
            .big_blind
            .trim()
            .parse::<f64>()
            .unwrap_or(100.0)
            .max(1e-9);

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label("Стеки");
                egui::Grid::new("stack_input_table")
                    .num_columns(3)
                    .striped(true)
                    .spacing(egui::vec2(6.0, 4.0))
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
            });

            ui.add_space(16.0);

            ui.vertical(|ui| {
                ui.label("Призы");
                egui::Grid::new("prize_input_table")
                    .num_columns(2)
                    .striped(true)
                    .spacing(egui::vec2(6.0, 4.0))
                    .show(ui, |ui| {
                        ui.label("Place");
                        ui.label("Prize %");
                        ui.end_row();

                        for index in 0..self.prize_percents.len() {
                            ui.label(format!("{}", index + 1));
                            ui.add(
                                egui::TextEdit::singleline(&mut self.prize_percents[index])
                                    .hint_text("нет"),
                            );
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
            });
        });

        ui.horizontal(|ui| {
            compact_param_field(ui, "SB:", &mut self.small_blind, 70.0);
            ui.add_space(8.0);
            compact_param_field(ui, "BB:", &mut self.big_blind, 70.0);
            ui.add_space(8.0);
            compact_param_field(ui, "Ante:", &mut self.ante, 70.0);
            ui.add_space(8.0);
            compact_param_field(ui, "Iter:", &mut self.max_iterations, 70.0);
            ui.add_space(8.0);
            compact_param_field(ui, "Tol:", &mut self.tolerance, 80.0);
        });

        ui.horizontal(|ui| {
            ui.label("Algorithm:");
            egui::ComboBox::from_id_salt("solver_algorithm")
                .selected_text(algorithm_label(self.algorithm))
                .width(220.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.algorithm,
                        Algorithm::FictitiousPlay,
                        algorithm_label(Algorithm::FictitiousPlay),
                    );
                    ui.selectable_value(
                        &mut self.algorithm,
                        Algorithm::Cfr,
                        algorithm_label(Algorithm::Cfr),
                    );
                    ui.selectable_value(
                        &mut self.algorithm,
                        Algorithm::Cfr3Max,
                        algorithm_label(Algorithm::Cfr3Max),
                    );
                    ui.selectable_value(
                        &mut self.algorithm,
                        Algorithm::Cfr4Max,
                        algorithm_label(Algorithm::Cfr4Max),
                    );
                    ui.selectable_value(
                        &mut self.algorithm,
                        Algorithm::Cfr5Max,
                        algorithm_label(Algorithm::Cfr5Max),
                    );
                    ui.selectable_value(
                        &mut self.algorithm,
                        Algorithm::Cfr6Max,
                        algorithm_label(Algorithm::Cfr6Max),
                    );
                    ui.selectable_value(
                        &mut self.algorithm,
                        Algorithm::Cfr7Max,
                        algorithm_label(Algorithm::Cfr7Max),
                    );
                    ui.selectable_value(
                        &mut self.algorithm,
                        Algorithm::Cfr8Max,
                        algorithm_label(Algorithm::Cfr8Max),
                    );
                    ui.selectable_value(
                        &mut self.algorithm,
                        Algorithm::Cfr9Max,
                        algorithm_label(Algorithm::Cfr9Max),
                    );
                });
        });
        ui.label(
            RichText::new(
                "FP: быстро, ±2%. CFR: 30 итераций, ±1% (HU) / ±1.5% (3-max). 4–9-max считают только CFR.",
            )
                .small()
                .weak(),
        );

        if let Some(pool_size) = self.parse_prize_sum() {
            let has_blank_prize = self
                .prize_percents
                .iter()
                .any(|text| text.trim().is_empty());
            if pool_size > 1.0 + 1e-6 {
                ui.colored_label(
                    egui::Color32::RED,
                    format!(
                        "Сумма payouts не может превышать 1.0 (получено {:.2})",
                        pool_size
                    ),
                );
            } else if has_blank_prize {
                ui.label(RichText::new("Пустые места без приза.").small().weak());
            } else if pool_size < 1.0 - 1e-6 {
                ui.label(format!(
                    "Pool size: {pool_size:.2} ({:.1}% уже разыграно — например, 3-м местом)",
                    (1.0 - pool_size) * 100.0
                ));
            }
        }
    }

    fn draw_action_row(&mut self, ui: &mut Ui, ctx: &Context) {
        ui.horizontal(|ui| {
            if ui.button("Вставить раздачу").clicked() {
                self.import_hand_from_clipboard();
            }
            let can_run = !self.computing;
            let calc_label = if self.locked_ranges.is_empty() {
                "Рассчитать"
            } else {
                "Пересчитать"
            };
            if ui
                .add_enabled(can_run, egui::Button::new(calc_label))
                .clicked()
            {
                self.start_calculation(ctx);
            }
            if !self.locked_ranges.is_empty() {
                ui.label(format!("локи: {}", self.locked_ranges.len()));
                if ui.button("Снять локи").clicked() {
                    self.locked_ranges.clear();
                }
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
                "Время: {} | Итерации: {} | converged={} | algorithm={}",
                format_duration(result.duration),
                result.output.iterations_used,
                result.output.converged,
                result.input.algorithm.as_str()
            ))
            .small(),
        );
        if result.input.stacks.len() == 4 || result.input.stacks.len() == 5 {
            let players = result.input.stacks.len();
            let ev_line = (0..players)
                .map(|index| {
                    let equity = result.output.equities.get(index).copied().unwrap_or(0.0);
                    format!("{} {:.1}%", position_label(players, index), equity * 100.0)
                })
                .collect::<Vec<_>>()
                .join("  ");
            ui.label(RichText::new(format!("$EV: {ev_line}")).strong());
        }
        ui.add_space(8.0);

        let path = self.selected_path.clone().or_else(|| {
            if self.tree.is_empty() {
                None
            } else {
                Some(vec![0])
            }
        });

        if let Some(path) = path {
            let range_id = node_at_path(&self.tree, &path).map(|node| node.range_id);
            let ev_range = node_at_path(&self.tree, &path).and_then(|node| node.ev_range);
            let label = node_at_path(&self.tree, &path).map(|node| node.label.clone());

            if let (Some(range_id), Some(label)) = (range_id, label) {
                let stacks_bb = self.tree_stack_bbs();
                let (is_raise, title) = node_at_path(&self.tree, &path)
                    .map(|node| {
                        let is_raise = matches!(node.action, Action::Raise);
                        let amount = stacks_bb.get(node.player()).copied().unwrap_or(0.0);
                        let verb = if is_raise { "raises" } else { "calls" };
                        (
                            is_raise,
                            format!(
                                "{} {verb} {amount:.2}bb: {:.1}%",
                                node.player(),
                                node.range_pct
                            ),
                        )
                    })
                    .unwrap_or((true, label.clone()));

                ui.horizontal(|ui| {
                    let is_locked = self.locked_ranges.contains_key(&range_id);
                    let lock_text = if is_locked { "Locked" } else { "Lock" };
                    if ui.selectable_label(is_locked, lock_text).clicked() {
                        if is_locked {
                            self.locked_ranges.remove(&range_id);
                        } else if let Some(node) = node_at_path(&self.tree, &path) {
                            self.locked_ranges.insert(range_id, node.range);
                        }
                    }
                    if ui.button("Изменить").clicked() {
                        self.open_range_editor(path.clone());
                    }
                    if is_locked {
                        ui.colored_label(
                            egui::Color32::from_rgb(200, 160, 60),
                            "диапазон зафиксирован",
                        );
                    }
                });

                {
                    let mode = self
                        .matrix_modes
                        .entry(label.clone())
                        .or_insert(MatrixMode::Ev);
                    if let Some(node) = node_at_path_mut(&mut self.tree, &path) {
                        range_matrix_ui(
                            ui,
                            &mut node.range,
                            ev_range.as_ref(),
                            mode,
                            &title,
                            false,
                            is_raise,
                        );
                    }
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
        if count == self.player_count {
            return;
        }
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
            } else if count >= 4 {
                self.prize_percents.push(String::new());
            } else {
                self.prize_percents.push("0".to_string());
            }
        }
        self.prize_percents.truncate(count);
        self.locked_ranges.clear();
        self.range_editor = None;
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

        let cache = self.equity_cache.clone();

        let payouts = normalize_payouts(&input.payouts);
        let eq_pre = icm_equity(&input.stacks, &payouts);

        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.computing = true;
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

    fn open_range_editor(&mut self, path: NodePath) {
        let Some(node) = node_at_path(&self.tree, &path) else {
            return;
        };
        self.range_editor = Some(RangeEditor {
            range_id: node.range_id,
            path,
            label: node.label.clone(),
            range: node.range,
            ev_range: node.ev_range,
            slider_pct: combo_share(&node.range) * 100.0,
            ranking: rank_hands_for_slider(&node.range, node.ev_range.as_ref()),
            lock: self.locked_ranges.contains_key(&node.range_id),
            matrix_mode: MatrixMode::Frequency,
            is_raise: matches!(node.action, Action::Raise),
        });
    }

    fn apply_range_editor(&mut self) {
        let Some(editor) = self.range_editor.take() else {
            return;
        };
        if let Some(node) = node_at_path_mut(&mut self.tree, &editor.path) {
            node.range = editor.range;
            node.range_pct = combo_share(&editor.range) * 100.0;
        }
        if let Some(result) = self.result.as_mut() {
            result.output.set_range(editor.range_id, editor.range);
        }
        if editor.lock {
            self.locked_ranges.insert(editor.range_id, editor.range);
        } else {
            self.locked_ranges.remove(&editor.range_id);
        }
    }

    fn draw_range_editor(&mut self, ctx: &Context) {
        let mut open = self.range_editor.is_some();
        if !open {
            return;
        }
        let mut apply = false;
        let mut cancel = false;
        let title = self
            .range_editor
            .as_ref()
            .map(|editor| format!("Диапазон: {}", editor.label))
            .unwrap_or_default();

        egui::Window::new(title)
            .id(egui::Id::new("solver_range_editor"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                let Some(editor) = self.range_editor.as_mut() else {
                    return;
                };

                ui.horizontal(|ui| {
                    ui.label("Range");
                    let mut pct = editor.slider_pct;
                    let width = (ui.available_width() - 8.0).max(180.0);
                    let slider = ui.add_sized(
                        [width, 18.0],
                        egui::Slider::new(&mut pct, 0.0..=100.0)
                            .suffix("%")
                            .max_decimals(1),
                    );
                    if slider.changed() {
                        editor.slider_pct = pct;
                        editor.range = fill_range_to_share(&editor.ranking, pct / 100.0);
                    }
                });
                ui.label(
                    RichText::new(
                        "Ползунок набирает руки по EV (или по силе). Клик по ячейке включает/выключает руку.",
                    )
                    .small()
                    .weak(),
                );
                ui.add_space(8.0);

                if range_matrix_ui(
                    ui,
                    &mut editor.range,
                    editor.ev_range.as_ref(),
                    &mut editor.matrix_mode,
                    "",
                    true,
                    editor.is_raise,
                ) {
                    editor.slider_pct = combo_share(&editor.range) * 100.0;
                }

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.checkbox(&mut editor.lock, "Lock");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Отмена").clicked() {
                            cancel = true;
                        }
                        if ui.button("OK").clicked() {
                            apply = true;
                        }
                    });
                });
            });

        if apply {
            self.apply_range_editor();
        } else if cancel || !open {
            self.range_editor = None;
        }
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

        if stacks.len() < 2 || stacks.len() > 9 {
            return Err("солвер поддерживает 2–9 игроков".to_string());
        }

        let payouts = parse_prize_payouts(&self.prize_percents)?;

        let prize_sum: f64 = payouts.iter().sum();
        if prize_sum > 1.0 + 1e-6 {
            return Err(format!(
                "Сумма payouts не может превышать 1.0 (получено {prize_sum:.2})"
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
        let tolerance = self
            .tolerance
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("некорректный tolerance: {}", self.tolerance))?;
        if tolerance <= 0.0 {
            return Err("tolerance должен быть больше 0".to_string());
        }
        let ante = self
            .ante
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("некорректный ante: {}", self.ante))?;
        if ante < 0.0 {
            return Err("ante не может быть отрицательным".to_string());
        }

        Ok(SolverInput {
            stacks,
            payouts,
            small_blind,
            big_blind,
            ante,
            button_index: match self.player_count {
                4 => 1,
                5 => 2,
                6 => 3,
                7 => 4,
                8 => 5,
                9 => 6,
                _ => 0,
            },
            max_iterations,
            tolerance,
            num_players: self.player_count,
            verbose_convergence: false,
            profile: false,
            algorithm: resolve_algorithm(self.algorithm, self.player_count),
            rank_cache_strict: false,
            locked_ranges: self.locked_ranges.clone(),
        })
    }

    fn tree_stack_bbs(&self) -> HashMap<String, f64> {
        let Some(result) = self.result.as_ref() else {
            return HashMap::new();
        };
        let bb = result.input.big_blind.max(1e-9);
        let players = result.input.stacks.len();
        (0..players)
            .map(|index| {
                (
                    position_label(players, index).to_string(),
                    result.input.stacks[index] / bb,
                )
            })
            .collect()
    }

    fn parse_prize_sum(&self) -> Option<f64> {
        let mut sum = 0.0;
        let mut any = false;
        for text in &self.prize_percents {
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            let value = text.parse::<f64>().ok()? / 100.0;
            sum += value;
            any = true;
        }
        any.then_some(sum)
    }

    fn show_outline_table(&self, ui: &mut Ui, result: &SolverResult) {
        let bb = result.input.big_blind.max(1e-9);
        let players = result.input.stacks.len();
        let range_shares = outline_range_shares(&result.output, players);
        let pool_size: f64 = result.input.payouts.iter().sum();
        let show_eq_real = pool_size < 1.0 - 1e-6;
        let columns = if show_eq_real { 7 } else { 6 };

        egui::Grid::new("solver_outline_table")
            .num_columns(columns)
            .striped(true)
            .show(ui, |ui| {
                ui.label("Pos");
                ui.label("Stack");
                right_label(ui, "EQPre%");
                right_label(ui, "EQPost%");
                if show_eq_real {
                    right_label(ui, "EQReal%");
                }
                right_label(ui, "EQDiff%");
                right_label(ui, "Range%");
                ui.end_row();

                for index in 0..players {
                    ui.label(position_label(players, index));
                    let stack_bb = result.input.stacks[index] / bb;
                    right_label(ui, &format!("{stack_bb:.2}"));
                    let eq_pre = result.eq_pre[index] * 100.0;
                    let eq_post = result.output.equities[index] * 100.0;
                    let eq_real = eq_post * pool_size;
                    let eq_diff = eq_post - eq_pre;
                    right_label(ui, &format!("{eq_pre:.2}"));
                    right_label(ui, &format!("{eq_post:.2}"));
                    if show_eq_real {
                        right_label(ui, &format!("{eq_real:.2}"));
                    }
                    right_label(ui, &format!("{:+.2}", eq_diff));
                    right_label(ui, &format!("{:.1}", range_shares[index] * 100.0));
                    ui.end_row();
                }
            });
    }
}

const PARAM_FIELD_WIDTH: f32 = 70.0;

fn algorithm_label(algorithm: Algorithm) -> &'static str {
    match algorithm {
        Algorithm::FictitiousPlay => "Fictitious Play (fast)",
        Algorithm::Cfr => "CFR HU",
        Algorithm::Cfr3Max => "CFR 3-max",
        Algorithm::Cfr4Max => "CFR 4-max",
        Algorithm::Cfr5Max => "CFR 5-max",
        Algorithm::Cfr6Max => "CFR 6-max",
        Algorithm::Cfr7Max => "CFR 7-max",
        Algorithm::Cfr8Max => "CFR 8-max",
        Algorithm::Cfr9Max => "CFR 9-max",
    }
}

fn resolve_algorithm(selected: Algorithm, player_count: usize) -> Algorithm {
    match (selected, player_count) {
        (_, 9) => Algorithm::Cfr9Max,
        (_, 8) => Algorithm::Cfr8Max,
        (_, 7) => Algorithm::Cfr7Max,
        (_, 6) => Algorithm::Cfr6Max,
        (_, 5) => Algorithm::Cfr5Max,
        (_, 4) => Algorithm::Cfr4Max,
        (Algorithm::FictitiousPlay, _) => Algorithm::FictitiousPlay,
        (Algorithm::Cfr, 3) => Algorithm::Cfr3Max,
        (Algorithm::Cfr3Max, 2) => Algorithm::Cfr,
        (Algorithm::Cfr5Max, 3) => Algorithm::Cfr3Max,
        (Algorithm::Cfr5Max, 2) => Algorithm::Cfr,
        (Algorithm::Cfr6Max, 3) => Algorithm::Cfr3Max,
        (Algorithm::Cfr6Max, 2) => Algorithm::Cfr,
        (Algorithm::Cfr7Max, 3) => Algorithm::Cfr3Max,
        (Algorithm::Cfr7Max, 2) => Algorithm::Cfr,
        (Algorithm::Cfr8Max, 3) => Algorithm::Cfr3Max,
        (Algorithm::Cfr8Max, 2) => Algorithm::Cfr,
        (Algorithm::Cfr9Max, 3) => Algorithm::Cfr3Max,
        (Algorithm::Cfr9Max, 2) => Algorithm::Cfr,
        (algorithm, _) => algorithm,
    }
}

fn format_stack(value: f64) -> String {
    if (value - value.round()).abs() < 1e-9 {
        format!("{}", value.round() as i64)
    } else {
        format!("{value}")
    }
}

fn compact_param_field(ui: &mut Ui, label: &str, value: &mut String, width: f32) {
    ui.label(label);
    ui.add(egui::TextEdit::singleline(value).desired_width(width.max(PARAM_FIELD_WIDTH)));
}

fn right_label(ui: &mut Ui, text: &str) {
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        ui.label(text);
    });
}

fn position_label(players: usize, index: usize) -> &'static str {
    match (players, index) {
        (2, 0) => "SB",
        (2, _) => "BB",
        (4, 0) => "CO",
        (4, 1) => "BTN",
        (4, 2) => "SB",
        (4, _) => "BB",
        (5, 0) => "HJ",
        (5, 1) => "CO",
        (5, 2) => "BTN",
        (5, 3) => "SB",
        (5, _) => "BB",
        (6, 0) => "UTG",
        (6, 1) => "HJ",
        (6, 2) => "CO",
        (6, 3) => "BTN",
        (6, 4) => "SB",
        (6, _) => "BB",
        (7, 0) => "UTG",
        (7, 1) => "MP",
        (7, 2) => "HJ",
        (7, 3) => "CO",
        (7, 4) => "BTN",
        (7, 5) => "SB",
        (7, _) => "BB",
        (8, 0) => "UTG",
        (8, 1) => "EP",
        (8, 2) => "MP",
        (8, 3) => "HJ",
        (8, 4) => "CO",
        (8, 5) => "BTN",
        (8, 6) => "SB",
        (8, _) => "BB",
        (9, 0) => "UTG",
        (9, 1) => "EP",
        (9, 2) => "MP1",
        (9, 3) => "MP2",
        (9, 4) => "HJ",
        (9, 5) => "CO",
        (9, 6) => "BTN",
        (9, 7) => "SB",
        (9, _) => "BB",
        (_, 0) => "BTN",
        (_, 1) => "SB",
        (_, _) => "BB",
    }
}

fn parse_prize_payouts(prize_percents: &[String]) -> Result<Vec<f64>, String> {
    let mut payouts = Vec::with_capacity(prize_percents.len());
    for (index, text) in prize_percents.iter().enumerate() {
        let text = text.trim();
        if text.is_empty() {
            payouts.push(None);
            continue;
        }
        let value = text
            .parse::<f64>()
            .map_err(|_| format!("некорректный приз для места {}", index + 1))?;
        payouts.push(Some(value / 100.0));
    }
    while payouts.last().is_some_and(|prize| prize.is_none()) {
        payouts.pop();
    }
    if payouts.is_empty() {
        return Err("укажите приз хотя бы за 1 место".to_string());
    }
    Ok(payouts
        .into_iter()
        .map(|prize| prize.unwrap_or(0.0))
        .collect())
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

    if players == 4 {
        if let Some(ranges) = output.four_max.as_ref() {
            return vec![
                combo_share(&ranges.utg_push),
                combo_share(&ranges.btn_push),
                combo_share(&ranges.sb_push),
                combo_share(&ranges.bb_call_vs_sb),
            ];
        }
    }

    if players == 5 {
        if let Some(ranges) = output.five_max.as_ref() {
            return vec![
                combo_share(&ranges.freq[poker_core::five_node(0, 0)]),
                combo_share(&ranges.freq[poker_core::five_node(1, 0)]),
                combo_share(&ranges.freq[poker_core::five_node(2, 0)]),
                combo_share(&ranges.freq[poker_core::five_node(3, 0)]),
                combo_share(&ranges.freq[poker_core::five_node(4, 8)]),
            ];
        }
    }

    if players == 6 {
        if let Some(ranges) = output.six_max.as_ref() {
            return vec![
                combo_share(&ranges.freq[poker_core::six_node(0, 0)]),
                combo_share(&ranges.freq[poker_core::six_node(1, 0)]),
                combo_share(&ranges.freq[poker_core::six_node(2, 0)]),
                combo_share(&ranges.freq[poker_core::six_node(3, 0)]),
                combo_share(&ranges.freq[poker_core::six_node(4, 0)]),
                combo_share(&ranges.freq[poker_core::six_node(5, 16)]),
            ];
        }
    }

    if players == 7 {
        if let Some(ranges) = output.seven_max.as_ref() {
            return vec![
                combo_share(&ranges.freq[poker_core::seven_node(0, 0)]),
                combo_share(&ranges.freq[poker_core::seven_node(1, 0)]),
                combo_share(&ranges.freq[poker_core::seven_node(2, 0)]),
                combo_share(&ranges.freq[poker_core::seven_node(3, 0)]),
                combo_share(&ranges.freq[poker_core::seven_node(4, 0)]),
                combo_share(&ranges.freq[poker_core::seven_node(5, 0)]),
                combo_share(&ranges.freq[poker_core::seven_node(6, 32)]),
            ];
        }
    }

    if players == 8 {
        if let Some(ranges) = output.eight_max.as_ref() {
            return vec![
                combo_share(&ranges.freq[poker_core::eight_node(0, 0)]),
                combo_share(&ranges.freq[poker_core::eight_node(1, 0)]),
                combo_share(&ranges.freq[poker_core::eight_node(2, 0)]),
                combo_share(&ranges.freq[poker_core::eight_node(3, 0)]),
                combo_share(&ranges.freq[poker_core::eight_node(4, 0)]),
                combo_share(&ranges.freq[poker_core::eight_node(5, 0)]),
                combo_share(&ranges.freq[poker_core::eight_node(6, 0)]),
                combo_share(&ranges.freq[poker_core::eight_node(7, 64)]),
            ];
        }
    }

    if players == 9 {
        if let Some(ranges) = output.nine_max.as_ref() {
            return vec![
                combo_share(&ranges.freq[poker_core::nine_node(0, 0)]),
                combo_share(&ranges.freq[poker_core::nine_node(1, 0)]),
                combo_share(&ranges.freq[poker_core::nine_node(2, 0)]),
                combo_share(&ranges.freq[poker_core::nine_node(3, 0)]),
                combo_share(&ranges.freq[poker_core::nine_node(4, 0)]),
                combo_share(&ranges.freq[poker_core::nine_node(5, 0)]),
                combo_share(&ranges.freq[poker_core::nine_node(6, 0)]),
                combo_share(&ranges.freq[poker_core::nine_node(7, 0)]),
                combo_share(&ranges.freq[poker_core::nine_node(8, 128)]),
            ];
        }
    }

    vec![0.0; players]
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
        assert_eq!(input.max_iterations, 200);
        assert!((input.tolerance - 0.001).abs() < 1e-12);
        assert_eq!(input.algorithm, Algorithm::FictitiousPlay);
    }

    #[test]
    fn resolve_algorithm_maps_by_player_count() {
        assert_eq!(resolve_algorithm(Algorithm::Cfr3Max, 2), Algorithm::Cfr);
        assert_eq!(resolve_algorithm(Algorithm::Cfr, 3), Algorithm::Cfr3Max);
        assert_eq!(resolve_algorithm(Algorithm::Cfr3Max, 3), Algorithm::Cfr3Max);
        assert_eq!(
            resolve_algorithm(Algorithm::FictitiousPlay, 4),
            Algorithm::Cfr4Max
        );
        assert_eq!(resolve_algorithm(Algorithm::Cfr4Max, 4), Algorithm::Cfr4Max);
        assert_eq!(resolve_algorithm(Algorithm::Cfr5Max, 5), Algorithm::Cfr5Max);
        assert_eq!(
            resolve_algorithm(Algorithm::FictitiousPlay, 5),
            Algorithm::Cfr5Max
        );
        assert_eq!(resolve_algorithm(Algorithm::Cfr6Max, 6), Algorithm::Cfr6Max);
        assert_eq!(
            resolve_algorithm(Algorithm::FictitiousPlay, 6),
            Algorithm::Cfr6Max
        );
        assert_eq!(resolve_algorithm(Algorithm::Cfr7Max, 7), Algorithm::Cfr7Max);
        assert_eq!(
            resolve_algorithm(Algorithm::FictitiousPlay, 7),
            Algorithm::Cfr7Max
        );
        assert_eq!(resolve_algorithm(Algorithm::Cfr8Max, 8), Algorithm::Cfr8Max);
        assert_eq!(
            resolve_algorithm(Algorithm::FictitiousPlay, 8),
            Algorithm::Cfr8Max
        );
        assert_eq!(resolve_algorithm(Algorithm::Cfr9Max, 9), Algorithm::Cfr9Max);
        assert_eq!(
            resolve_algorithm(Algorithm::FictitiousPlay, 9),
            Algorithm::Cfr9Max
        );
    }

    #[test]
    fn parse_solver_input_blank_places_have_no_prize() {
        let mut tab = SolverTab::default();
        tab.set_player_count(4);
        tab.prize_percents = vec![
            "40".to_string(),
            "30".to_string(),
            String::new(),
            String::new(),
        ];
        let input = tab.parse_input().expect("blank 3rd and 4th");
        assert_eq!(input.payouts, vec![0.4, 0.3]);

        tab.prize_percents = vec![
            "100".to_string(),
            String::new(),
            String::new(),
            String::new(),
        ];
        let input = tab.parse_input().expect("only 1st place");
        assert_eq!(input.payouts, vec![1.0]);

        tab.prize_percents = vec![
            "40".to_string(),
            String::new(),
            "20".to_string(),
            String::new(),
        ];
        let input = tab.parse_input().expect("blank 2nd keeps later place");
        assert_eq!(input.payouts, vec![0.4, 0.0, 0.2]);
    }

    #[test]
    fn parse_solver_input_allows_partial_payouts() {
        let mut tab = SolverTab::default();
        tab.player_count = 2;
        tab.stack_chips = vec!["1000".to_string(), "1000".to_string()];
        tab.prize_percents = vec!["50".to_string(), "30".to_string()];
        let input = tab.parse_input().expect("partial payouts should parse");
        assert_eq!(input.payouts, vec![0.5, 0.3]);
    }

    #[test]
    fn parse_solver_input_rejects_payouts_over_one() {
        let mut tab = SolverTab::default();
        tab.prize_percents = vec!["60".to_string(), "50".to_string()];
        let error = tab.parse_input().expect_err("overfull payouts should fail");
        assert!(error.contains("не может превышать 1.0"));
    }

    #[test]
    fn outline_range_share_for_btn() {
        let tab = SolverTab::default();
        let input = tab.parse_input().expect("parse");
        let cache = EquityCache::from_bytes(CACHE_BYTES).expect("embedded cache");
        let output = solve(&input, &cache);
        let shares = outline_range_shares(&output, 3);
        assert!(
            shares[0] > 0.20 && shares[0] < 0.32,
            "BTN range share expected ~26%, got {:.1}%",
            shares[0] * 100.0
        );
    }
}
