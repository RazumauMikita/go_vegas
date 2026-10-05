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
    hands: Vec<SolverHand>,
    active: usize,
    next_hand_id: u64,
    settings_open: bool,
    settings_anim: f32,
}

struct SolverHand {
    id: u64,
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
            hands: vec![SolverHand::new(1)],
            active: 0,
            next_hand_id: 2,
            settings_open: true,
            settings_anim: 1.0,
        }
    }
}

impl Default for SolverHand {
    fn default() -> Self {
        Self::new(1)
    }
}

impl SolverTab {
    pub fn ui(&mut self, ctx: &Context) {
        egui::TopBottomPanel::top("solver_hand_tabs")
            .resizable(false)
            .frame(
                egui::Frame::none()
                    .fill(ctx.style().visuals.extreme_bg_color)
                    .inner_margin(egui::Margin::symmetric(8.0, 4.0)),
            )
            .show(ctx, |ui| {
                self.draw_hand_tabs(ui);
            });

        if let Some(hand) = self.hands.get_mut(self.active) {
            hand.ui(ctx, &mut self.settings_open, &mut self.settings_anim);
        }
    }

    pub fn poll(&mut self, ctx: &Context) {
        let mut computing = false;
        for hand in &mut self.hands {
            hand.poll(ctx);
            computing |= hand.computing;
        }
        if computing {
            ctx.request_repaint();
        }
    }

    fn draw_hand_tabs(&mut self, ui: &mut Ui) {
        let mut select = None;
        let mut close = None;
        let mut add = false;
        let closable = self.hands.len() > 1;

        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(4.0, 0.0);
            ui.set_height(28.0);
            if ui
                .selectable_label(self.settings_open, "Настройки")
                .on_hover_text("Показать или скрыть панель настроек")
                .clicked()
            {
                self.settings_open = !self.settings_open;
            }
            if self.hands.get(self.active).is_some_and(|hand| hand.computing) {
                ui.spinner();
            }
            ui.separator();
            let tabs_width = (ui.available_width() - 32.0).max(80.0);

            egui::ScrollArea::horizontal()
                .id_salt("solver_hand_tab_scroll")
                .max_width(tabs_width)
                .auto_shrink([true, true])
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(4.0, 0.0);
                        ui.set_height(28.0);
                        for (index, hand) in self.hands.iter().enumerate() {
                            let title = hand.tab_title();
                            let (clicked, closed, middle) = draw_hand_tab(
                                ui,
                                &title,
                                index == self.active,
                                hand.computing,
                                closable,
                            );
                            if closed || (middle && closable) {
                                close = Some(index);
                            } else if clicked {
                                select = Some(index);
                            }
                        }
                    });
                });

            let add_response = ui.add_sized(
                [24.0, 22.0],
                egui::Button::new(RichText::new("+").size(18.0)),
            );
            if add_response.on_hover_text("Новая раздача").clicked() {
                add = true;
            }
        });

        if let Some(index) = select {
            self.active = index;
        }
        if let Some(index) = close {
            self.close_hand(index);
        }
        if add {
            self.add_hand();
        }
    }

    fn add_hand(&mut self) {
        let id = self.next_hand_id;
        self.next_hand_id += 1;
        let cache = self
            .hands
            .first()
            .map(|hand| hand.equity_cache.clone())
            .unwrap_or_else(|| {
                EquityCache::from_bytes(CACHE_BYTES).expect("embedded cache corrupted")
            });
        self.hands.push(SolverHand::with_cache(id, cache));
        self.active = self.hands.len() - 1;
    }

    fn close_hand(&mut self, index: usize) {
        if self.hands.len() <= 1 || index >= self.hands.len() {
            return;
        }
        self.hands.remove(index);
        if self.active >= self.hands.len() {
            self.active = self.hands.len() - 1;
        } else if index < self.active {
            self.active -= 1;
        }
    }
}

impl SolverHand {
    fn new(id: u64) -> Self {
        Self::with_cache(
            id,
            EquityCache::from_bytes(CACHE_BYTES).expect("embedded cache corrupted"),
        )
    }

    fn with_cache(id: u64, equity_cache: EquityCache) -> Self {
        Self {
            id,
            player_count: 3,
            stack_chips: vec!["1000".to_string(), "1000".to_string(), "1000".to_string()],
            prize_percents: vec!["50".to_string(), "30".to_string(), "20".to_string()],
            small_blind: "50".to_string(),
            big_blind: "100".to_string(),
            ante: "0".to_string(),
            max_iterations: "200".to_string(),
            tolerance: "0.001".to_string(),
            algorithm: Algorithm::FictitiousPlay,
            equity_cache,
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

    fn tab_title(&self) -> String {
        format!("#{}", self.id)
    }

    fn ui(&mut self, ctx: &Context, settings_open: &mut bool, settings_anim: &mut f32) {
        let overlay_rect = ctx.available_rect();

        if self.result.is_some() && !self.tree.is_empty() {
            let window_width = ctx.screen_rect().width();
            let min_width = 200.0;
            let max_width = (window_width - 320.0).max(min_width + 80.0);
            let default_width = (window_width * 0.42).clamp(min_width, max_width);
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
                                false,
                            );
                        });
                    if let Some(path) = editor_path {
                        self.open_range_editor(path);
                    }
                });

            if let Some(result) = self.result.clone() {
                let max_outline = (ctx.available_rect().height() * 0.42).max(90.0);
                egui::TopBottomPanel::bottom("solver_outline_panel")
                    .resizable(true)
                    .default_height(148.0)
                    .min_height(72.0)
                    .max_height(max_outline)
                    .show(ctx, |ui| {
                        ui.heading("Outline");
                        egui::ScrollArea::both()
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                self.show_outline_table(ui, &result);
                            });
                    });
            }
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::central_panel(&ctx.style()).inner_margin(egui::Margin::same(6.0)))
            .show(ctx, |ui| {
                if let Some(result) = self.result.clone() {
                    self.draw_result_panels(ui, &result);
                } else {
                    ui.vertical_centered(|ui| {
                        ui.add_space(40.0);
                        ui.label("Введите параметры и нажмите «Рассчитать».");
                        if !*settings_open && ui.button("Открыть настройки").clicked() {
                            *settings_open = true;
                        }
                    });
                }
            });

        self.draw_range_editor(ctx);
        self.draw_settings_drawer(ctx, overlay_rect, settings_open, settings_anim);
    }

    fn draw_settings_drawer(
        &mut self,
        ctx: &Context,
        overlay_rect: egui::Rect,
        settings_open: &mut bool,
        settings_anim: &mut f32,
    ) {
        if ctx.input(|i| i.key_pressed(egui::Key::Escape))
            && *settings_open
            && self.range_editor.is_none()
        {
            *settings_open = false;
        }

        let dt = ctx.input(|i| i.unstable_dt).clamp(0.0, 1.0 / 30.0);
        let target = if *settings_open { 1.0 } else { 0.0 };
        *settings_anim += (target - *settings_anim) * (dt * 14.0).clamp(0.0, 1.0);
        if (*settings_anim - target).abs() > 0.002 {
            ctx.request_repaint();
        } else {
            *settings_anim = target;
        }

        if *settings_anim <= 0.001 {
            return;
        }

        let panel_w = 380.0_f32.min(overlay_rect.width() * 0.92).max(280.0);
        let x = overlay_rect.left() + (*settings_anim - 1.0) * panel_w;
        let panel_rect = egui::Rect::from_min_size(
            egui::pos2(x, overlay_rect.top()),
            egui::vec2(panel_w, overlay_rect.height()),
        );
        let dim_alpha = (*settings_anim * 120.0).round() as u8;
        let dim_rect = egui::Rect::from_min_max(
            egui::pos2(panel_rect.right().max(overlay_rect.left()), overlay_rect.top()),
            overlay_rect.max,
        );

        if dim_rect.width() > 1.0 && dim_rect.height() > 1.0 {
            let dim_response = egui::Area::new(egui::Id::new("solver_settings_dim"))
                .order(egui::Order::Middle)
                .fixed_pos(dim_rect.min)
                .movable(false)
                .interactable(true)
                .fade_in(false)
                .show(ctx, |ui| {
                    let (rect, response) =
                        ui.allocate_exact_size(dim_rect.size(), egui::Sense::click());
                    ui.painter().rect_filled(
                        rect,
                        0.0,
                        egui::Color32::from_black_alpha(dim_alpha),
                    );
                    response
                })
                .inner;

            if *settings_open && dim_response.clicked() {
                *settings_open = false;
            }
        }

        egui::Area::new(egui::Id::new("solver_settings_drawer"))
            .order(egui::Order::Foreground)
            .fixed_pos(panel_rect.min)
            .movable(false)
            .interactable(true)
            .constrain(false)
            .fade_in(false)
            .show(ctx, |ui| {
                ui.set_min_size(panel_rect.size());
                ui.set_max_size(panel_rect.size());
                egui::Frame::window(&ctx.style())
                    .fill(ctx.style().visuals.panel_fill)
                    .inner_margin(egui::Margin::same(12.0))
                    .shadow(egui::Shadow {
                        offset: egui::vec2(8.0, 0.0),
                        blur: 22.0,
                        spread: 0.0,
                        color: egui::Color32::from_black_alpha(70),
                    })
                    .show(ui, |ui| {
                        ui.set_min_height(overlay_rect.height() - 24.0);
                        ui.set_width(panel_w - 24.0);
                        ui.spacing_mut().item_spacing = egui::vec2(6.0, 6.0);

                        ui.horizontal(|ui| {
                            ui.heading("Настройки");
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui.button("×").on_hover_text("Закрыть").clicked() {
                                        *settings_open = false;
                                    }
                                },
                            );
                        });
                        ui.separator();
                        self.draw_action_row(ui, ctx);
                        ui.separator();
                        egui::ScrollArea::vertical()
                            .id_salt("solver_settings_scroll")
                            .auto_shrink([false, true])
                            .show(ui, |ui| {
                                self.draw_settings(ui);
                            });
                    });
            });
    }

    fn poll(&mut self, ctx: &Context) {
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
        ui.horizontal_wrapped(|ui| {
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

        let side_by_side = ui.available_width() >= 560.0;
        if side_by_side {
            ui.horizontal(|ui| {
                ui.vertical(|ui| self.draw_stack_table(ui, bb));
                ui.add_space(16.0);
                ui.vertical(|ui| self.draw_prize_table(ui));
            });
        } else {
            self.draw_stack_table(ui, bb);
            ui.add_space(8.0);
            self.draw_prize_table(ui);
        }

        ui.horizontal_wrapped(|ui| {
            ui.horizontal(|ui| {
                compact_param_field(ui, "SB:", &mut self.small_blind, 70.0);
            });
            ui.horizontal(|ui| {
                compact_param_field(ui, "BB:", &mut self.big_blind, 70.0);
            });
            ui.horizontal(|ui| {
                compact_param_field(ui, "Ante:", &mut self.ante, 70.0);
            });
            ui.horizontal(|ui| {
                compact_param_field(ui, "Iter:", &mut self.max_iterations, 70.0);
            });
            ui.horizontal(|ui| {
                compact_param_field(ui, "Tol:", &mut self.tolerance, 80.0);
            });
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Algorithm:");
            let combo_width = (ui.available_width() - 8.0).clamp(160.0, 220.0);
            egui::ComboBox::from_id_salt(("solver_algorithm", self.id))
                .selected_text(algorithm_label(self.algorithm))
                .width(combo_width)
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

    fn draw_stack_table(&mut self, ui: &mut Ui, bb: f64) {
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
    }

    fn draw_prize_table(&mut self, ui: &mut Ui) {
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
                        egui::TextEdit::singleline(&mut self.prize_percents[index]).hint_text("нет"),
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
    }

    fn draw_action_row(&mut self, ui: &mut Ui, ctx: &Context) {
        ui.horizontal_wrapped(|ui| {
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

                ui.add_space(4.0);
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

        let screen = ctx.screen_rect();
        let max_size = screen.size() * 0.94;
        let default_size = egui::vec2(
            (max_size.x * 0.72).clamp(560.0, 980.0),
            (max_size.y * 0.88).clamp(520.0, 920.0),
        );
        egui::Window::new(title)
            .id(egui::Id::new(("solver_range_editor", self.id)))
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .constrain(true)
            .default_size(default_size)
            .min_size([480.0, 420.0])
            .max_size(max_size)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                let Some(editor) = self.range_editor.as_mut() else {
                    return;
                };

                ui.horizontal(|ui| {
                    ui.label("Range");
                    let mut pct = editor.slider_pct;
                    let width = (ui.available_width() - 8.0).max(120.0);
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
                ui.add_space(6.0);

                let buttons_h = 32.0;
                let matrix_h = (ui.available_height() - buttons_h).max(240.0);
                let matrix_w = ui.available_width();
                ui.allocate_ui(egui::vec2(matrix_w, matrix_h), |ui| {
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
                });

                ui.add_space(6.0);
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

        let col_w = (ui.available_width() / columns as f32).max(56.0);
        egui::Grid::new("solver_outline_table")
            .num_columns(columns)
            .striped(true)
            .min_col_width(col_w)
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

fn draw_hand_tab(
    ui: &mut Ui,
    title: &str,
    selected: bool,
    computing: bool,
    closable: bool,
) -> (bool, bool, bool) {
    let mut close_clicked = false;
    let mut close_contains = false;
    let visuals = ui.visuals().clone();
    let fill = if selected {
        visuals.panel_fill
    } else {
        visuals.extreme_bg_color
    };
    let text_color = if selected {
        visuals.strong_text_color()
    } else {
        visuals.weak_text_color()
    };
    let rounding = egui::Rounding {
        nw: 5.0,
        ne: 5.0,
        sw: 0.0,
        se: 0.0,
    };

    let inner = egui::Frame::none()
        .fill(fill)
        .rounding(rounding)
        .inner_margin(egui::Margin::symmetric(8.0, 4.0))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = egui::vec2(4.0, 0.0);
            ui.horizontal(|ui| {
                ui.set_height(20.0);
                if computing {
                    ui.add(egui::Spinner::new().size(11.0));
                } else {
                    draw_hand_icon(ui, text_color, fill);
                }
                ui.add(egui::Label::new(RichText::new(title).color(text_color)).selectable(false));
                if closable {
                    let close = ui.add(
                        egui::Button::new(RichText::new("×").size(14.0).color(text_color))
                            .frame(false)
                            .sense(egui::Sense::click())
                            .min_size(egui::vec2(18.0, 18.0)),
                    );
                    close_contains = close.hovered() || close.contains_pointer();
                    if close.on_hover_text("Закрыть").clicked() {
                        close_clicked = true;
                    }
                }
            });
        });

    let response = inner
        .response
        .interact(egui::Sense::click())
        .on_hover_text(format!("Раздача {title}"));
    if close_contains && (response.clicked() || close_clicked) {
        close_clicked = true;
    }
    (
        response.clicked() && !close_clicked,
        close_clicked,
        response.middle_clicked() && !close_clicked,
    )
}

fn draw_hand_icon(ui: &mut Ui, color: egui::Color32, fill: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(13.0, 14.0), egui::Sense::hover());
    let painter = ui.painter();
    let rounding = 1.5;
    let stroke = egui::Stroke::new(1.15_f32, color);
    let back = egui::Rect::from_min_size(rect.min + egui::vec2(0.0, 0.5), egui::vec2(8.0, 11.0));
    let front = egui::Rect::from_min_size(rect.min + egui::vec2(3.5, 2.0), egui::vec2(8.0, 11.0));
    painter.rect_stroke(back, rounding, stroke);
    painter.rect_filled(front, rounding, fill);
    painter.rect_stroke(front, rounding, stroke);
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
        let tab = SolverHand::default();
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
        let mut tab = SolverHand::default();
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
        let mut tab = SolverHand::default();
        tab.player_count = 2;
        tab.stack_chips = vec!["1000".to_string(), "1000".to_string()];
        tab.prize_percents = vec!["50".to_string(), "30".to_string()];
        let input = tab.parse_input().expect("partial payouts should parse");
        assert_eq!(input.payouts, vec![0.5, 0.3]);
    }

    #[test]
    fn parse_solver_input_rejects_payouts_over_one() {
        let mut tab = SolverHand::default();
        tab.prize_percents = vec!["60".to_string(), "50".to_string()];
        let error = tab.parse_input().expect_err("overfull payouts should fail");
        assert!(error.contains("не может превышать 1.0"));
    }

    #[test]
    fn outline_range_share_for_btn() {
        let tab = SolverHand::default();
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

    #[test]
    fn solver_starts_with_one_hand_tab() {
        let tab = SolverTab::default();
        assert_eq!(tab.hands.len(), 1);
        assert_eq!(tab.active, 0);
        assert_eq!(tab.hands[0].tab_title(), "#1");
    }

    #[test]
    fn add_hand_opens_empty_tab() {
        let mut tab = SolverTab::default();
        tab.add_hand();
        assert_eq!(tab.hands.len(), 2);
        assert_eq!(tab.active, 1);
        assert_eq!(tab.hands[1].tab_title(), "#2");
        assert!(tab.hands[1].result.is_none());
        assert!(!tab.hands[1].computing);
    }

    #[test]
    fn close_last_hand_is_noop() {
        let mut tab = SolverTab::default();
        tab.close_hand(0);
        assert_eq!(tab.hands.len(), 1);
        assert_eq!(tab.active, 0);
    }

    #[test]
    fn close_hand_selects_neighbor() {
        let mut tab = SolverTab::default();
        tab.add_hand();
        tab.add_hand();
        assert_eq!(tab.active, 2);
        tab.close_hand(2);
        assert_eq!(tab.hands.len(), 2);
        assert_eq!(tab.active, 1);

        tab.active = 0;
        let remaining_id = tab.hands[1].id;
        tab.close_hand(0);
        assert_eq!(tab.hands.len(), 1);
        assert_eq!(tab.active, 0);
        assert_eq!(tab.hands[0].id, remaining_id);
    }

}
