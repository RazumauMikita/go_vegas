use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use egui::{Color32, Context, FontId, RichText, Sense, Ui};
use poker_core::{
    parse_hand_histories, solve, solver_seat_label, table_position_label, Algorithm, EquityCache,
    HeroActionKind, ImportedHand, SolverInput, SolverOutput,
};

use crate::tabs::solver_tab::OpenHandRequest;
use crate::tabs::strategy_tree::{
    build_strategy_tree, find_hero_decision_path, node_at_path, Action,
};
use crate::widgets::{range_matrix_ui, MatrixMode};

const CACHE_BYTES: &[u8] = include_bytes!("../../assets/equity_cache.bin");

const HEADER: Color32 = Color32::from_rgb(70, 141, 196);
const ROW_SEL: Color32 = Color32::from_rgb(196, 226, 248);
const ROW_HOVER: Color32 = Color32::from_rgb(232, 244, 252);
const ROW_BG: Color32 = Color32::WHITE;
const TEXT: Color32 = Color32::from_rgb(32, 32, 32);
const GREEN: Color32 = Color32::from_rgb(46, 160, 67);
const RED: Color32 = Color32::from_rgb(200, 60, 60);
const YELLOW: Color32 = Color32::from_rgb(196, 150, 40);
const GRID: Color32 = Color32::from_rgb(176, 192, 208);
const ROW_H: f32 = 22.0;

pub struct ImportTab {
    id: u64,
    tournaments: Vec<ImportedTournament>,
    selected_tournament: usize,
    selected_hand: usize,
    status: String,
    error: Option<String>,
    analyze_modal: Option<AnalyzeModal>,
    worker_rx: Option<mpsc::Receiver<AnalyzeMessage>>,
    analyzing: bool,
    analyze_progress: (usize, usize),
    open_request: Option<OpenHandRequest>,
    matrix_mode: MatrixMode,
    last_settings: Option<AnalyzeSettings>,
    tournament_sort: Option<(usize, bool)>,
    hand_sort: Option<(usize, bool)>,
    outline_sort: Option<(usize, bool)>,
}

struct ImportedTournament {
    id: String,
    buy_in: f64,
    fee: f64,
    start_players: usize,
    start: String,
    end: String,
    hands: Vec<HandRow>,
}

struct HandRow {
    hand: ImportedHand,
    analysis: Option<HandAnalysis>,
}

#[derive(Clone)]
struct HandAnalysis {
    diff: f64,
    verdict: Verdict,
    range: [f64; 169],
    evs: Option<[f64; 169]>,
    title: String,
    is_raise: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Correct,
    Warning,
    Mistake,
}

#[derive(Clone)]
struct AnalyzeSettings {
    prizes: Vec<(f64, f64)>,
    warning: f64,
    mistake: f64,
    max_bb: f64,
    fold_allin_blinds: bool,
    multithread: bool,
}

struct AnalyzeModal {
    tournament_index: usize,
    hand_index: Option<usize>,
    prizes: Vec<(String, String)>,
    prize_pool: f64,
    places_paid: String,
    total_chips: String,
    warning: f32,
    mistake: f32,
    max_bb: String,
    fold_allin_blinds: bool,
    multithread: bool,
}

enum AnalyzeMessage {
    Hand {
        tournament: usize,
        hand: usize,
        analysis: Option<HandAnalysis>,
    },
    Progress {
        done: usize,
        total: usize,
    },
    Done,
}

impl Default for ImportTab {
    fn default() -> Self {
        Self::new(1)
    }
}

impl ImportTab {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            tournaments: Vec::new(),
            selected_tournament: 0,
            selected_hand: 0,
            status: String::new(),
            error: None,
            analyze_modal: None,
            worker_rx: None,
            analyzing: false,
            analyze_progress: (0, 0),
            open_request: None,
            matrix_mode: MatrixMode::Ev,
            last_settings: None,
            tournament_sort: None,
            hand_sort: None,
            outline_sort: None,
        }
    }

    pub fn tab_title(&self) -> String {
        format!("Import {}", self.id)
    }

    pub fn is_analyzing(&self) -> bool {
        self.analyzing
    }
    pub fn ui(&mut self, ctx: &Context) {
        self.handle_shortcuts(ctx);
        self.draw_analyze_modal(ctx);

        egui::TopBottomPanel::top("import_toolbar")
            .resizable(false)
            .frame(
                egui::Frame::none()
                    .fill(ctx.style().visuals.extreme_bg_color)
                    .inner_margin(egui::Margin::symmetric(8.0, 6.0)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!self.analyzing, egui::Button::new("Загрузить файлы"))
                        .clicked()
                    {
                        self.pick_files();
                    }
                    if self.analyzing {
                        ui.spinner();
                        ui.label(format!(
                            "Quick Analyze: {} / {}",
                            self.analyze_progress.0, self.analyze_progress.1
                        ));
                    } else if !self.status.is_empty() {
                        ui.label(RichText::new(&self.status).small());
                    }
                    if let Some(error) = &self.error {
                        ui.colored_label(Color32::RED, error);
                    }
                });
            });

        if self.tournaments.is_empty() {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(48.0);
                    ui.label("Загрузите файлы истории рук, чтобы анализировать турниры.");
                    if ui.button("Загрузить файлы").clicked() {
                        self.pick_files();
                    }
                });
            });
            return;
        }

        self.clamp_selection();

        egui::SidePanel::left("import_lists")
            .resizable(true)
            .default_width(ctx.screen_rect().width() * 0.58)
            .width_range(360.0..=ctx.screen_rect().width() * 0.82)
            .show(ctx, |ui| {
                let tournaments_h = (ui.available_height() * 0.38).clamp(120.0, 280.0);
                ui.allocate_ui(egui::vec2(ui.available_width(), tournaments_h), |ui| {
                    self.draw_tournament_table(ui);
                });
                ui.separator();
                self.draw_hand_table(ui);
            });

        let max_outline = (ctx.available_rect().height() * 0.42).max(90.0);
        egui::TopBottomPanel::bottom("import_outline")
            .resizable(true)
            .default_height(148.0)
            .min_height(72.0)
            .max_height(max_outline)
            .show(ctx, |ui| {
                ui.heading("Outline");
                egui::ScrollArea::both()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        self.draw_outline(ui);
                    });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::central_panel(&ctx.style()).inner_margin(egui::Margin::same(6.0)))
            .show(ctx, |ui| {
                self.draw_matrix(ui);
            });
    }

    pub fn poll(&mut self, ctx: &Context) {
        if let Some(rx) = &self.worker_rx {
            loop {
                match rx.try_recv() {
                    Ok(AnalyzeMessage::Hand {
                        tournament,
                        hand,
                        analysis,
                    }) => {
                        if let Some(row) = self
                            .tournaments
                            .get_mut(tournament)
                            .and_then(|t| t.hands.get_mut(hand))
                        {
                            row.analysis = analysis;
                        }
                    }
                    Ok(AnalyzeMessage::Progress { done, total }) => {
                        self.analyze_progress = (done, total);
                        ctx.request_repaint();
                    }
                    Ok(AnalyzeMessage::Done) => {
                        self.analyzing = false;
                        self.worker_rx = None;
                        self.status = format!(
                            "Quick Analyze завершён ({}/{})",
                            self.analyze_progress.0, self.analyze_progress.1
                        );
                        ctx.request_repaint();
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        self.analyzing = false;
                        self.worker_rx = None;
                        self.error = Some("поток анализа завершился неожиданно".into());
                        ctx.request_repaint();
                        break;
                    }
                }
            }
        }
        if self.analyzing {
            ctx.request_repaint();
        }
    }

    pub fn take_open_request(&mut self) -> Option<OpenHandRequest> {
        self.open_request.take()
    }

    fn handle_shortcuts(&mut self, ctx: &Context) {
        let (open, analyze) = ctx.input(|i| {
            (
                i.modifiers.alt && i.key_pressed(egui::Key::O),
                i.modifiers.alt && i.key_pressed(egui::Key::A),
            )
        });
        if open {
            self.queue_open_hand();
        }
        if analyze && !self.analyzing && !self.tournaments.is_empty() {
            self.open_analyze_modal(self.selected_tournament, None);
        }
    }

    fn clamp_selection(&mut self) {
        if self.selected_tournament >= self.tournaments.len() {
            self.selected_tournament = self.tournaments.len().saturating_sub(1);
        }
        if let Some(tournament) = self.tournaments.get(self.selected_tournament) {
            if self.selected_hand >= tournament.hands.len() {
                self.selected_hand = tournament.hands.len().saturating_sub(1);
            }
        }
    }

    pub fn import_paths(&mut self, files: Vec<PathBuf>) {
        self.import_files(files);
    }

    fn pick_files(&mut self) {
        self.error = None;
        let files = rfd::FileDialog::new()
            .add_filter("Hand history", &["txt"])
            .set_title("Import from Hand History Files")
            .pick_files();
        if let Some(files) = files {
            self.import_files(files);
        }
    }

    fn import_files(&mut self, files: Vec<PathBuf>) {
        let mut added_hands = 0usize;
        for path in files {
            let text = match std::fs::read_to_string(&path) {
                Ok(text) => text,
                Err(err) => {
                    self.error = Some(format!(
                        "не удалось прочитать {}: {err}",
                        path.display()
                    ));
                    continue;
                }
            };
            for hand in parse_hand_histories(&text) {
                if self.push_hand(hand) {
                    added_hands += 1;
                }
            }
        }
        self.tournaments.sort_by(|a, b| b.start.cmp(&a.start).then(b.id.cmp(&a.id)));
        for tournament in &mut self.tournaments {
            tournament
                .hands
                .sort_by(|a, b| b.hand.datetime.cmp(&a.hand.datetime).then(b.hand.hand_id.cmp(&a.hand.hand_id)));
        }
        let added_tournaments = self.tournaments.len();
        self.status = format!(
            "загружено {added_hands} рук, {added_tournaments} турниров"
        );
        if !self.tournaments.is_empty() && self.selected_tournament >= self.tournaments.len() {
            self.selected_tournament = 0;
            self.selected_hand = 0;
        }
    }

    fn push_hand(&mut self, hand: ImportedHand) -> bool {
        let Some(existing) = self
            .tournaments
            .iter_mut()
            .find(|tournament| tournament.id == hand.tournament_id)
        else {
            let datetime = hand.datetime.clone();
            self.tournaments.push(ImportedTournament {
                id: hand.tournament_id.clone(),
                buy_in: hand.buy_in,
                fee: hand.fee,
                start_players: hand.players.len(),
                start: datetime.clone(),
                end: datetime,
                hands: vec![HandRow {
                    hand,
                    analysis: None,
                }],
            });
            return true;
        };

        if existing.hands.iter().any(|row| row.hand.hand_id == hand.hand_id) {
            return false;
        }
        if existing.start.is_empty() || (!hand.datetime.is_empty() && hand.datetime < existing.start)
        {
            existing.start = hand.datetime.clone();
        }
        if hand.datetime > existing.end {
            existing.end = hand.datetime.clone();
        }
        existing.start_players = existing.start_players.max(hand.players.len());
        existing.hands.push(HandRow {
            hand,
            analysis: None,
        });
        true
    }

    fn selected_row(&self) -> Option<&HandRow> {
        self.tournaments
            .get(self.selected_tournament)
            .and_then(|t| t.hands.get(self.selected_hand))
    }

    fn draw_tournament_table(&mut self, ui: &mut Ui) {
        let cols = [140.0, 90.0, 64.0, 150.0, 150.0];
        if let Some(col) = draw_header(
            ui,
            &cols,
            &["Tournament", "Buyin", "Players", "Start", "End"],
            self.tournament_sort,
        ) {
            toggle_sort(&mut self.tournament_sort, col, matches!(col, 0));
        }
        let order = tournament_order(&self.tournaments, self.tournament_sort);
        let analyzing = self.analyzing;
        let mut select = None;
        let mut analyze = None;
        egui::ScrollArea::vertical()
            .id_salt("import_tournament_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for &index in &order {
                    let tournament = &self.tournaments[index];
                    let buyin = format!("{:.2}+{:.2}", tournament.buy_in, tournament.fee);
                    let players = tournament.start_players.to_string();
                    let values = [
                        tournament.id.as_str(),
                        buyin.as_str(),
                        players.as_str(),
                        tournament.start.as_str(),
                        tournament.end.as_str(),
                    ];
                    let response = draw_row(
                        ui,
                        &cols,
                        &values,
                        index == self.selected_tournament,
                        None,
                    );
                    if response.clicked() || response.secondary_clicked() {
                        select = Some(index);
                    }
                    response.context_menu(|ui| {
                        if ui
                            .add_enabled(!analyzing, egui::Button::new("Quick Analyze"))
                            .clicked()
                        {
                            analyze = Some(index);
                            ui.close_menu();
                        }
                    });
                }
            });
        if let Some(index) = select {
            self.selected_tournament = index;
            self.selected_hand = 0;
        }
        if let Some(index) = analyze {
            self.selected_tournament = index;
            self.open_analyze_modal(index, None);
        }
    }

    fn draw_hand_table(&mut self, ui: &mut Ui) {
        let cols = [72.0, 132.0, 52.0, 44.0, 44.0, 56.0, 64.0, 56.0, 90.0];
        if let Some(col) = draw_header(
            ui,
            &cols,
            &[
                "Diff%", "Date", "Cards", "Action", "Pos", "Stack", "BB(eff)", "Players", "Level",
            ],
            self.hand_sort,
        ) {
            toggle_sort(&mut self.hand_sort, col, matches!(col, 2 | 3 | 4));
        }
        let Some(tournament) = self.tournaments.get(self.selected_tournament) else {
            return;
        };
        let order = hand_order(tournament, self.hand_sort);
        let mut select = None;
        let mut open = false;
        let mut analyze_one = None;
        let selected = self.selected_hand;
        let analyzing = self.analyzing;
        egui::ScrollArea::vertical()
            .id_salt("import_hand_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for &index in &order {
                    let row = &tournament.hands[index];
                    let hand = &row.hand;
                    let pos = hand
                        .hero_player()
                        .and_then(|p| solver_seat_label(p.position, hand.players.len()))
                        .unwrap_or("?");
                    let action = hand.hero_action.map(|a| a.code()).unwrap_or("");
                    let cards = hand.hero_combo_label.as_deref().unwrap_or("");
                    let stack = format!("{:.2}", hand.hero_stack_bb());
                    let eff = format!("{:.2}", hand.effective_bb());
                    let players = hand.players.len().to_string();
                    let diff = row
                        .analysis
                        .as_ref()
                        .map(|a| format!("{:+.2}", a.diff))
                        .unwrap_or_default();
                    let values = [
                        diff.as_str(),
                        hand.datetime.as_str(),
                        cards,
                        action,
                        pos,
                        stack.as_str(),
                        eff.as_str(),
                        players.as_str(),
                        hand.level_label.as_str(),
                    ];
                    let verdict = row.analysis.as_ref().map(|a| a.verdict);
                    let response = draw_row(ui, &cols, &values, index == selected, verdict);
                    if response.clicked() || response.secondary_clicked() {
                        select = Some(index);
                    }
                    if response.double_clicked() {
                        select = Some(index);
                        open = true;
                    }
                    response.context_menu(|ui| {
                        if ui.button("Open Hand").clicked() {
                            select = Some(index);
                            open = true;
                            ui.close_menu();
                        }
                        if ui
                            .add_enabled(!analyzing, egui::Button::new("Quick Analyze"))
                            .clicked()
                        {
                            analyze_one = Some(index);
                            ui.close_menu();
                        }
                    });
                }
            });
        if let Some(index) = select {
            self.selected_hand = index;
        }
        if open {
            self.queue_open_hand();
        }
        if let Some(index) = analyze_one {
            self.selected_hand = index;
            self.open_analyze_modal(self.selected_tournament, Some(index));
        }
    }

    fn draw_matrix(&mut self, ui: &mut Ui) {
        let Some(row) = self.selected_row() else {
            ui.label("Выберите руку в списке.");
            return;
        };
        let Some(analysis) = row.analysis.clone() else {
            ui.vertical_centered(|ui| {
                ui.add_space(24.0);
                ui.label("Правый клик по турниру → Quick Analyze, чтобы рассчитать руки.");
            });
            return;
        };
        let mut range = analysis.range;
        let highlight = row.hand.hero_combo.map(|c| c as usize);
        range_matrix_ui(
            ui,
            &mut range,
            analysis.evs.as_ref(),
            &mut self.matrix_mode,
            &analysis.title,
            false,
            analysis.is_raise,
            true,
            highlight,
        );
    }

    fn draw_outline(&mut self, ui: &mut Ui) {
        let cols = [56.0, 140.0, 72.0, 64.0, 56.0];
        if let Some(col) = draw_header(
            ui,
            &cols,
            &["Pos", "Name", "Stack", "BB", "Action"],
            self.outline_sort,
        ) {
            toggle_sort(&mut self.outline_sort, col, matches!(col, 0 | 1 | 4));
        }
        let sort = self.outline_sort;
        let Some(row) = self.selected_row() else {
            return;
        };
        let hand = &row.hand;
        let n = hand.players.len();
        let bb = hand.big_blind.max(1e-9);
        let mut rows: Vec<(usize, &str, String, String, &str)> = (0..n)
            .filter_map(|index| {
                let label = table_position_label(n, index);
                let player = hand.players.iter().find(|p| {
                    solver_seat_label(p.position, n) == Some(label)
                })?;
                let action = hand
                    .player_action(&player.name)
                    .map(|a| a.code())
                    .unwrap_or("");
                Some((
                    index,
                    player.name.as_str(),
                    format!("{:.0}", player.stack),
                    format!("{:.2}", player.stack / bb),
                    action,
                ))
            })
            .collect();
        if let Some((col, asc)) = sort {
            rows.sort_by(|a, b| {
                let ord = match col {
                    0 => a.0.cmp(&b.0),
                    1 => a.1.cmp(b.1),
                    2 => cmp_f64_str(&a.2, &b.2),
                    3 => cmp_f64_str(&a.3, &b.3),
                    _ => a.4.cmp(b.4),
                };
                if asc {
                    ord
                } else {
                    ord.reverse()
                }
            });
        }
        for (index, name, stack, stack_bb, action) in rows {
            let label = table_position_label(n, index);
            let values = [label, name, stack.as_str(), stack_bb.as_str(), action];
            draw_row(ui, &cols, &values, name == hand.hero_name, None);
        }
    }

    fn open_analyze_modal(&mut self, tournament_index: usize, hand_index: Option<usize>) {
        let Some(tournament) = self.tournaments.get(tournament_index) else {
            return;
        };
        let start_players = tournament.start_players.max(2);
        let prize_pool = tournament.buy_in * start_players as f64;
        let (pcts, places) = if start_players <= 3 {
            (vec![65.0, 35.0], 2)
        } else {
            (vec![50.0, 30.0, 20.0], 3)
        };
        let prizes = if let Some(settings) = &self.last_settings {
            settings
                .prizes
                .iter()
                .map(|(dollars, pct)| (format!("{dollars:.2}"), format!("{pct:.1}")))
                .collect()
        } else {
            pcts.iter()
                .map(|pct| {
                    (
                        format!("{:.2}", prize_pool * pct / 100.0),
                        format!("{pct:.1}"),
                    )
                })
                .collect()
        };
        let start_stack = tournament
            .hands
            .first()
            .and_then(|row| row.hand.hero_stack())
            .unwrap_or(1500.0);
        let settings = self.last_settings.clone();
        self.analyze_modal = Some(AnalyzeModal {
            tournament_index,
            hand_index,
            prizes,
            prize_pool,
            places_paid: places.to_string(),
            total_chips: format!("{:.0}", start_stack * start_players as f64),
            warning: settings.as_ref().map(|s| s.warning as f32).unwrap_or(0.10),
            mistake: settings.as_ref().map(|s| s.mistake as f32).unwrap_or(0.10),
            max_bb: settings
                .as_ref()
                .map(|s| format!("{:.0}", s.max_bb))
                .unwrap_or_else(|| "15".into()),
            fold_allin_blinds: settings
                .as_ref()
                .map(|s| s.fold_allin_blinds)
                .unwrap_or(true),
            multithread: settings.as_ref().map(|s| s.multithread).unwrap_or(true),
        });
    }

    fn draw_analyze_modal(&mut self, ctx: &Context) {
        let Some(mut modal) = self.analyze_modal.take() else {
            return;
        };
        let mut open = true;
        let mut confirmed = false;
        let mut cancelled = false;
        egui::Window::new("Quick Analyze Settings")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.label(RichText::new("ICM Prizes").strong());
                egui::Grid::new("qa_prizes")
                    .num_columns(3)
                    .spacing(egui::vec2(8.0, 4.0))
                    .show(ui, |ui| {
                        ui.label("Place");
                        ui.label("Prize($)");
                        ui.label("%");
                        ui.end_row();
                        for (index, (dollars, pct)) in modal.prizes.iter_mut().enumerate() {
                            ui.label(format!("{}", index + 1));
                            let dollar_edit = ui.add(
                                egui::TextEdit::singleline(dollars).desired_width(80.0),
                            );
                            let pct_edit =
                                ui.add(egui::TextEdit::singleline(pct).desired_width(60.0));
                            if pct_edit.changed() {
                                if let Ok(value) = pct.trim().parse::<f64>() {
                                    *dollars = format!("{:.2}", modal.prize_pool * value / 100.0);
                                }
                            } else if dollar_edit.changed() && modal.prize_pool > 0.0 {
                                if let Ok(value) = dollars.trim().parse::<f64>() {
                                    *pct = format!("{:.1}", value / modal.prize_pool * 100.0);
                                }
                            }
                            ui.end_row();
                        }
                    });
                ui.horizontal(|ui| {
                    ui.label("Places Paid");
                    let paid_edit =
                        ui.add(egui::TextEdit::singleline(&mut modal.places_paid).desired_width(40.0));
                    if paid_edit.changed() {
                        if let Ok(n) = modal.places_paid.trim().parse::<usize>() {
                            let n = n.clamp(1, 9);
                            while modal.prizes.len() < n {
                                modal.prizes.push(("0.00".into(), "0.0".into()));
                            }
                            modal.prizes.truncate(n);
                        }
                    }
                    ui.label("Total Chips");
                    ui.add(
                        egui::TextEdit::singleline(&mut modal.total_chips).desired_width(80.0),
                    );
                    ui.label("Reentry Mode  Off");
                });
                ui.separator();
                ui.label(RichText::new("Options").strong());
                ui.horizontal(|ui| {
                    ui.label("EQDiff% Warning");
                    ui.add(egui::Slider::new(&mut modal.warning, 0.0..=2.0).max_decimals(2));
                });
                ui.horizontal(|ui| {
                    ui.label("EQDiff% Mistake");
                    ui.add(egui::Slider::new(&mut modal.mistake, 0.0..=2.0).max_decimals(2));
                });
                ui.separator();
                ui.label(RichText::new("Filters").strong());
                ui.horizontal(|ui| {
                    ui.label("Max. effective BBs");
                    ui.add(egui::TextEdit::singleline(&mut modal.max_bb).desired_width(50.0));
                });
                ui.checkbox(&mut modal.fold_allin_blinds, "Fold all-in Blinds");
                ui.separator();
                ui.label(RichText::new("Performance").strong());
                ui.checkbox(&mut modal.multithread, "Use Multi-Threading");
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("OK").clicked() {
                        confirmed = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancelled = true;
                    }
                });
            });

        if confirmed {
            match modal.to_settings() {
                Ok(settings) => {
                    self.last_settings = Some(settings.clone());
                    self.start_analyze(
                        modal.tournament_index,
                        modal.hand_index,
                        settings,
                        ctx,
                    );
                }
                Err(err) => self.error = Some(err),
            }
        } else if open && !cancelled {
            self.analyze_modal = Some(modal);
        }
    }

    fn start_analyze(
        &mut self,
        tournament_index: usize,
        hand_index: Option<usize>,
        settings: AnalyzeSettings,
        ctx: &Context,
    ) {
        let Some(tournament) = self.tournaments.get(tournament_index) else {
            return;
        };
        let jobs: Vec<(usize, ImportedHand)> = tournament
            .hands
            .iter()
            .enumerate()
            .filter(|(index, _)| hand_index.is_none_or(|only| only == *index))
            .map(|(index, row)| (index, row.hand.clone()))
            .collect();
        if jobs.is_empty() {
            return;
        }
        let total = jobs.len();
        let cache = match EquityCache::from_bytes(CACHE_BYTES) {
            Ok(cache) => cache,
            Err(_) => {
                self.error = Some("не удалось загрузить equity cache".into());
                return;
            }
        };
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.analyzing = true;
        self.analyze_progress = (0, total);
        self.error = None;
        ctx.request_repaint();

        let workers = if settings.multithread { 2 } else { 1 };
        let queue = Arc::new(Mutex::new(jobs));
        let remaining = Arc::new(AtomicUsize::new(workers));
        let done = Arc::new(AtomicUsize::new(0));
        let output_cache: Arc<Mutex<HashMap<InputKey, SolverOutput>>> =
            Arc::new(Mutex::new(HashMap::new()));

        for _ in 0..workers {
            let tx = tx.clone();
            let queue = queue.clone();
            let remaining = remaining.clone();
            let done = done.clone();
            let cache = cache.clone();
            let settings = settings.clone();
            let output_cache = output_cache.clone();
            thread::spawn(move || {
                loop {
                    let job = queue.lock().ok().and_then(|mut q| q.pop());
                    let Some((hand_idx, hand)) = job else {
                        break;
                    };
                    let analysis = analyze_imported_hand(&hand, &cache, &settings, &output_cache);
                    let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
                    let _ = tx.send(AnalyzeMessage::Hand {
                        tournament: tournament_index,
                        hand: hand_idx,
                        analysis,
                    });
                    let _ = tx.send(AnalyzeMessage::Progress {
                        done: finished,
                        total,
                    });
                }
                if remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                    let _ = tx.send(AnalyzeMessage::Done);
                }
            });
        }
    }

    fn queue_open_hand(&mut self) {
        let Some(row) = self.selected_row() else {
            return;
        };
        let hand = &row.hand;
        let Some(stacks) = hand.solver_stacks() else {
            self.error = Some("не удалось собрать стеки для солвера".into());
            return;
        };
        let n = stacks.len();
        let Some(hero) = hand.hero_player() else {
            self.error = Some("герой не найден в руке".into());
            return;
        };
        let Some(hero_label) = solver_seat_label(hero.position, n) else {
            self.error = Some("неизвестная позиция героя".into());
            return;
        };
        let actions_before = actions_before_hero(hand);
        let prizes = self
            .last_settings
            .as_ref()
            .map(|s| prize_percent_strings(&s.prizes, n))
            .unwrap_or_else(|| default_prize_strings(n));
        self.open_request = Some(OpenHandRequest {
            player_count: n,
            stacks,
            prize_percents: prizes,
            small_blind: hand.small_blind,
            big_blind: hand.big_blind,
            ante: hand.ante,
            actions_before,
            hero_label: hero_label.to_string(),
        });
    }
}

impl AnalyzeModal {
    fn to_settings(&self) -> Result<AnalyzeSettings, String> {
        let mut prizes = Vec::new();
        for (dollars, pct) in &self.prizes {
            let dollars = dollars
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("некорректный приз ${dollars}"))?;
            let pct = pct
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("некорректный % {pct}"))?;
            prizes.push((dollars, pct));
        }
        if prizes.is_empty() {
            return Err("укажите хотя бы одно призовое место".into());
        }
        let max_bb = self
            .max_bb
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("некорректный Max. effective BBs: {}", self.max_bb))?;
        Ok(AnalyzeSettings {
            prizes,
            warning: self.warning.max(0.0) as f64,
            mistake: self.mistake.max(self.warning) as f64,
            max_bb,
            fold_allin_blinds: self.fold_allin_blinds,
            multithread: self.multithread,
        })
    }
}

type InputKey = (usize, Vec<i64>, i64, i64, i64, Vec<i64>);

fn analyze_imported_hand(
    hand: &ImportedHand,
    cache: &EquityCache,
    settings: &AnalyzeSettings,
    output_cache: &Mutex<HashMap<InputKey, SolverOutput>>,
) -> Option<HandAnalysis> {
    if hand.effective_bb() > settings.max_bb + 1e-9 {
        return None;
    }
    let combo = hand.hero_combo? as usize;
    let hero = hand.hero_player()?;
    let n = hand.players.len();
    let hero_label = solver_seat_label(hero.position, n)?;
    let hero_action = hand.hero_action?;
    if matches!(hero_action, HeroActionKind::Check) {
        return None;
    }

    let actions_before = actions_before_hero(hand);
    let facing_shove = hand
        .preflop_actions
        .iter()
        .take_while(|action| action.name != hand.hero_name)
        .any(|action| action.all_in || action.kind == HeroActionKind::Raise);
    if !settings.fold_allin_blinds && facing_shove && hero_label == "BB" {
        return None;
    }

    let input = solver_input(hand, settings)?;
    let key = input_key(&input);
    let output = {
        if let Ok(map) = output_cache.lock() {
            if let Some(output) = map.get(&key) {
                output.clone()
            } else {
                drop(map);
                let output = solve(&input, cache);
                if let Ok(mut map) = output_cache.lock() {
                    map.insert(key, output.clone());
                }
                output
            }
        } else {
            solve(&input, cache)
        }
    };

    let tree = build_strategy_tree(&output, input.button_index);
    let path = find_hero_decision_path(&tree, &actions_before, hero_label)?;
    let node = node_at_path(&tree, &path)?;
    let is_raise = matches!(node.action, Action::Raise);
    let took_action = if is_raise {
        matches!(hero_action, HeroActionKind::Raise)
    } else {
        matches!(hero_action, HeroActionKind::Call | HeroActionKind::Raise)
    };
    let freq = node.range[combo].clamp(0.0, 1.0);
    let diff = if let Some(evs) = node.ev_range {
        if took_action {
            evs[combo]
        } else {
            -evs[combo]
        }
    } else {
        let signed = if took_action { 1.0 } else { -1.0 };
        signed * (2.0 * freq - 1.0)
    };
    let verdict = if diff >= -settings.warning {
        Verdict::Correct
    } else if diff >= -settings.mistake {
        Verdict::Warning
    } else {
        Verdict::Mistake
    };
    Some(HandAnalysis {
        diff,
        verdict,
        range: node.range,
        evs: node.ev_range,
        title: node.label.clone(),
        is_raise,
    })
}

fn actions_before_hero(hand: &ImportedHand) -> Vec<(String, bool)> {
    let n = hand.players.len();
    let mut actions = Vec::new();
    for action in &hand.preflop_actions {
        if action.name == hand.hero_name {
            break;
        }
        let Some(player) = hand.players.iter().find(|p| p.name == action.name) else {
            continue;
        };
        let Some(label) = solver_seat_label(player.position, n) else {
            continue;
        };
        actions.push((label.to_string(), action.kind.is_fold()));
    }
    actions
}

fn solver_input(hand: &ImportedHand, settings: &AnalyzeSettings) -> Option<SolverInput> {
    let stacks = hand.solver_stacks()?;
    let n = stacks.len();
    let mut payouts: Vec<f64> = settings.prizes.iter().map(|(_, pct)| pct / 100.0).collect();
    payouts.truncate(n);
    while payouts.last().is_some_and(|v| *v == 0.0) {
        payouts.pop();
    }
    if payouts.is_empty() {
        payouts.push(1.0);
    }
    Some(SolverInput {
        stacks,
        payouts,
        small_blind: hand.small_blind,
        big_blind: hand.big_blind,
        ante: hand.ante,
        button_index: button_index_for(n),
        max_iterations: iterations_for(n),
        tolerance: 0.001,
        num_players: n,
        verbose_convergence: false,
        profile: false,
        algorithm: algorithm_for(n),
        rank_cache_strict: false,
        locked_ranges: HashMap::new(),
    })
}

fn input_key(input: &SolverInput) -> InputKey {
    (
        input.num_players,
        input.stacks.iter().map(|s| (s * 10.0).round() as i64).collect(),
        (input.small_blind * 10.0).round() as i64,
        (input.big_blind * 10.0).round() as i64,
        (input.ante * 10.0).round() as i64,
        input
            .payouts
            .iter()
            .map(|p| (p * 10000.0).round() as i64)
            .collect(),
    )
}

fn button_index_for(players: usize) -> usize {
    match players {
        4 => 1,
        5 => 2,
        6 => 3,
        7 => 4,
        8 => 5,
        9 => 6,
        _ => 0,
    }
}

fn algorithm_for(players: usize) -> Algorithm {
    match players {
        2 => Algorithm::Cfr,
        4 => Algorithm::Cfr4Max,
        5 => Algorithm::Cfr5Max,
        6 => Algorithm::Cfr6Max,
        7 => Algorithm::Cfr7Max,
        8 => Algorithm::Cfr8Max,
        9 => Algorithm::Cfr9Max,
        _ => Algorithm::Cfr3Max,
    }
}

fn iterations_for(players: usize) -> usize {
    match players {
        2 | 3 => 150,
        4 | 5 => 80,
        _ => 40,
    }
}

fn prize_percent_strings(prizes: &[(f64, f64)], players: usize) -> Vec<String> {
    let mut out: Vec<String> = prizes
        .iter()
        .map(|(_, pct)| {
            if *pct == 0.0 {
                String::new()
            } else if (pct - pct.round()).abs() < 1e-9 {
                format!("{}", *pct as i64)
            } else {
                format!("{pct}")
            }
        })
        .collect();
    if out.len() > players {
        out.truncate(players);
    }
    while out.len() < players {
        out.push(String::new());
    }
    out
}

fn default_prize_strings(players: usize) -> Vec<String> {
    if players <= 2 {
        vec!["65".into(), "35".into()]
    } else {
        let mut out = vec!["50".into(), "30".into(), "20".into()];
        while out.len() < players {
            out.push(String::new());
        }
        out.truncate(players);
        out
    }
}

fn toggle_sort(sort: &mut Option<(usize, bool)>, col: usize, default_asc: bool) {
    *sort = match *sort {
        Some((current, asc)) if current == col => Some((current, !asc)),
        _ => Some((col, default_asc)),
    };
}

fn tournament_order(tournaments: &[ImportedTournament], sort: Option<(usize, bool)>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..tournaments.len()).collect();
    let Some((col, asc)) = sort else {
        return order;
    };
    order.sort_by(|&a, &b| {
        let left = &tournaments[a];
        let right = &tournaments[b];
        let ord = match col {
            0 => left.id.cmp(&right.id),
            1 => left
                .buy_in
                .partial_cmp(&right.buy_in)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    left.fee
                        .partial_cmp(&right.fee)
                        .unwrap_or(std::cmp::Ordering::Equal),
                ),
            2 => left.start_players.cmp(&right.start_players),
            3 => left.start.cmp(&right.start),
            _ => left.end.cmp(&right.end),
        };
        if asc {
            ord
        } else {
            ord.reverse()
        }
    });
    order
}

fn hand_order(tournament: &ImportedTournament, sort: Option<(usize, bool)>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..tournament.hands.len()).collect();
    let Some((col, asc)) = sort else {
        return order;
    };
    order.sort_by(|&a, &b| {
        let left = &tournament.hands[a];
        let right = &tournament.hands[b];
        let ord = match col {
            0 => cmp_opt_f64(
                left.analysis.as_ref().map(|item| item.diff),
                right.analysis.as_ref().map(|item| item.diff),
            ),
            1 => left.hand.datetime.cmp(&right.hand.datetime),
            2 => left
                .hand
                .hero_combo_label
                .as_deref()
                .unwrap_or("")
                .cmp(right.hand.hero_combo_label.as_deref().unwrap_or("")),
            3 => left
                .hand
                .hero_action
                .map(|action| action.code())
                .unwrap_or("")
                .cmp(right.hand.hero_action.map(|action| action.code()).unwrap_or("")),
            4 => pos_rank(&left.hand).cmp(&pos_rank(&right.hand)),
            5 => cmp_f64(left.hand.hero_stack_bb(), right.hand.hero_stack_bb()),
            6 => cmp_f64(left.hand.effective_bb(), right.hand.effective_bb()),
            7 => left.hand.players.len().cmp(&right.hand.players.len()),
            _ => cmp_f64(left.hand.big_blind, right.hand.big_blind)
                .then(cmp_f64(left.hand.ante, right.hand.ante)),
        };
        if asc {
            ord
        } else {
            ord.reverse()
        }
    });
    order
}

fn pos_rank(hand: &ImportedHand) -> u8 {
    hand.hero_player()
        .and_then(|player| solver_position_index_for_sort(player.position, hand.players.len()))
        .unwrap_or(99)
}

fn solver_position_index_for_sort(
    position: poker_core::Position,
    players: usize,
) -> Option<u8> {
    poker_core::solver_position_index(position, players).map(|index| index as u8)
}

fn cmp_f64(a: f64, b: f64) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
}

fn cmp_f64_str(a: &str, b: &str) -> std::cmp::Ordering {
    let parse = |text: &str| text.parse::<f64>().ok();
    cmp_opt_f64(parse(a), parse(b))
}

fn cmp_opt_f64(a: Option<f64>, b: Option<f64>) -> std::cmp::Ordering {
    match (a, b) {
        (Some(x), Some(y)) => cmp_f64(x, y),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

fn stretched_cols(ui: &Ui, cols: &[f32]) -> Vec<f32> {
    let mut widths = cols.to_vec();
    let extra = (ui.available_width() - widths.iter().sum::<f32>()).max(0.0);
    if let Some(last) = widths.last_mut() {
        *last += extra;
    }
    widths
}

fn draw_grid_lines(ui: &Ui, rect: egui::Rect, widths: &[f32]) {
    let mut x = rect.left();
    for (index, width) in widths.iter().enumerate() {
        x += *width;
        if index + 1 < widths.len() {
            ui.painter().line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                egui::Stroke::new(1.0_f32, GRID),
            );
        }
    }
}

fn draw_header(
    ui: &mut Ui,
    cols: &[f32],
    labels: &[&str],
    sort: Option<(usize, bool)>,
) -> Option<usize> {
    let widths = stretched_cols(ui, cols);
    let width = widths.iter().sum::<f32>().max(1.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, ROW_H), Sense::hover());
    ui.painter().rect_filled(rect, 0.0, HEADER);
    let font = FontId::proportional(12.0);
    let mut clicked = None;
    let mut x = rect.left();
    for (index, (label, col)) in labels.iter().zip(widths.iter()).enumerate() {
        let col_rect = egui::Rect::from_min_size(egui::pos2(x, rect.top()), egui::vec2(*col, ROW_H));
        let response = ui.interact(col_rect, ui.id().with("import_hdr").with(index), Sense::click());
        if response.clicked() {
            clicked = Some(index);
        }
        let text = match sort {
            Some((col_index, true)) if col_index == index => format!("{label} ▲"),
            Some((col_index, false)) if col_index == index => format!("{label} ▼"),
            _ => (*label).to_string(),
        };
        ui.painter().text(
            egui::pos2(x + 6.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            text,
            font.clone(),
            Color32::WHITE,
        );
        x += *col;
    }
    draw_grid_lines(ui, rect, &widths);
    clicked
}

fn draw_row(
    ui: &mut Ui,
    cols: &[f32],
    values: &[&str],
    selected: bool,
    verdict: Option<Verdict>,
) -> egui::Response {
    let widths = stretched_cols(ui, cols);
    let width = widths.iter().sum::<f32>().max(1.0);
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, ROW_H), Sense::click());
    let bg = if selected {
        ROW_SEL
    } else if response.hovered() {
        ROW_HOVER
    } else {
        ROW_BG
    };
    ui.painter().rect_filled(rect, 0.0, bg);
    let font = FontId::proportional(12.0);
    let mut x = rect.left() + 6.0;
    let mut cursor = rect.left();
    for (i, (value, col)) in values.iter().zip(widths.iter()).enumerate() {
        let mut color = TEXT;
        let mut text = (*value).to_string();
        if i == 0 {
            if let Some(verdict) = verdict {
                let mark = match verdict {
                    Verdict::Correct => "✔",
                    Verdict::Warning => "!",
                    Verdict::Mistake => "✘",
                };
                color = match verdict {
                    Verdict::Correct => GREEN,
                    Verdict::Warning => YELLOW,
                    Verdict::Mistake => RED,
                };
                if !text.is_empty() {
                    text = format!("{mark} {text}");
                } else {
                    text = mark.to_string();
                }
            }
        }
        ui.painter().text(
            egui::pos2(x, rect.center().y),
            egui::Align2::LEFT_CENTER,
            text,
            font.clone(),
            color,
        );
        cursor += *col;
        x = cursor + 6.0;
    }
    draw_grid_lines(ui, rect, &widths);
    response
}
