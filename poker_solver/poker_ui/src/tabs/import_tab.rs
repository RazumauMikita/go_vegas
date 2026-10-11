use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use egui::popup::PopupCloseBehavior;
use egui::{Align2, Color32, Context, FontId, RichText, Sense, Stroke, Ui};
use poker_core::{
    eight_node, five_node, nine_node, parse_hand_histories, seven_node, six_node, solve,
    solver_position_index, solver_seat_label, table_position_label, Algorithm, EquityCache,
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
const GRID: Color32 = Color32::from_rgb(176, 192, 208);
const ROW_H: f32 = 22.0;
const PRIZE_SLOTS: usize = 9;
const WARN_YELLOW: Color32 = Color32::from_rgb(255, 196, 0);
const MISTAKE_RED: Color32 = Color32::from_rgb(204, 48, 48);

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
    reviewed: bool,
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum DiffMark {
    None,
    Excluded,
    Scored(Verdict),
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
    prizes_total: String,
    places_paid: String,
    total_chips: String,
    bounty_mode: BountyMode,
    warning: f32,
    mistake: f32,
    max_bb: f32,
    fold_allin_blinds: bool,
    multithread: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BountyMode {
    Off,
    Regular,
    Progressive,
}

impl BountyMode {
    const ALL: [Self; 3] = [Self::Off, Self::Regular, Self::Progressive];

    fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Regular => "Regular",
            Self::Progressive => "Progressive",
        }
    }
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
                            row.reviewed = true;
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
                    reviewed: false,
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
            reviewed: false,
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
                        DiffMark::None,
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
                    let mark = match (&row.analysis, row.reviewed) {
                        (Some(analysis), _) => DiffMark::Scored(analysis.verdict),
                        (None, true) => DiffMark::Excluded,
                        (None, false) => DiffMark::None,
                    };
                    let response = draw_row(ui, &cols, &values, index == selected, mark);
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
        let reviewed = row.reviewed;
        let Some(analysis) = row.analysis.clone() else {
            ui.vertical_centered(|ui| {
                ui.add_space(24.0);
                ui.label(if reviewed {
                    "Спот не подходит под push/fold — расчёт не выполняется."
                } else {
                    "Правый клик по турниру → Quick Analyze, чтобы рассчитать руки."
                });
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
            draw_row(ui, &cols, &values, name == hand.hero_name, DiffMark::None);
        }
    }

    fn open_analyze_modal(&mut self, tournament_index: usize, hand_index: Option<usize>) {
        let Some(tournament) = self.tournaments.get(tournament_index) else {
            return;
        };
        let start_players = tournament.start_players.max(2);
        let prize_pool = tournament.buy_in * start_players as f64;
        let pcts = if start_players <= 3 {
            vec![65.0, 35.0]
        } else {
            vec![50.0, 30.0, 20.0]
        };
        let prizes = if let Some(settings) = &self.last_settings {
            pad_prize_rows(
                settings
                    .prizes
                    .iter()
                    .map(|(dollars, pct)| {
                        if *dollars == 0.0 && *pct == 0.0 {
                            (String::new(), String::new())
                        } else {
                            (format!("{dollars:.1}"), format!("{pct:.1}"))
                        }
                    })
                    .collect(),
            )
        } else {
            pad_prize_rows(
                pcts.iter()
                    .map(|pct| {
                        (
                            format!("{:.1}", prize_pool * pct / 100.0),
                            format!("{pct:.1}"),
                        )
                    })
                    .collect(),
            )
        };
        let start_stack = tournament
            .hands
            .first()
            .and_then(|row| row.hand.hero_stack())
            .unwrap_or(1500.0);
        let settings = self.last_settings.clone();
        let paid = prizes
            .iter()
            .filter(|(dollars, pct)| !dollars.trim().is_empty() || !pct.trim().is_empty())
            .count()
            .max(1);
        let prizes_total = prize_dollar_sum(&prizes);
        self.analyze_modal = Some(AnalyzeModal {
            tournament_index,
            hand_index,
            prizes,
            prize_pool,
            prizes_total: format!("{prizes_total:.1}"),
            places_paid: paid.to_string(),
            total_chips: format!("{:.0}", start_stack * start_players as f64),
            bounty_mode: BountyMode::Off,
            warning: settings.as_ref().map(|s| s.warning as f32).unwrap_or(0.10),
            mistake: settings.as_ref().map(|s| s.mistake as f32).unwrap_or(0.10),
            max_bb: settings.as_ref().map(|s| s.max_bb as f32).unwrap_or(15.0),
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
        let screen = ctx.screen_rect();
        egui::Area::new(egui::Id::new("qa_modal_dim"))
            .order(egui::Order::Middle)
            .fixed_pos(screen.min)
            .interactable(true)
            .show(ctx, |ui| {
                let (rect, _) = ui.allocate_exact_size(screen.size(), Sense::click());
                ui.painter()
                    .rect_filled(rect, 0.0, Color32::from_black_alpha(90));
            });

        let mut open = true;
        let mut confirmed = false;
        let mut cancelled = false;
        egui::Window::new("Quick Analyze Settings")
            .id(egui::Id::new("qa_settings_window"))
            .order(egui::Order::Foreground)
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .default_size(egui::vec2(572.0, 410.0))
            .show(ctx, |ui| {
                ui.set_min_width(546.0);
                ui.spacing_mut().item_spacing = egui::vec2(8.0, 6.0);
                ui.horizontal(|ui| {
                    win_group_box(ui, "ICM Prizes", egui::vec2(292.0, 318.0), |ui| {
                        modal.draw_icm_prizes(ui);
                    });
                    win_group_box(ui, "Options", egui::vec2(246.0, 318.0), |ui| {
                        modal.draw_options(ui);
                    });
                });
                ui.add_space(4.0);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_sized(egui::vec2(80.0, 26.0), egui::Button::new("Cancel"))
                        .clicked()
                    {
                        cancelled = true;
                    }
                    ui.add_space(6.0);
                    if ui
                        .add_sized(egui::vec2(80.0, 26.0), egui::Button::new("OK"))
                        .clicked()
                    {
                        confirmed = true;
                    }
                });
            });

        if confirmed {
            modal.apply_places_paid();
            modal.apply_prizes_total();
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
            if dollars.trim().is_empty() && pct.trim().is_empty() {
                continue;
            }
            let dollars = dollars
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("некорректный приз ${dollars}"))?;
            let pct = pct
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("некорректный % {pct}"))?;
            if dollars == 0.0 && pct == 0.0 {
                continue;
            }
            prizes.push((dollars, pct));
        }
        if prizes.is_empty() {
            return Err("укажите хотя бы одно призовое место".into());
        }
        Ok(AnalyzeSettings {
            prizes,
            warning: self.warning.max(0.0) as f64,
            mistake: self.mistake.abs().max(self.warning) as f64,
            max_bb: self.max_bb.max(1.0) as f64,
            fold_allin_blinds: self.fold_allin_blinds,
            multithread: self.multithread,
        })
    }

    fn draw_icm_prizes(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.set_width(32.0);
                ui.spacing_mut().item_spacing = egui::vec2(0.0, 3.0);
                self.draw_prize_toolbar(ui);
            });
            ui.add_space(6.0);
            ui.vertical(|ui| {
                self.draw_prize_table(ui);
                ui.add_space(8.0);
                self.draw_prize_summary(ui);
            });
        });
    }

    fn draw_prize_toolbar(&mut self, ui: &mut Ui) {
        let new_btn = toolbar_icon_button(ui, "New prize structure", |painter, rect| {
            paint_doc_icon(painter, rect);
            let tip = rect.right_bottom() - egui::vec2(1.0, 2.0);
            painter.add(egui::Shape::convex_polygon(
                vec![
                    tip + egui::vec2(-5.0, -1.0),
                    tip + egui::vec2(1.0, -1.0),
                    tip + egui::vec2(-2.0, 3.0),
                ],
                painter.ctx().style().visuals.text_color(),
                Stroke::NONE,
            ));
        });
        let popup_id = ui.make_persistent_id("qa_prize_presets");
        if new_btn.clicked() {
            ui.memory_mut(|mem| mem.toggle_popup(popup_id));
        }
        egui::popup::popup_below_widget(
            ui,
            popup_id,
            &new_btn,
            PopupCloseBehavior::CloseOnClick,
            |ui| {
                ui.set_min_width(168.0);
                if ui.button("SNG  50 / 30 / 20").clicked() {
                    self.apply_preset(&[50.0, 30.0, 20.0]);
                }
                if ui.button("Heads-Up  65 / 35").clicked() {
                    self.apply_preset(&[65.0, 35.0]);
                }
                if ui.button("Winner Take All").clicked() {
                    self.apply_preset(&[100.0]);
                }
                if ui.button("Clear").clicked() {
                    self.apply_preset(&[]);
                    if let Some(first) = self.prizes.first_mut() {
                        *first = ("0.0".into(), "0.0".into());
                    }
                    self.places_paid = "1".into();
                }
            },
        );

        if toolbar_icon_button(ui, "Open prize structure", paint_folder_icon).clicked() {
            self.load_prizes();
        }
        if toolbar_icon_button(ui, "Save prize structure", paint_save_icon).clicked() {
            self.save_prizes();
        }
        if toolbar_icon_button(ui, "Copy prize structure", paint_copy_icon).clicked() {
            self.copy_prizes();
        }
    }

    fn draw_prize_table(&mut self, ui: &mut Ui) {
        if self.prizes.len() != PRIZE_SLOTS {
            self.prizes = pad_prize_rows(std::mem::take(&mut self.prizes));
        }
        let cols = [28.0, 92.0, 68.0];
        let total_w = cols.iter().sum::<f32>();
        let header_h = 22.0;
        let (header, _) = ui.allocate_exact_size(egui::vec2(total_w, header_h), Sense::hover());
        let header_fill = ui.visuals().widgets.inactive.bg_fill;
        let grid = ui.visuals().widgets.noninteractive.bg_stroke.color;
        ui.painter().rect_filled(header, 0.0, header_fill);
        let font = FontId::proportional(12.0);
        let color = ui.visuals().text_color();
        let labels = ["", "Prize ($)", "%"];
        let mut x = header.left();
        for (label, width) in labels.iter().zip(cols.iter()) {
            ui.painter().text(
                egui::pos2(x + *width * 0.5, header.center().y),
                Align2::CENTER_CENTER,
                *label,
                font.clone(),
                color,
            );
            x += *width;
        }
        draw_table_grid(ui, header, &cols, grid);

        let body_h = ROW_H * 7.0;
        let pool = self.prize_pool;
        let mut pct_changed = false;
        let mut dollar_changed = false;
        egui::ScrollArea::vertical()
            .id_salt("qa_prize_scroll")
            .max_height(body_h)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
                for index in 0..PRIZE_SLOTS {
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(total_w, ROW_H), Sense::hover());
                    ui.painter()
                        .rect_filled(rect, 0.0, ui.visuals().extreme_bg_color);
                    ui.painter().text(
                        egui::pos2(rect.left() + cols[0] * 0.5, rect.center().y),
                        Align2::CENTER_CENTER,
                        format!("{}", index + 1),
                        font.clone(),
                        color,
                    );
                    let (dollars, pct) = &mut self.prizes[index];
                    let dollar_rect = egui::Rect::from_min_size(
                        egui::pos2(rect.left() + cols[0] + 2.0, rect.top() + 1.0),
                        egui::vec2(cols[1] - 4.0, ROW_H - 2.0),
                    );
                    let pct_rect = egui::Rect::from_min_size(
                        egui::pos2(rect.left() + cols[0] + cols[1] + 2.0, rect.top() + 1.0),
                        egui::vec2(cols[2] - 4.0, ROW_H - 2.0),
                    );
                    let dollar_edit = ui.put(
                        dollar_rect,
                        egui::TextEdit::singleline(dollars)
                            .frame(false)
                            .horizontal_align(egui::Align::Center)
                            .margin(egui::vec2(4.0, 2.0))
                            .id(ui.id().with("qa_dollar").with(index)),
                    );
                    let pct_edit = ui.put(
                        pct_rect,
                        egui::TextEdit::singleline(pct)
                            .frame(false)
                            .horizontal_align(egui::Align::Center)
                            .margin(egui::vec2(4.0, 2.0))
                            .id(ui.id().with("qa_pct").with(index)),
                    );
                    if pct_edit.changed() {
                        if let Ok(value) = pct.trim().parse::<f64>() {
                            *dollars = if value == 0.0 {
                                String::new()
                            } else {
                                format!("{:.1}", pool * value / 100.0)
                            };
                        }
                        pct_changed = true;
                    } else if dollar_edit.changed() && pool > 0.0 {
                        if let Ok(value) = dollars.trim().parse::<f64>() {
                            *pct = if value == 0.0 {
                                String::new()
                            } else {
                                format!("{:.1}", value / pool * 100.0)
                            };
                        }
                        dollar_changed = true;
                    }
                    draw_table_grid(ui, rect, &cols, grid);
                    ui.painter().rect_stroke(rect, 0.0, Stroke::new(1.0_f32, grid));
                }
            });
        if pct_changed || dollar_changed {
            self.refresh_totals();
        }
    }

    fn draw_prize_summary(&mut self, ui: &mut Ui) {
        let field_w = 108.0;
        egui::Grid::new("qa_icm_summary")
            .num_columns(2)
            .spacing(egui::vec2(10.0, 4.0))
            .min_col_width(88.0)
            .show(ui, |ui| {
                ui.label("Places Paid");
                let paid = ui.add(
                    egui::TextEdit::singleline(&mut self.places_paid)
                        .desired_width(field_w)
                        .horizontal_align(egui::Align::Center),
                );
                ui.end_row();
                ui.label("Prizes($)");
                let total = ui.add(
                    egui::TextEdit::singleline(&mut self.prizes_total)
                        .desired_width(field_w)
                        .horizontal_align(egui::Align::Center),
                );
                ui.end_row();
                ui.label("Total Chips");
                ui.add(
                    egui::TextEdit::singleline(&mut self.total_chips)
                        .desired_width(field_w)
                        .horizontal_align(egui::Align::Center),
                );
                ui.end_row();
                ui.label("Bounty Mode");
                egui::ComboBox::from_id_salt("qa_bounty_mode")
                    .selected_text(self.bounty_mode.label())
                    .width(field_w)
                    .show_ui(ui, |ui| {
                        for mode in BountyMode::ALL {
                            ui.selectable_value(&mut self.bounty_mode, mode, mode.label());
                        }
                    });
                ui.end_row();

                if paid.lost_focus() {
                    self.apply_places_paid();
                }
                if total.lost_focus() {
                    self.apply_prizes_total();
                }
            });
    }

    fn draw_options(&mut self, ui: &mut Ui) {
        ui.label("EQDiff% Warning:");
        ui.horizontal(|ui| {
            paint_warning_icon(ui, 18.0);
            ui.spacing_mut().slider_width = (ui.available_width() - 56.0).max(80.0);
            ui.add(
                egui::Slider::new(&mut self.warning, 0.0..=1.0)
                    .show_value(false)
                    .max_decimals(2)
                    .handle_shape(egui::style::HandleShape::Rect { aspect_ratio: 0.42 }),
            );
            ui.label(format!("{:+.2}", self.warning));
        });
        ui.add_space(8.0);
        ui.label("EQDiff% Mistake:");
        ui.horizontal(|ui| {
            paint_mistake_icon(ui, 18.0);
            ui.spacing_mut().slider_width = (ui.available_width() - 56.0).max(80.0);
            let mut mistake = -self.mistake;
            let slider = ui.add(
                egui::Slider::new(&mut mistake, -1.0..=0.0)
                    .show_value(false)
                    .max_decimals(2)
                    .handle_shape(egui::style::HandleShape::Rect { aspect_ratio: 0.42 }),
            );
            if slider.changed() {
                self.mistake = -mistake;
            }
            ui.label(format!("{:+.2}", -self.mistake));
        });
        ui.add_space(14.0);
        ui.label(RichText::new("Filters").strong());
        ui.horizontal(|ui| {
            ui.label("Max. effective BB:");
            ui.add(
                egui::DragValue::new(&mut self.max_bb)
                    .range(1.0..=200.0)
                    .speed(1.0)
                    .max_decimals(0),
            );
        });
        ui.checkbox(&mut self.fold_allin_blinds, "Strict all-in filter");
        ui.add_space(10.0);
        ui.label(RichText::new("Performance").strong());
        ui.checkbox(&mut self.multithread, "Use Multi-Threading");
    }

    fn apply_preset(&mut self, pcts: &[f64]) {
        self.prizes = pad_prize_rows(
            pcts.iter()
                .map(|pct| {
                    (
                        format!("{:.1}", self.prize_pool * pct / 100.0),
                        format!("{pct:.1}"),
                    )
                })
                .collect(),
        );
        self.places_paid = pcts.len().max(1).to_string();
        self.refresh_totals();
    }

    fn apply_places_paid(&mut self) {
        let Ok(n) = self.places_paid.trim().parse::<usize>() else {
            return;
        };
        let n = n.clamp(1, PRIZE_SLOTS);
        self.places_paid = n.to_string();
        for i in 0..PRIZE_SLOTS {
            if i >= n {
                self.prizes[i] = (String::new(), String::new());
            } else if self.prizes[i].0.trim().is_empty() && self.prizes[i].1.trim().is_empty() {
                self.prizes[i] = ("0.0".into(), "0.0".into());
            }
        }
        self.refresh_totals();
    }

    fn apply_prizes_total(&mut self) {
        let Ok(value) = self.prizes_total.trim().parse::<f64>() else {
            self.refresh_totals();
            return;
        };
        if value <= 0.0 {
            self.refresh_totals();
            return;
        }
        self.prize_pool = value;
        for (dollars, pct) in &mut self.prizes {
            if let Ok(p) = pct.trim().parse::<f64>() {
                *dollars = if p == 0.0 {
                    String::new()
                } else {
                    format!("{:.1}", value * p / 100.0)
                };
            }
        }
        self.refresh_totals();
    }

    fn refresh_totals(&mut self) {
        self.prizes_total = format!("{:.1}", prize_dollar_sum(&self.prizes));
    }

    fn load_prizes(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Prize structure", &["txt"])
            .set_title("Open Prize Structure")
            .pick_file()
        else {
            return;
        };
        let Ok(text) = std::fs::read_to_string(path) else {
            return;
        };
        let mut loaded = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line
                .split(|c: char| c == ',' || c == ';' || c == '\t' || c.is_whitespace())
                .filter(|part| !part.is_empty())
                .collect();
            if parts.len() >= 2 {
                loaded.push((parts[0].to_string(), parts[1].to_string()));
            } else if let Some(part) = parts.first() {
                if let Ok(pct) = part.parse::<f64>() {
                    loaded.push((
                        format!("{:.1}", self.prize_pool * pct / 100.0),
                        format!("{pct:.1}"),
                    ));
                }
            }
        }
        if loaded.is_empty() {
            return;
        }
        self.prizes = pad_prize_rows(loaded);
        self.refresh_totals();
    }

    fn save_prizes(&self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Prize structure", &["txt"])
            .set_title("Save Prize Structure")
            .save_file()
        else {
            return;
        };
        let mut text = String::new();
        for (dollars, pct) in &self.prizes {
            if dollars.trim().is_empty() && pct.trim().is_empty() {
                continue;
            }
            text.push_str(&format!("{dollars},{pct}\n"));
        }
        let _ = std::fs::write(path, text);
    }

    fn copy_prizes(&self) {
        let text = self
            .prizes
            .iter()
            .filter(|(_, pct)| !pct.trim().is_empty())
            .map(|(_, pct)| pct.clone())
            .collect::<Vec<_>>()
            .join("/");
        let _ = arboard::Clipboard::new().and_then(|mut cb| cb.set_text(text));
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
    if !is_push_fold_spot(hand) {
        return None;
    }

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

    let spot = if (5..=9).contains(&n) {
        mask_spot(hand, &output)?
    } else {
        let actions_before = actions_before_hero(hand);
        let tree = build_strategy_tree(&output, input.button_index);
        let path = find_hero_decision_path(&tree, &actions_before, hero_label)?;
        let node = node_at_path(&tree, &path)?;
        Spot {
            range: node.range,
            evs: node.ev_range,
            title: node.label.clone(),
            is_raise: matches!(node.action, Action::Raise),
        }
    };
    let took_action = if spot.is_raise {
        matches!(hero_action, HeroActionKind::Raise)
    } else {
        matches!(hero_action, HeroActionKind::Call | HeroActionKind::Raise)
    };
    let freq = spot.range[combo].clamp(0.0, 1.0);
    let diff = if let Some(evs) = spot.evs {
        if took_action {
            evs[combo]
        } else {
            -evs[combo]
        }
    } else {
        let signed = if took_action { 1.0 } else { -1.0 };
        signed * (2.0 * freq - 1.0)
    };
    let verdict = if diff > settings.warning {
        Verdict::Correct
    } else if diff >= -settings.mistake {
        Verdict::Warning
    } else {
        Verdict::Mistake
    };
    Some(HandAnalysis {
        diff,
        verdict,
        range: spot.range,
        evs: spot.evs,
        title: spot.title,
        is_raise: spot.is_raise,
    })
}

struct Spot {
    range: [f64; 169],
    evs: Option<[f64; 169]>,
    title: String,
    is_raise: bool,
}

fn mask_spot(hand: &ImportedHand, output: &SolverOutput) -> Option<Spot> {
    let n = hand.players.len();
    let hero = hand.hero_player()?;
    let hero_seat = solver_position_index(hero.position, n)?;
    let (mask, shovers) = prior_shove_mask(hand, hero_seat);
    let node = node_index(n, hero_seat, mask)?;
    let (range, evs) = ranges_at(output, n, node)?;
    let is_raise = mask == 0;
    let hero_label = solver_seat_label(hero.position, n).unwrap_or("?");
    let title = if shovers.is_empty() {
        format!("{hero_label} push")
    } else {
        format!("{hero_label} call (vs {})", shovers.join("+"))
    };
    Some(Spot {
        range,
        evs: Some(evs),
        title,
        is_raise,
    })
}

fn prior_shove_mask(hand: &ImportedHand, hero_seat: usize) -> (u32, Vec<String>) {
    let n = hand.players.len();
    let mut mask = 0u32;
    let mut shovers = Vec::new();
    for action in &hand.preflop_actions {
        if action.name == hand.hero_name {
            break;
        }
        let Some(player) = hand.players.iter().find(|p| p.name == action.name) else {
            continue;
        };
        let Some(seat) = solver_position_index(player.position, n) else {
            continue;
        };
        if seat < hero_seat && !action.kind.is_fold() {
            mask |= 1 << seat;
            if let Some(label) = solver_seat_label(player.position, n) {
                shovers.push(label.to_string());
            }
        }
    }
    (mask, shovers)
}

fn node_index(players: usize, actor: usize, mask: u32) -> Option<usize> {
    Some(match players {
        5 => five_node(actor, mask),
        6 => six_node(actor, mask),
        7 => seven_node(actor, mask),
        8 => eight_node(actor, mask),
        9 => nine_node(actor, mask),
        _ => return None,
    })
}

fn ranges_at(output: &SolverOutput, players: usize, node: usize) -> Option<([f64; 169], [f64; 169])> {
    let (freq, ev) = match players {
        5 => output.five_max.as_ref().map(|r| (&r.freq, &r.ev)),
        6 => output.six_max.as_ref().map(|r| (&r.freq, &r.ev)),
        7 => output.seven_max.as_ref().map(|r| (&r.freq, &r.ev)),
        8 => output.eight_max.as_ref().map(|r| (&r.freq, &r.ev)),
        9 => output.nine_max.as_ref().map(|r| (&r.freq, &r.ev)),
        _ => None,
    }?;
    Some((*freq.get(node)?, *ev.get(node)?))
}

fn is_push_fold_spot(hand: &ImportedHand) -> bool {
    let Some(hero_action) = hand.hero_action else {
        return false;
    };
    if matches!(hero_action, HeroActionKind::Check) {
        return false;
    }

    let mut facing_all_in = false;
    let mut raised_short = false;
    for action in &hand.preflop_actions {
        let is_hero = action.name == hand.hero_name;
        match action.kind {
            HeroActionKind::Fold | HeroActionKind::Check => {}
            HeroActionKind::Raise => {
                if is_hero && !action.all_in {
                    return false;
                }
                if raised_short {
                    return false;
                }
                if action.all_in {
                    facing_all_in = true;
                } else {
                    raised_short = true;
                }
            }
            HeroActionKind::Call => {
                if facing_all_in || action.all_in {
                    facing_all_in = true;
                } else {
                    return false;
                }
            }
        }
    }
    true
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
    mark: DiffMark,
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
        if i == 0 && !matches!(mark, DiffMark::None) {
            let icon = egui::Rect::from_center_size(
                egui::pos2(x + 7.0, rect.center().y),
                egui::vec2(14.0, 14.0),
            );
            let number_color = match mark {
                DiffMark::Scored(Verdict::Correct) => GREEN,
                DiffMark::Scored(Verdict::Mistake) => RED,
                _ => TEXT,
            };
            match mark {
                DiffMark::Excluded => paint_excluded_mark(ui.painter(), icon),
                DiffMark::Scored(Verdict::Correct) => paint_correct_mark(ui.painter(), icon),
                DiffMark::Scored(Verdict::Warning) => paint_warning_mark(ui.painter(), icon),
                DiffMark::Scored(Verdict::Mistake) => paint_mistake_mark(ui.painter(), icon),
                DiffMark::None => {}
            }
            if !value.is_empty() && !matches!(mark, DiffMark::Excluded) {
                ui.painter().text(
                    egui::pos2(x + 18.0, rect.center().y),
                    egui::Align2::LEFT_CENTER,
                    *value,
                    font.clone(),
                    number_color,
                );
            }
        } else {
            ui.painter().text(
                egui::pos2(x, rect.center().y),
                egui::Align2::LEFT_CENTER,
                *value,
                font.clone(),
                TEXT,
            );
        }
        cursor += *col;
        x = cursor + 6.0;
    }
    draw_grid_lines(ui, rect, &widths);
    response
}

fn pad_prize_rows(mut prizes: Vec<(String, String)>) -> Vec<(String, String)> {
    prizes.truncate(PRIZE_SLOTS);
    while prizes.len() < PRIZE_SLOTS {
        prizes.push((String::new(), String::new()));
    }
    prizes
}

fn prize_dollar_sum(prizes: &[(String, String)]) -> f64 {
    prizes
        .iter()
        .filter_map(|(dollars, _)| dollars.trim().parse::<f64>().ok())
        .sum()
}

fn win_group_box(ui: &mut Ui, title: &str, size: egui::Vec2, add_contents: impl FnOnce(&mut Ui)) {
    let (_, rect) = ui.allocate_space(size);
    let painter = ui.painter_at(rect);
    let stroke = Stroke::new(1.0_f32, ui.visuals().widgets.noninteractive.bg_stroke.color);
    let top = rect.top() + 8.0;
    let border = egui::Rect::from_min_max(egui::pos2(rect.left(), top), rect.max);
    painter.rect_stroke(border, 2.0, stroke);

    let font = FontId::proportional(12.5);
    let color = ui.visuals().text_color();
    let galley = painter.layout_no_wrap(title.to_owned(), font, color);
    let title_pos = egui::pos2(rect.left() + 10.0, top - galley.size().y * 0.5);
    let bg = egui::Rect::from_min_size(
        title_pos - egui::vec2(4.0, 0.0),
        galley.size() + egui::vec2(8.0, 0.0),
    );
    painter.rect_filled(bg, 0.0, ui.visuals().window_fill());
    painter.galley(title_pos, galley, color);

    let inner = egui::Rect::from_min_max(
        egui::pos2(rect.left() + 8.0, top + 10.0),
        egui::pos2(rect.right() - 8.0, rect.bottom() - 8.0),
    );
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(inner)
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    add_contents(&mut child);
}

fn toolbar_icon_button(
    ui: &mut Ui,
    hover: &str,
    paint: impl FnOnce(&egui::Painter, egui::Rect),
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(30.0, 26.0), Sense::click());
    let response = response.on_hover_text(hover);
    let visuals = ui.style().interact(&response);
    if response.hovered() || response.has_focus() {
        ui.painter().rect_filled(rect, 3.0, visuals.weak_bg_fill);
        ui.painter()
            .rect_stroke(rect, 3.0, Stroke::new(1.0_f32, visuals.bg_stroke.color));
    }
    paint(ui.painter(), rect.shrink2(egui::vec2(6.0, 5.0)));
    response
}

fn paint_doc_icon(painter: &egui::Painter, rect: egui::Rect) {
    let color = painter.ctx().style().visuals.text_color();
    let stroke = Stroke::new(1.2_f32, color);
    let fold = 5.0;
    let body = egui::Rect::from_min_max(rect.min, rect.max - egui::vec2(0.0, 1.0));
    painter.rect_stroke(body, 1.0, stroke);
    painter.line_segment(
        [
            egui::pos2(body.right() - fold, body.top()),
            egui::pos2(body.right() - fold, body.top() + fold),
        ],
        stroke,
    );
    painter.line_segment(
        [
            egui::pos2(body.right() - fold, body.top() + fold),
            egui::pos2(body.right(), body.top() + fold),
        ],
        stroke,
    );
    painter.line_segment(
        [
            egui::pos2(body.right() - fold, body.top()),
            egui::pos2(body.right(), body.top() + fold),
        ],
        stroke,
    );
}

fn paint_folder_icon(painter: &egui::Painter, rect: egui::Rect) {
    let color = painter.ctx().style().visuals.text_color();
    let stroke = Stroke::new(1.2_f32, color);
    let tab = rect.height() * 0.28;
    painter.rect_stroke(
        egui::Rect::from_min_max(
            egui::pos2(rect.left(), rect.top() + tab),
            rect.max,
        ),
        1.0,
        stroke,
    );
    painter.line_segment(
        [
            rect.left_top() + egui::vec2(0.0, tab),
            rect.left_top() + egui::vec2(0.0, tab * 0.2),
        ],
        stroke,
    );
    painter.line_segment(
        [
            rect.left_top() + egui::vec2(0.0, tab * 0.2),
            rect.left_top() + egui::vec2(rect.width() * 0.45, tab * 0.2),
        ],
        stroke,
    );
    painter.line_segment(
        [
            rect.left_top() + egui::vec2(rect.width() * 0.45, tab * 0.2),
            rect.left_top() + egui::vec2(rect.width() * 0.55, tab),
        ],
        stroke,
    );
}

fn paint_save_icon(painter: &egui::Painter, rect: egui::Rect) {
    let color = painter.ctx().style().visuals.text_color();
    let stroke = Stroke::new(1.2_f32, color);
    painter.rect_stroke(rect, 1.0, stroke);
    let label = egui::Rect::from_min_max(
        egui::pos2(rect.left() + 3.0, rect.top() + 1.5),
        egui::pos2(rect.right() - 3.0, rect.top() + rect.height() * 0.42),
    );
    painter.rect_stroke(label, 0.0, stroke);
    painter.line_segment(
        [
            egui::pos2(rect.left() + 4.0, rect.bottom() - 4.0),
            egui::pos2(rect.right() - 4.0, rect.bottom() - 4.0),
        ],
        stroke,
    );
}

fn paint_copy_icon(painter: &egui::Painter, rect: egui::Rect) {
    let color = painter.ctx().style().visuals.text_color();
    let stroke = Stroke::new(1.2_f32, color);
    let back = egui::Rect::from_min_max(rect.min, rect.max - egui::vec2(3.0, 3.0));
    let front = egui::Rect::from_min_max(rect.min + egui::vec2(3.0, 3.0), rect.max);
    painter.rect_stroke(back, 1.0, stroke);
    painter.rect_filled(front, 1.0, painter.ctx().style().visuals.window_fill());
    painter.rect_stroke(front, 1.0, stroke);
}

fn paint_warning_icon(ui: &mut Ui, size: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), Sense::hover());
    paint_warning_mark(ui.painter(), rect);
}

fn paint_mistake_icon(ui: &mut Ui, size: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), Sense::hover());
    paint_mistake_mark(ui.painter(), rect);
}

fn paint_correct_mark(painter: &egui::Painter, rect: egui::Rect) {
    let stroke = Stroke::new(2.0_f32, GREEN);
    let mid = egui::pos2(
        rect.left() + rect.width() * 0.38,
        rect.bottom() - rect.height() * 0.28,
    );
    painter.line_segment(
        [
            egui::pos2(rect.left() + rect.width() * 0.12, rect.center().y),
            mid,
        ],
        stroke,
    );
    painter.line_segment(
        [
            mid,
            egui::pos2(
                rect.right() - rect.width() * 0.08,
                rect.top() + rect.height() * 0.22,
            ),
        ],
        stroke,
    );
}

fn paint_excluded_mark(painter: &egui::Painter, rect: egui::Rect) {
    let stroke = Stroke::new(1.7_f32, Color32::from_rgb(148, 154, 162));
    let r = rect.shrink(2.2);
    painter.line_segment([r.left_top(), r.right_bottom()], stroke);
    painter.line_segment([r.right_top(), r.left_bottom()], stroke);
}

fn paint_warning_mark(painter: &egui::Painter, rect: egui::Rect) {
    let c = rect.center();
    let h = rect.height() * 0.46;
    painter.add(egui::Shape::convex_polygon(
        vec![
            egui::pos2(c.x, c.y - h),
            egui::pos2(c.x + h * 0.92, c.y + h * 0.72),
            egui::pos2(c.x - h * 0.92, c.y + h * 0.72),
        ],
        WARN_YELLOW,
        Stroke::NONE,
    ));
    painter.text(
        c + egui::vec2(0.0, rect.height() * 0.08),
        Align2::CENTER_CENTER,
        "!",
        FontId::proportional(rect.height() * 0.62),
        Color32::BLACK,
    );
}

fn paint_mistake_mark(painter: &egui::Painter, rect: egui::Rect) {
    let c = rect.center();
    let radius = rect.height() * 0.46;
    painter.circle_filled(c, radius, MISTAKE_RED);
    let s = radius * 0.38;
    let stroke = Stroke::new((rect.height() * 0.12).max(1.4), Color32::WHITE);
    painter.line_segment([c + egui::vec2(-s, -s), c + egui::vec2(s, s)], stroke);
    painter.line_segment([c + egui::vec2(s, -s), c + egui::vec2(-s, s)], stroke);
}

#[cfg(test)]
mod spot_filter_tests {
    use super::{is_push_fold_spot, prior_shove_mask};
    use poker_core::{parse_imported_hand, solver_position_index};

    const OPEN_THEN_SHOVE: &str = r"PokerStars Hand #262330970830: Tournament #4038016204, $0.42+$0.08 USD Hold'em No Limit - Level V (75/150) - 2026/10/07 20:59:08 ET
Table '4038016204 1' 9-max Seat #8 is the button
Seat 2: wagnerrocha (890 in chips) 
Seat 4: mkHasNutzz (3420 in chips) 
Seat 5: MosherA (4472 in chips) 
Seat 6: bbcbcv3 (945 in chips) 
Seat 8: boladoph (3773 in chips) 
wagnerrocha: posts small blind 75
mkHasNutzz: posts big blind 150
*** HOLE CARDS ***
Dealt to bbcbcv3 [9d 8d]
MosherA: folds 
bbcbcv3: folds 
boladoph: raises 300 to 450
wagnerrocha: raises 440 to 890 and is all-in
mkHasNutzz: folds 
boladoph: calls 440
*** FLOP *** [Jd Th 8h]
*** SUMMARY ***
Total pot 1930 | Rake 0 
";

    const TWO_SHOVS: &str = r"PokerStars Hand #262330946442: Tournament #4038016204, $0.42+$0.08 USD Hold'em No Limit - Level V (75/150) - 2026/10/07 20:54:29 ET
Table '4038016204 1' 9-max Seat #9 is the button
Seat 1: guridebarba (859 in chips) 
Seat 2: wagnerrocha (1265 in chips) 
Seat 3: shadestacian (1087 in chips) 
Seat 4: mkHasNutzz (3046 in chips) 
Seat 5: MosherA (1363 in chips) 
Seat 6: bbcbcv3 (1395 in chips) 
Seat 8: boladoph (3140 in chips) 
Seat 9: whorvatte (1345 in chips) 
guridebarba: posts small blind 75
wagnerrocha: posts big blind 150
*** HOLE CARDS ***
Dealt to bbcbcv3 [Td 6s]
shadestacian: raises 937 to 1087 and is all-in
mkHasNutzz: folds 
MosherA: raises 276 to 1363 and is all-in
bbcbcv3: folds 
boladoph: folds 
whorvatte: folds 
guridebarba: calls 784 and is all-in
wagnerrocha: folds 
*** SUMMARY ***
Total pot 3183 | Rake 0 
";

    #[test]
    fn minraise_then_shove_is_not_push_fold() {
        let hand = parse_imported_hand(OPEN_THEN_SHOVE).unwrap();
        assert!(!is_push_fold_spot(&hand));
    }

    #[test]
    fn two_all_in_raises_before_hero_fold_is_push_fold() {
        let hand = parse_imported_hand(TWO_SHOVS).unwrap();
        assert!(is_push_fold_spot(&hand));
        let hero = hand.hero_player().unwrap();
        let seat = solver_position_index(hero.position, hand.players.len()).unwrap();
        let (mask, shovers) = prior_shove_mask(&hand, seat);
        assert_eq!(seat, 3);
        assert_eq!(mask, 0b101);
        assert_eq!(shovers, vec!["UTG".to_string(), "MP".to_string()]);
    }
}

fn draw_table_grid(ui: &Ui, rect: egui::Rect, cols: &[f32], color: Color32) {
    let stroke = Stroke::new(1.0_f32, color);
    let mut x = rect.left();
    for (index, width) in cols.iter().enumerate() {
        x += *width;
        if index + 1 < cols.len() {
            ui.painter()
                .line_segment([egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())], stroke);
        }
    }
}
