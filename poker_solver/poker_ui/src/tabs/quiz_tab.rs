use std::f32::consts::{FRAC_PI_2, TAU};
use std::str::FromStr;
use std::sync::mpsc;
use std::thread;

use egui::{Align2, Color32, Context, FontId, RichText, Sense, Stroke, Ui};
use poker_core::{solve, table_position_label, Card, EquityCache, SolverOutput};

use crate::quiz::{
    dealer_seat, generate_quiz, grade_answer, grade_range, hero_seat_index, solver_input_for,
    QuizCase, QuizLibrary, QuizSettings, QuizSourceKind, QuizTaskMode, QuizVerdict,
    DEFAULT_CATEGORY,
};
use crate::tabs::strategy_tree::{
    build_strategy_tree, display_player, display_vs, find_hero_decision_path, node_at_path,
    same_player, TreeNode,
};
use crate::widgets::{
    combo_share, fill_range_to_share, range_matrix_ui, range_notation, rank_hands_for_slider,
    MatrixMode,
};

const CACHE_BYTES: &[u8] = include_bytes!("../../assets/equity_cache.bin");
const HEADER: Color32 = Color32::from_rgb(70, 141, 196);
const GREEN: Color32 = Color32::from_rgb(46, 160, 67);
const RED: Color32 = Color32::from_rgb(200, 60, 60);
const MIXED: Color32 = Color32::from_rgb(210, 150, 20);
const FELT: Color32 = Color32::from_rgb(26, 92, 52);
const FELT_EDGE: Color32 = Color32::from_rgb(196, 154, 62);
const TEXT: Color32 = Color32::from_rgb(32, 32, 32);
const TEXT_FAINT: Color32 = Color32::from_rgb(120, 120, 120);
const SEAT_FOLD: Color32 = Color32::from_rgb(188, 188, 188);
const SEAT_IN: Color32 = Color32::from_rgb(206, 236, 206);
const CHIP_SB: Color32 = Color32::from_rgb(228, 228, 236);
const CHIP_BB: Color32 = Color32::from_rgb(236, 196, 64);
const CHIP_SHOVE: Color32 = Color32::from_rgb(196, 56, 56);
const CHIP_CALL: Color32 = Color32::from_rgb(56, 132, 220);

enum QuizView {
    Lobby,
    Playing,
    Results,
}

struct QuizSession {
    spots: Vec<QuizCase>,
    index: usize,
    verdicts: Vec<Option<QuizVerdict>>,
    choices: Vec<Option<bool>>,
    range_scores: Vec<Option<f64>>,
    worker_rx: Option<mpsc::Receiver<SolveMessage>>,
    computing: bool,
    solution: Option<SolvedSpot>,
    revealed: bool,
    error: Option<String>,
    matrix_mode: MatrixMode,
    task_mode: QuizTaskMode,
    user_range: [f64; 169],
    slider_pct: f64,
    ranking: Vec<usize>,
}

#[derive(Clone)]
struct VillainRange {
    label: String,
    range: [f64; 169],
}

#[derive(Clone)]
struct SolvedSpot {
    range: [f64; 169],
    evs: Option<[f64; 169]>,
    title: String,
    is_raise: bool,
    freq: f64,
    ev: Option<f64>,
    villains: Vec<VillainRange>,
    callers: Vec<VillainRange>,
}

enum SolveMessage {
    Done(SolvedSpot),
    Failed(String),
}

struct PlayView {
    computing: bool,
    revealed: bool,
    can_answer: bool,
    action_name: &'static str,
    error: Option<String>,
    solution: Option<SolvedSpot>,
    combo: usize,
    hero_index: usize,
    cards: Option<[String; 2]>,
    combo_label_text: String,
    actions: Vec<(String, bool)>,
    stacks: Vec<f64>,
    n: usize,
    sb: f64,
    bb: f64,
    hero_label: String,
    title: String,
    task_mode: QuizTaskMode,
}

impl PlayView {
    fn from_session(session: Option<&QuizSession>) -> Option<Self> {
        let session = session?;
        let spot = session.spots.get(session.index)?;
        let computing = session.computing;
        let revealed = session.revealed;
        let can_answer = !computing && !revealed && session.solution.is_some();
        let is_raise = session
            .solution
            .as_ref()
            .map(|s| s.is_raise)
            .unwrap_or(true);
        let solution = session.solution.clone();
        let hero_label = spot.hero_label.clone();
        let title = solution
            .as_ref()
            .map(|s| s.title.clone())
            .unwrap_or_else(|| format!("{} to act", display_player(&hero_label)));
        Some(Self {
            computing,
            revealed,
            can_answer,
            action_name: if is_raise { "Push" } else { "Call" },
            error: session.error.clone(),
            solution,
            combo: spot.hero_combo as usize,
            hero_index: hero_seat_index(spot.player_count, &spot.hero_label).unwrap_or(0),
            cards: spot.hero_cards.clone(),
            combo_label_text: spot.hero_combo_label.clone(),
            actions: spot.actions_before.clone(),
            stacks: spot.stacks.clone(),
            n: spot.player_count,
            sb: spot.small_blind,
            bb: spot.big_blind,
            hero_label,
            title,
            task_mode: session.task_mode,
        })
    }
}

pub struct QuizTab {
    library: QuizLibrary,
    view: QuizView,
    setup_open: bool,
    setup: QuizSettings,
    session: Option<QuizSession>,
    selected_category: String,
    selected_case: Option<String>,
    new_category: String,
    error: Option<String>,
    status: String,
}

impl Default for QuizTab {
    fn default() -> Self {
        let library = QuizLibrary::load();
        Self {
            selected_category: library
                .categories()
                .into_iter()
                .next()
                .unwrap_or_else(|| DEFAULT_CATEGORY.to_string()),
            library,
            view: QuizView::Lobby,
            setup_open: false,
            setup: QuizSettings::default(),
            session: None,
            selected_case: None,
            new_category: String::new(),
            error: None,
            status: String::new(),
        }
    }
}

impl QuizTab {
    pub fn poll(&mut self, ctx: &Context) {
        if matches!(self.view, QuizView::Lobby) {
            self.library = QuizLibrary::load();
        }
        let Some(session) = self.session.as_mut() else {
            return;
        };
        if let Some(rx) = &session.worker_rx {
            match rx.try_recv() {
                Ok(SolveMessage::Done(solution)) => {
                    session.ranking = rank_hands_for_slider(&solution.range, solution.evs.as_ref());
                    session.slider_pct = combo_share(&session.user_range) * 100.0;
                    session.solution = Some(solution);
                    session.computing = false;
                    session.worker_rx = None;
                    session.error = None;
                    ctx.request_repaint();
                }
                Ok(SolveMessage::Failed(message)) => {
                    session.error = Some(message);
                    session.computing = false;
                    session.worker_rx = None;
                    ctx.request_repaint();
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    if session.computing {
                        session.error = Some("поток расчёта завершился неожиданно".into());
                        session.computing = false;
                    }
                    session.worker_rx = None;
                    ctx.request_repaint();
                }
                Err(mpsc::TryRecvError::Empty) => {
                    if session.computing {
                        ctx.request_repaint();
                    }
                }
            }
        }
    }

    pub fn ui(&mut self, ctx: &Context) {
        self.handle_shortcuts(ctx);
        self.draw_setup_modal(ctx);

        egui::TopBottomPanel::top("quiz_toolbar")
            .resizable(false)
            .frame(
                egui::Frame::none()
                    .fill(ctx.style().visuals.extreme_bg_color)
                    .inner_margin(egui::Margin::symmetric(8.0, 6.0)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.set_min_height(24.0);
                    match self.view {
                        QuizView::Lobby => {
                            if ui.button("Новый квиз").clicked() {
                                self.open_setup();
                            }
                        }
                        QuizView::Playing => {
                            if let Some(session) = &self.session {
                                ui.label(format!(
                                    "Вопрос {} / {}",
                                    session.index + 1,
                                    session.spots.len()
                                ));
                                ui.separator();
                                ui.label(score_label(session));
                            }
                            if ui.button("Завершить").clicked() {
                                self.view = QuizView::Results;
                            }
                        }
                        QuizView::Results => {
                            if ui.button("В лобби").clicked() {
                                self.session = None;
                                self.view = QuizView::Lobby;
                            }
                            if ui.button("Ещё раз").clicked() {
                                self.restart_quiz(ctx);
                            }
                        }
                    }
                    if let Some(error) = &self.error {
                        ui.colored_label(Color32::RED, error);
                    } else if !self.status.is_empty() && matches!(self.view, QuizView::Lobby) {
                        ui.label(RichText::new(&self.status).small());
                    }
                });
            });

        match self.view {
            QuizView::Lobby => self.draw_lobby(ctx),
            QuizView::Playing => self.draw_play(ctx),
            QuizView::Results => self.draw_results(ctx),
        }
    }

    fn open_setup(&mut self) {
        self.setup.category = self.selected_category.clone();
        if !self.library.cases_in(&self.selected_category).is_empty()
            && self.setup.source == QuizSourceKind::Saved
        {
            self.setup.category = self.selected_category.clone();
        }
        self.setup_open = true;
        self.error = None;
    }

    fn start_quiz(&mut self, ctx: &Context) {
        match generate_quiz(&self.setup, &self.library) {
            Ok(spots) => {
                let mut session = QuizSession {
                    verdicts: vec![None; spots.len()],
                    choices: vec![None; spots.len()],
                    range_scores: vec![None; spots.len()],
                    spots,
                    index: 0,
                    worker_rx: None,
                    computing: false,
                    solution: None,
                    revealed: false,
                    error: None,
                    matrix_mode: MatrixMode::Frequency,
                    task_mode: self.setup.task_mode,
                    user_range: [0.0; 169],
                    slider_pct: 0.0,
                    ranking: Vec::new(),
                };
                start_solve(&mut session, ctx);
                self.session = Some(session);
                self.view = QuizView::Playing;
                self.setup_open = false;
                self.error = None;
            }
            Err(message) => self.error = Some(message),
        }
    }

    fn restart_quiz(&mut self, ctx: &Context) {
        self.start_quiz(ctx);
    }

    fn handle_shortcuts(&mut self, ctx: &Context) {
        if !matches!(self.view, QuizView::Playing) {
            return;
        }
        let (fold, action, next) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::F),
                i.key_pressed(egui::Key::P) || i.key_pressed(egui::Key::C),
                i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Space),
            )
        });
        let range_mode = self
            .session
            .as_ref()
            .is_some_and(|session| session.task_mode == QuizTaskMode::Range);
        if fold && !range_mode {
            self.answer(false);
        }
        if action && !range_mode {
            self.answer(true);
        }
        if next {
            if self.session.as_ref().is_some_and(|s| s.revealed) {
                self.advance(ctx);
            } else if range_mode {
                self.answer_range();
            }
        }
    }

    fn answer(&mut self, took_action: bool) {
        let Some(session) = self.session.as_mut() else {
            return;
        };
        if session.task_mode != QuizTaskMode::Hand
            || session.computing
            || session.revealed
            || session.solution.is_none()
        {
            return;
        }
        let Some(solution) = session.solution.as_ref() else {
            return;
        };
        let verdict = grade_answer(solution.freq, solution.ev, took_action);
        session.choices[session.index] = Some(took_action);
        session.verdicts[session.index] = Some(verdict);
        session.revealed = true;
    }

    fn answer_range(&mut self) {
        let Some(session) = self.session.as_mut() else {
            return;
        };
        if session.task_mode != QuizTaskMode::Range
            || session.computing
            || session.revealed
            || session.solution.is_none()
        {
            return;
        }
        let Some(solution) = session.solution.as_ref() else {
            return;
        };
        let (verdict, accuracy) = grade_range(&session.user_range, &solution.range);
        session.verdicts[session.index] = Some(verdict);
        session.range_scores[session.index] = Some(accuracy);
        session.revealed = true;
    }

    fn advance(&mut self, ctx: &Context) {
        let Some(session) = self.session.as_mut() else {
            return;
        };
        if !session.revealed {
            return;
        }
        if session.index + 1 >= session.spots.len() {
            self.view = QuizView::Results;
            return;
        }
        session.index += 1;
        session.solution = None;
        session.revealed = false;
        session.error = None;
        session.user_range = [0.0; 169];
        session.slider_pct = 0.0;
        session.ranking.clear();
        start_solve(session, ctx);
    }

    fn skip_question(&mut self, ctx: &Context) {
        let Some(session) = self.session.as_mut() else {
            return;
        };
        if session.index + 1 >= session.spots.len() {
            self.view = QuizView::Results;
            return;
        }
        session.index += 1;
        session.solution = None;
        session.revealed = false;
        session.error = None;
        session.user_range = [0.0; 169];
        session.slider_pct = 0.0;
        session.ranking.clear();
        start_solve(session, ctx);
    }

    fn draw_lobby(&mut self, ctx: &Context) {
        let categories = self.library.categories();
        egui::SidePanel::left("quiz_categories")
            .resizable(true)
            .default_width(240.0)
            .width_range(180.0..=360.0)
            .show(ctx, |ui| {
                ui.heading("Категории");
                ui.add_space(4.0);
                egui::ScrollArea::vertical()
                    .id_salt("quiz_category_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for category in &categories {
                            let count = self.library.cases_in(category).len();
                            let selected = self.selected_category == *category;
                            if ui
                                .selectable_label(selected, format!("{category} ({count})"))
                                .clicked()
                            {
                                self.selected_category = category.clone();
                                self.selected_case = None;
                            }
                        }
                    });
                ui.separator();
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.new_category)
                            .hint_text("Новая категория")
                            .desired_width(140.0),
                    );
                    if ui.button("+").clicked() {
                        let name = self.new_category.trim().to_string();
                        if !name.is_empty() {
                            self.selected_category = name;
                            self.new_category.clear();
                        }
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading(&self.selected_category);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Квиз по категории").clicked() {
                        self.setup.source = QuizSourceKind::Saved;
                        self.setup.category = self.selected_category.clone();
                        self.setup.question_count =
                            self.library.cases_in(&self.selected_category).len().max(1);
                        self.open_setup();
                    }
                });
            });
            ui.label(
                RichText::new("Сохраняйте проблемные раздачи из импорта и разбирайте их здесь.")
                    .small()
                    .weak(),
            );
            ui.add_space(8.0);

            let cases: Vec<QuizCase> = self
                .library
                .cases_in(&self.selected_category)
                .into_iter()
                .cloned()
                .collect();
            if cases.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(40.0);
                    ui.label("В этой категории пока нет раздач.");
                    ui.label(
                        "Откройте импорт турнира → контекстное меню раздачи → «Сохранить в квиз».",
                    );
                    if ui.button("Случайный квиз").clicked() {
                        self.setup.source = QuizSourceKind::Generated;
                        self.open_setup();
                    }
                });
                return;
            }

            let mut start_one = None;
            let mut delete = None;
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let cols = [72.0, 52.0, 64.0, 72.0, 220.0];
                    ui.horizontal(|ui| {
                        header_cell(ui, cols[0], "Cards");
                        header_cell(ui, cols[1], "Pos");
                        header_cell(ui, cols[2], "Players");
                        header_cell(ui, cols[3], "Stack");
                        header_cell(ui, cols[4], "Source");
                    });
                    for case in &cases {
                        let selected = self.selected_case.as_deref() == Some(case.id.as_str());
                        let stack = hero_stack_bb(case)
                            .map(|bb| format!("{bb:.1}"))
                            .unwrap_or_else(|| "—".into());
                        let players = case.player_count.to_string();
                        let values = [
                            case.hero_combo_label.as_str(),
                            case.hero_label.as_str(),
                            players.as_str(),
                            stack.as_str(),
                            case.source.as_str(),
                        ];
                        let response = draw_case_row(ui, &cols, &values, selected);
                        if response.clicked() {
                            self.selected_case = Some(case.id.clone());
                        }
                        if response.double_clicked() {
                            start_one = Some(case.clone());
                        }
                        response.context_menu(|ui| {
                            if ui.button("Начать с этой раздачи").clicked() {
                                start_one = Some(case.clone());
                                ui.close_menu();
                            }
                            if ui.button("Удалить").clicked() {
                                delete = Some(case.id.clone());
                                ui.close_menu();
                            }
                        });
                    }
                });
            if let Some(case) = start_one {
                self.start_single(case, ctx);
            }
            if let Some(id) = delete {
                self.library.remove(&id);
                let _ = self.library.save();
                if self.selected_case.as_deref() == Some(id.as_str()) {
                    self.selected_case = None;
                }
            }
        });
    }

    fn start_single(&mut self, case: QuizCase, ctx: &Context) {
        let mut session = QuizSession {
            verdicts: vec![None],
            choices: vec![None],
            range_scores: vec![None],
            spots: vec![case],
            index: 0,
            worker_rx: None,
            computing: false,
            solution: None,
            revealed: false,
            error: None,
            matrix_mode: MatrixMode::Frequency,
            task_mode: self.setup.task_mode,
            user_range: [0.0; 169],
            slider_pct: 0.0,
            ranking: Vec::new(),
        };
        start_solve(&mut session, ctx);
        self.session = Some(session);
        self.view = QuizView::Playing;
    }

    fn draw_play(&mut self, ctx: &Context) {
        let PlayView {
            computing,
            revealed,
            can_answer,
            action_name,
            error,
            solution,
            combo,
            hero_index,
            cards,
            combo_label_text,
            actions,
            stacks,
            n,
            sb,
            bb,
            hero_label,
            title,
            task_mode,
        } = match PlayView::from_session(self.session.as_ref()) {
            Some(view) => view,
            None => return,
        };
        let show_cards = task_mode == QuizTaskMode::Hand;

        egui::SidePanel::left("quiz_hero_range")
            .resizable(true)
            .default_width(340.0)
            .width_range(280.0..=440.0)
            .show(ctx, |ui| {
                ui.heading(if revealed {
                    "Nash"
                } else if task_mode == QuizTaskMode::Range {
                    "Ваш диапазон"
                } else {
                    "Диапазон"
                });
                if computing {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Считаем Nash…");
                    });
                }
                egui::ScrollArea::vertical()
                    .id_salt("quiz_left_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if task_mode == QuizTaskMode::Range {
                            self.draw_user_range_editor(ui, revealed);
                        } else if revealed {
                            if let Some(solution) = solution.as_ref() {
                                self.draw_reveal(ui, solution, combo);
                            }
                        } else {
                            ui.label(
                                RichText::new("После ответа здесь будет Nash-диапазон")
                                    .small()
                                    .weak(),
                            );
                        }
                    });
            });

        egui::SidePanel::right("quiz_ranges_panel")
            .resizable(true)
            .default_width(240.0)
            .width_range(180.0..=360.0)
            .show(ctx, |ui| {
                ui.heading("Оппоненты");
                if let Some(error) = error.as_ref() {
                    ui.colored_label(Color32::RED, error);
                    if ui.button("Пропустить").clicked() {
                        self.skip_question(ctx);
                    }
                }
                egui::ScrollArea::vertical()
                    .id_salt("quiz_right_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if let Some(solution) = solution.as_ref() {
                            self.draw_participant_ranges(ui, solution);
                        }
                    });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label(RichText::new(title).heading());
            ui.label(action_summary(&actions, &hero_label));
            ui.add_space(6.0);
            draw_poker_table(
                ui,
                n,
                &stacks,
                sb,
                bb,
                hero_index,
                &hero_label,
                &actions,
                if show_cards { cards.as_ref() } else { None },
                if show_cards { &combo_label_text } else { "" },
            );
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if task_mode == QuizTaskMode::Hand {
                    ui.add_enabled_ui(can_answer, |ui| {
                        if ui
                            .add_sized(egui::vec2(110.0, 36.0), egui::Button::new("Fold"))
                            .clicked()
                        {
                            self.answer(false);
                        }
                        if ui
                            .add_sized(egui::vec2(110.0, 36.0), egui::Button::new(action_name))
                            .clicked()
                        {
                            self.answer(true);
                        }
                    });
                } else {
                    ui.add_enabled_ui(can_answer, |ui| {
                        if ui
                            .add_sized(egui::vec2(140.0, 36.0), egui::Button::new("Проверить"))
                            .clicked()
                        {
                            self.answer_range();
                        }
                    });
                }
                if revealed
                    && ui
                        .add_sized(egui::vec2(110.0, 36.0), egui::Button::new("Дальше"))
                        .clicked()
                {
                    self.advance(ctx);
                }
            });
        });
    }

    fn draw_participant_ranges(&mut self, ui: &mut Ui, solution: &SolvedSpot) {
        if !solution.villains.is_empty() {
            ui.label(RichText::new("Уже в банке").strong());
            for villain in &solution.villains {
                draw_named_range(ui, villain);
            }
        }
        if !solution.callers.is_empty() {
            ui.label(RichText::new("Колл vs hero").strong());
            for caller in &solution.callers {
                draw_named_range(ui, caller);
            }
        }
        if solution.villains.is_empty() && solution.callers.is_empty() {
            ui.label(
                RichText::new("Нет диапазонов соперников на этом споте.")
                    .small()
                    .weak(),
            );
        } else {
            ui.label(
                RichText::new("Наведите на процент, чтобы увидеть матрицу")
                    .small()
                    .weak(),
            );
        }
    }

    fn draw_user_range_editor(&mut self, ui: &mut Ui, revealed: bool) {
        let (mut user_range, is_raise, mut mode, mut slider_pct, ranking) = {
            let Some(session) = self.session.as_ref() else {
                return;
            };
            (
                session.user_range,
                session
                    .solution
                    .as_ref()
                    .map(|s| s.is_raise)
                    .unwrap_or(true),
                session.matrix_mode,
                session.slider_pct,
                session.ranking.clone(),
            )
        };
        if revealed {
            if let Some(solution) = self.session.as_ref().and_then(|s| s.solution.clone()) {
                self.draw_reveal(ui, &solution, 0);
            }
        } else {
            draw_square_matrix(
                ui,
                &mut user_range,
                None,
                &mut mode,
                "Выберите диапазон",
                is_raise,
                false,
                None,
            );
        }
        if !revealed && !ranking.is_empty() {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label("Range");
                ui.spacing_mut().interact_size.y = 36.0;
                ui.spacing_mut().slider_rail_height = 16.0;
                let drag_w = 72.0;
                let slider_w = (ui.available_width() - drag_w - 12.0).max(80.0);
                ui.spacing_mut().slider_width = slider_w;
                let mut pct = slider_pct;
                let slider = ui.add_sized(
                    [slider_w, 36.0],
                    egui::Slider::new(&mut pct, 0.0..=100.0)
                        .show_value(false)
                        .clamping(egui::SliderClamping::Edits)
                        .handle_shape(egui::style::HandleShape::Rect { aspect_ratio: 0.45 }),
                );
                let mut typed = slider_pct;
                let drag = ui.add_sized(
                    [drag_w, 36.0],
                    egui::DragValue::new(&mut typed)
                        .range(0.0..=100.0)
                        .suffix("%")
                        .max_decimals(1)
                        .speed(0.1)
                        .update_while_editing(true)
                        .clamp_existing_to_range(false),
                );
                if slider.dragged() || slider.drag_started() {
                    slider_pct = pct.clamp(0.0, 100.0);
                    user_range = fill_range_to_share(&ranking, slider_pct / 100.0);
                } else if drag.changed() && (drag.has_focus() || drag.dragged()) {
                    slider_pct = typed.clamp(0.0, 100.0);
                    user_range = fill_range_to_share(&ranking, slider_pct / 100.0);
                }
            });
        }
        if let Some(session) = self.session.as_mut() {
            session.user_range = user_range;
            session.matrix_mode = mode;
            session.slider_pct = slider_pct;
        }
    }

    fn draw_reveal(&mut self, ui: &mut Ui, solution: &SolvedSpot, combo: usize) {
        let (verdict, took_action, range_score, task_mode) = {
            let Some(session) = self.session.as_ref() else {
                return;
            };
            (
                session.verdicts[session.index],
                session.choices[session.index],
                session.range_scores[session.index],
                session.task_mode,
            )
        };
        ui.horizontal(|ui| {
            match verdict {
                Some(QuizVerdict::Correct) => {
                    ui.colored_label(GREEN, RichText::new("Верно").strong())
                }
                Some(QuizVerdict::Mixed) => {
                    ui.colored_label(MIXED, RichText::new("Смешанно").strong())
                }
                Some(QuizVerdict::Wrong) => ui.colored_label(RED, RichText::new("Ошибка").strong()),
                None => ui.label(""),
            };
            if task_mode == QuizTaskMode::Range {
                ui.label(format!(
                    "Совпадение: {:.0}%",
                    range_score.unwrap_or(0.0) * 100.0
                ));
            } else {
                let user = match took_action {
                    Some(true) if solution.is_raise => "Push",
                    Some(true) => "Call",
                    Some(false) => "Fold",
                    None => "—",
                };
                ui.label(format!(
                    "Вы: {user}  ·  Nash: {:.0}% {}  ·  EV {:+.2}",
                    solution.freq * 100.0,
                    if solution.is_raise { "push" } else { "call" },
                    solution.ev.unwrap_or(0.0)
                ));
            }
        });
        let share = combo_share(&solution.range) * 100.0;
        let notation = range_notation(&solution.range);
        let summary = if notation.is_empty() {
            format!("{}  {share:.1}%", display_vs(&solution.title))
        } else {
            format!("{}  {share:.1}%  {notation}", display_vs(&solution.title))
        };
        ui.label(RichText::new(summary).small());
        ui.add_space(6.0);
        let mut range = solution.range;
        let mut mode = MatrixMode::Frequency;
        draw_square_matrix(
            ui,
            &mut range,
            None,
            &mut mode,
            "",
            solution.is_raise,
            false,
            if task_mode == QuizTaskMode::Hand {
                Some(combo)
            } else {
                None
            },
        );
    }

    fn draw_results(&mut self, ctx: &Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(session) = self.session.as_ref() else {
                return;
            };
            ui.heading("Результат");
            ui.add_space(6.0);
            ui.label(score_label(session));
            ui.add_space(12.0);
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for (index, spot) in session.spots.iter().enumerate() {
                        let verdict = session.verdicts.get(index).copied().flatten();
                        let choice = session.choices.get(index).copied().flatten();
                        let mark = match verdict {
                            Some(QuizVerdict::Correct) => ("Верно", GREEN),
                            Some(QuizVerdict::Mixed) => ("Смешанно", MIXED),
                            Some(QuizVerdict::Wrong) => ("Ошибка", RED),
                            None => ("Пропуск", Color32::GRAY),
                        };
                        let user = if let Some(score) =
                            session.range_scores.get(index).copied().flatten()
                        {
                            format!("{:.0}%", score * 100.0)
                        } else {
                            match choice {
                                Some(true) => "Push/Call".to_string(),
                                Some(false) => "Fold".to_string(),
                                None => "—".to_string(),
                            }
                        };
                        ui.horizontal(|ui| {
                            ui.colored_label(mark.1, format!("{:>2}. {}", index + 1, mark.0));
                            ui.label(format!(
                                "{} {}  {}bb  {user}",
                                display_player(&spot.hero_label),
                                spot.hero_combo_label,
                                hero_stack_bb(spot)
                                    .map(|bb| format!("{bb:.1}"))
                                    .unwrap_or_else(|| "—".into())
                            ));
                            if !spot.source.is_empty() && spot.source != "generated" {
                                ui.label(RichText::new(&spot.source).small().weak());
                            }
                        });
                    }
                });
        });
    }

    fn draw_setup_modal(&mut self, ctx: &Context) {
        if !self.setup_open {
            return;
        }
        let screen = ctx.screen_rect();
        egui::Area::new(egui::Id::new("quiz_modal_dim"))
            .order(egui::Order::Middle)
            .fixed_pos(screen.min)
            .interactable(true)
            .show(ctx, |ui| {
                let (rect, _) = ui.allocate_exact_size(screen.size(), Sense::click());
                ui.painter()
                    .rect_filled(rect, 0.0, Color32::from_black_alpha(90));
            });

        let mut open = true;
        let mut start = false;
        let mut cancel = false;
        let categories = self.library.categories();
        egui::Window::new("Настройки квиза")
            .id(egui::Id::new("quiz_setup_window"))
            .order(egui::Order::Foreground)
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .default_size(egui::vec2(520.0, 470.0))
            .show(ctx, |ui| {
                ui.set_min_width(500.0);
                ui.spacing_mut().item_spacing = egui::vec2(8.0, 6.0);

                ui.horizontal(|ui| {
                    ui.label("Источник:");
                    if ui
                        .selectable_label(
                            self.setup.source == QuizSourceKind::Generated,
                            "Случайные споты",
                        )
                        .clicked()
                    {
                        self.setup.source = QuizSourceKind::Generated;
                    }
                    if ui
                        .selectable_label(
                            self.setup.source == QuizSourceKind::Saved,
                            "Сохранённые кейсы",
                        )
                        .clicked()
                    {
                        self.setup.source = QuizSourceKind::Saved;
                    }
                });

                if self.setup.source == QuizSourceKind::Saved {
                    ui.horizontal(|ui| {
                        ui.label("Категория:");
                        egui::ComboBox::from_id_salt("quiz_setup_category")
                            .selected_text(&self.setup.category)
                            .show_ui(ui, |ui| {
                                for category in &categories {
                                    ui.selectable_value(
                                        &mut self.setup.category,
                                        category.clone(),
                                        category,
                                    );
                                }
                            });
                    });
                } else {
                    self.draw_generated_setup(ui);
                }

                ui.horizontal(|ui| {
                    ui.label("Режим:");
                    if ui
                        .selectable_label(self.setup.task_mode == QuizTaskMode::Hand, "Рука")
                        .on_hover_text("Конкретные карты, Fold или Push/Call")
                        .clicked()
                    {
                        self.setup.task_mode = QuizTaskMode::Hand;
                    }
                    if ui
                        .selectable_label(self.setup.task_mode == QuizTaskMode::Range, "Диапазон")
                        .on_hover_text("Нарисовать диапазон действия и сравнить с Nash")
                        .clicked()
                    {
                        self.setup.task_mode = QuizTaskMode::Range;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Вопросов:");
                    ui.add(
                        egui::DragValue::new(&mut self.setup.question_count)
                            .range(1..=100)
                            .speed(1.0),
                    );
                });

                ui.add_space(8.0);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_sized(egui::vec2(88.0, 26.0), egui::Button::new("Старт"))
                        .clicked()
                    {
                        start = true;
                    }
                    if ui
                        .add_sized(egui::vec2(88.0, 26.0), egui::Button::new("Отмена"))
                        .clicked()
                    {
                        cancel = true;
                    }
                });
            });

        if start {
            self.start_quiz(ctx);
        } else if !open || cancel {
            self.setup_open = false;
        }
    }

    fn draw_generated_setup(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Игроки:");
            if ui
                .selectable_label(self.setup.players_exact, "Точное")
                .clicked()
            {
                self.setup.players_exact = true;
                sync_seat_stacks(&mut self.setup);
            }
            if ui
                .selectable_label(!self.setup.players_exact, "Диапазон")
                .clicked()
            {
                self.setup.players_exact = false;
                self.setup.per_seat = false;
            }
            if self.setup.players_exact {
                if ui
                    .add(
                        egui::DragValue::new(&mut self.setup.players)
                            .range(2..=9)
                            .speed(1),
                    )
                    .changed()
                {
                    sync_seat_stacks(&mut self.setup);
                    self.setup.prize_percents =
                        crate::quiz::default_prize_strings(self.setup.players);
                }
            } else {
                ui.add(
                    egui::DragValue::new(&mut self.setup.players_min)
                        .range(2..=9)
                        .speed(1),
                );
                ui.label("–");
                ui.add(
                    egui::DragValue::new(&mut self.setup.players_max)
                        .range(2..=9)
                        .speed(1),
                );
            }
        });

        ui.horizontal(|ui| {
            ui.label("Стеки:");
            if ui
                .selectable_label(self.setup.stacks_exact, "Точные")
                .clicked()
            {
                self.setup.stacks_exact = true;
            }
            if ui
                .selectable_label(!self.setup.stacks_exact, "Диапазон")
                .clicked()
            {
                self.setup.stacks_exact = false;
                self.setup.per_seat = false;
            }
            if self.setup.stacks_exact {
                ui.add(
                    egui::DragValue::new(&mut self.setup.stack_bb)
                        .range(1.0..=40.0)
                        .suffix(" bb")
                        .speed(0.5),
                );
                if self.setup.players_exact
                    && ui
                        .checkbox(&mut self.setup.per_seat, "по позициям")
                        .changed()
                    && self.setup.per_seat
                {
                    sync_seat_stacks(&mut self.setup);
                }
            } else {
                ui.add(
                    egui::DragValue::new(&mut self.setup.stack_min_bb)
                        .range(1.0..=40.0)
                        .suffix(" bb")
                        .speed(0.5),
                );
                ui.label("–");
                ui.add(
                    egui::DragValue::new(&mut self.setup.stack_max_bb)
                        .range(1.0..=40.0)
                        .suffix(" bb")
                        .speed(0.5),
                );
            }
        });

        if self.setup.stacks_exact && self.setup.per_seat && self.setup.players_exact {
            egui::Grid::new("quiz_seat_stacks")
                .num_columns(2)
                .spacing(egui::vec2(8.0, 4.0))
                .show(ui, |ui| {
                    for index in 0..self.setup.players {
                        ui.label(display_player(table_position_label(
                            self.setup.players,
                            index,
                        )));
                        ui.add(
                            egui::DragValue::new(&mut self.setup.seat_stacks_bb[index])
                                .range(1.0..=40.0)
                                .suffix(" bb")
                                .speed(0.5),
                        );
                        ui.end_row();
                    }
                });
        }

        ui.horizontal_wrapped(|ui| {
            ui.label("Призы %:");
            for prize in &mut self.setup.prize_percents {
                ui.add(
                    egui::TextEdit::singleline(prize)
                        .desired_width(42.0)
                        .hint_text("нет"),
                );
            }
        });
        ui.checkbox(
            &mut self.setup.include_facing_shove,
            "включая споты против пуша",
        );
    }
}

fn start_solve(session: &mut QuizSession, ctx: &Context) {
    let Some(spot) = session.spots.get(session.index).cloned() else {
        return;
    };
    let input = match solver_input_for(&spot) {
        Ok(input) => input,
        Err(message) => {
            session.error = Some(message);
            session.computing = false;
            return;
        }
    };
    let cache = match EquityCache::from_bytes(CACHE_BYTES) {
        Ok(cache) => cache,
        Err(_) => {
            session.error = Some("не удалось загрузить equity cache".into());
            return;
        }
    };
    let (tx, rx) = mpsc::channel();
    session.worker_rx = Some(rx);
    session.computing = true;
    session.solution = None;
    ctx.request_repaint();
    thread::spawn(move || {
        let output = solve(&input, &cache);
        match extract_spot(&output, &input, &spot) {
            Some(solution) => {
                let _ = tx.send(SolveMessage::Done(solution));
            }
            None => {
                let _ = tx.send(SolveMessage::Failed(
                    "не удалось найти спот героя в дереве".into(),
                ));
            }
        }
    });
}

fn extract_spot(
    output: &SolverOutput,
    input: &poker_core::SolverInput,
    case: &QuizCase,
) -> Option<SolvedSpot> {
    let tree = build_strategy_tree(output, input.button_index, &input.stacks, input.big_blind);
    let path = find_quiz_hero_path(&tree, &case.actions_before, &case.hero_label)?;
    let node = node_at_path(&tree, &path)?;
    let combo = case.hero_combo as usize;
    if combo >= 169 {
        return None;
    }
    Some(SolvedSpot {
        range: node.range,
        evs: node.ev_range,
        title: node.label.clone(),
        is_raise: node.action.is_raise(),
        freq: node.range[combo].clamp(0.0, 1.0),
        ev: node.ev_range.map(|evs| evs[combo]),
        villains: villains_on_path(&tree, &path),
        callers: callers_of(node),
    })
}

fn find_quiz_hero_path(
    tree: &[TreeNode],
    actions_before: &[(String, bool)],
    hero_label: &str,
) -> Option<Vec<usize>> {
    find_hero_decision_path(tree, actions_before, hero_label)
        .or_else(|| scan_hero_path(tree, actions_before, hero_label))
}

fn scan_hero_path(
    tree: &[TreeNode],
    actions_before: &[(String, bool)],
    hero: &str,
) -> Option<Vec<usize>> {
    let shovers: Vec<&str> = actions_before
        .iter()
        .filter(|(_, is_fold)| !*is_fold)
        .map(|(pos, _)| pos.as_str())
        .collect();
    let folded: Vec<&str> = actions_before
        .iter()
        .filter(|(_, is_fold)| *is_fold)
        .map(|(pos, _)| pos.as_str())
        .collect();
    scan_hero_nodes(tree, Vec::new(), &[], &shovers, &folded, hero)
}

fn scan_hero_nodes(
    nodes: &[TreeNode],
    path: Vec<usize>,
    ancestors: &[&str],
    shovers: &[&str],
    folded: &[&str],
    hero: &str,
) -> Option<Vec<usize>> {
    for (index, node) in nodes.iter().enumerate() {
        let mut next_path = path.clone();
        next_path.push(index);
        if same_player(node.player(), hero) {
            let shoves_ok = shovers
                .iter()
                .all(|shove| ancestors.iter().any(|name| same_player(name, shove)));
            let folds_ok = folded
                .iter()
                .all(|name| !ancestors.iter().any(|prev| same_player(prev, name)));
            if shoves_ok && folds_ok {
                return Some(next_path);
            }
        }
        let mut next_ancestors = ancestors.to_vec();
        next_ancestors.push(node.player());
        if let Some(found) = scan_hero_nodes(
            &node.children,
            next_path,
            &next_ancestors,
            shovers,
            folded,
            hero,
        ) {
            return Some(found);
        }
    }
    None
}

fn callers_of(node: &TreeNode) -> Vec<VillainRange> {
    node.children
        .iter()
        .filter(|child| child.action.is_call())
        .map(|child| VillainRange {
            label: child.label.clone(),
            range: child.range,
        })
        .collect()
}

fn villains_on_path(tree: &[TreeNode], path: &[usize]) -> Vec<VillainRange> {
    let mut villains = Vec::new();
    if path.len() <= 1 {
        return villains;
    }
    let mut nodes = tree;
    for (depth, &index) in path.iter().enumerate() {
        let Some(node) = nodes.get(index) else {
            break;
        };
        if depth + 1 < path.len() {
            villains.push(VillainRange {
                label: node.label.clone(),
                range: node.range,
            });
            nodes = node.children.as_slice();
        }
    }
    villains
}

fn draw_named_range(ui: &mut Ui, villain: &VillainRange) {
    let share = combo_share(&villain.range) * 100.0;
    let title = display_vs(&villain.label);
    ui.add_space(4.0);
    ui.add(
        egui::Label::new(RichText::new(format!("{title}  {share:.1}%")).strong())
            .sense(Sense::hover()),
    )
    .on_hover_cursor(egui::CursorIcon::Help)
    .on_hover_ui_at_pointer(|ui| {
        draw_hover_matrix(ui, &villain.range, &title);
    });
}

fn draw_hover_matrix(ui: &mut Ui, range: &[f64; 169], title: &str) {
    let mut range = *range;
    let mut mode = MatrixMode::Frequency;
    let notation = range_notation(&range);
    ui.set_width(268.0);
    ui.label(RichText::new(title).strong());
    if !notation.is_empty() {
        ui.label(RichText::new(notation).small());
    }
    ui.allocate_ui(egui::vec2(268.0, 292.0), |ui| {
        range_matrix_ui(
            ui, &mut range, None, &mut mode, "", false, false, false, None,
        );
    });
}

fn draw_square_matrix(
    ui: &mut Ui,
    range: &mut [f64; 169],
    hand_evs: Option<&[f64; 169]>,
    mode: &mut MatrixMode,
    title: &str,
    is_raise: bool,
    show_mode_toggle: bool,
    highlight_combo: Option<usize>,
) {
    let width = ui.available_width().max(182.0);
    let extra = if show_mode_toggle { 28.0 } else { 0.0 };
    let header = if title.is_empty() { 0.0 } else { 26.0 };
    ui.allocate_ui(egui::vec2(width, width + header + extra + 28.0), |ui| {
        range_matrix_ui(
            ui,
            range,
            hand_evs,
            mode,
            title,
            false,
            is_raise,
            show_mode_toggle,
            highlight_combo,
        );
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SeatAction {
    Waiting,
    ToAct,
    Fold,
    Shove,
    Call,
}

fn draw_poker_table(
    ui: &mut Ui,
    n: usize,
    stacks: &[f64],
    sb: f64,
    bb: f64,
    hero_index: usize,
    hero_label: &str,
    actions: &[(String, bool)],
    hero_cards: Option<&[String; 2]>,
    combo_label: &str,
) {
    let width = ui.available_width().min(640.0);
    let height = (ui.available_height() - 56.0).clamp(280.0, 460.0);
    ui.allocate_ui_with_layout(
        egui::vec2(ui.available_width(), height),
        egui::Layout::top_down(egui::Align::Center),
        |ui| {
            draw_poker_table_inner(
                ui,
                n,
                stacks,
                sb,
                bb,
                hero_index,
                hero_label,
                actions,
                hero_cards,
                combo_label,
                width,
                height,
            );
        },
    );
}

fn draw_poker_table_inner(
    ui: &mut Ui,
    n: usize,
    stacks: &[f64],
    sb: f64,
    bb: f64,
    hero_index: usize,
    hero_label: &str,
    actions: &[(String, bool)],
    hero_cards: Option<&[String; 2]>,
    combo_label: &str,
    width: f32,
    height: f32,
) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), Sense::hover());
    let painter = ui.painter_at(rect);
    let center = rect.center();
    let pad_x = 52.0_f32;
    let pad_y = 40.0_f32;
    let rx = ((rect.width() * 0.5 - pad_x) / 1.38).clamp(96.0, 168.0);
    let ry = ((rect.height() * 0.5 - pad_y) / 1.38).clamp(64.0, 110.0);
    let table = egui::Rect::from_center_size(center, egui::vec2(rx * 2.0, ry * 2.0));
    painter.rect_filled(table, ry, FELT);
    painter.rect_stroke(table, ry, Stroke::new(5.0_f32, FELT_EDGE));

    let statuses = seat_actions(n, hero_index, actions);
    let dealer = dealer_seat(n);
    let mut seats = Vec::with_capacity(n);

    for index in 0..n {
        let rel = (index + n - hero_index) % n;
        let angle = FRAC_PI_2 + rel as f32 * TAU / n as f32;
        let pos = egui::pos2(
            (center.x + rx * 1.38 * angle.cos()).clamp(rect.left() + pad_x, rect.right() - pad_x),
            (center.y + ry * 1.38 * angle.sin()).clamp(rect.top() + pad_y, rect.bottom() - pad_y),
        );
        seats.push((index, pos, angle));
    }

    for &(index, pos, _) in &seats {
        let status = statuses.get(index).copied().unwrap_or(SeatAction::Waiting);
        let chip_pos = felt_in_front(center, rx, ry, pos, 0.86);
        match status {
            SeatAction::Shove => {
                let amount = stacks.get(index).copied().unwrap_or(0.0) / bb.max(1e-9);
                draw_chip_pile(&painter, chip_pos, 5, CHIP_SHOVE, &format!("{amount:.1}"));
            }
            SeatAction::Call => {
                let amount = stacks.get(index).copied().unwrap_or(0.0) / bb.max(1e-9);
                draw_chip_pile(&painter, chip_pos, 4, CHIP_CALL, &format!("{amount:.1}"));
            }
            _ if index == sb_index(n) => {
                let label = if (sb / bb.max(1e-9) - 0.5).abs() < 0.05 {
                    "SB".to_string()
                } else {
                    format!("{:.1}", sb / bb.max(1e-9))
                };
                draw_chip_pile(&painter, chip_pos, 1, CHIP_SB, &label);
            }
            _ if index == bb_index(n) => {
                draw_chip_pile(&painter, chip_pos, 2, CHIP_BB, "BB");
            }
            _ => {}
        }
        if index == dealer {
            let dealer_pos =
                felt_in_front(center, rx, ry, pos, 0.90) + tangent_of(toward(pos, center)) * 18.0;
            draw_dealer_button(&painter, dealer_pos);
        }
    }

    for &(index, pos, _) in &seats {
        let label = display_player(table_position_label(n, index));
        let stack_bb = stacks.get(index).copied().unwrap_or(0.0) / bb.max(1e-9);
        let is_hero = index == hero_index || label == display_player(hero_label);
        let status = statuses.get(index).copied().unwrap_or(SeatAction::Waiting);
        let (fill, stroke, caption) = seat_style(is_hero, status);
        let bubble = egui::Rect::from_center_size(pos, egui::vec2(88.0, 44.0));
        painter.rect_filled(bubble, 8.0, fill);
        painter.rect_stroke(bubble, 8.0, Stroke::new(1.5_f32, stroke));
        let name_color = if status == SeatAction::Fold {
            TEXT_FAINT
        } else {
            TEXT
        };
        painter.text(
            bubble.center() - egui::vec2(0.0, 10.0),
            Align2::CENTER_CENTER,
            label,
            FontId::proportional(13.0),
            name_color,
        );
        painter.text(
            bubble.center() + egui::vec2(0.0, 8.0),
            Align2::CENTER_CENTER,
            format!("{stack_bb:.1} bb"),
            FontId::proportional(11.0),
            name_color,
        );
        if let Some(badge) = caption {
            painter.text(
                bubble.center() + egui::vec2(0.0, 28.0),
                Align2::CENTER_CENTER,
                badge,
                FontId::proportional(11.0),
                stroke,
            );
        }
        if is_hero {
            if let Some(cards) = hero_cards {
                let inward = toward(pos, center);
                let card_anchor = felt_in_front(center, rx, ry, pos, 0.62);
                paint_hole_card(
                    &painter,
                    egui::Rect::from_center_size(
                        card_anchor - egui::vec2(24.0, 0.0),
                        egui::vec2(44.0, 60.0),
                    ),
                    &cards[0],
                );
                paint_hole_card(
                    &painter,
                    egui::Rect::from_center_size(
                        card_anchor + egui::vec2(24.0, 0.0),
                        egui::vec2(44.0, 60.0),
                    ),
                    &cards[1],
                );
                painter.text(
                    card_anchor + inward * 38.0,
                    Align2::CENTER_CENTER,
                    combo_label,
                    FontId::proportional(11.0),
                    TEXT,
                );
            }
        }
    }
}

fn seat_actions(n: usize, hero_index: usize, actions: &[(String, bool)]) -> Vec<SeatAction> {
    let mut statuses = vec![SeatAction::Waiting; n];
    if hero_index < n {
        statuses[hero_index] = SeatAction::ToAct;
    }
    let mut seen_shove = false;
    for (pos, is_fold) in actions {
        let Some(index) = (0..n).find(|&i| same_player(table_position_label(n, i), pos.as_str()))
        else {
            continue;
        };
        if *is_fold {
            statuses[index] = SeatAction::Fold;
        } else if seen_shove {
            statuses[index] = SeatAction::Call;
            seen_shove = true;
        } else {
            statuses[index] = SeatAction::Shove;
            seen_shove = true;
        }
    }
    statuses
}

fn seat_style(is_hero: bool, status: SeatAction) -> (Color32, Color32, Option<&'static str>) {
    match status {
        SeatAction::Fold => (SEAT_FOLD, Color32::from_gray(120), Some("Fold")),
        SeatAction::Shove => (SEAT_IN, CHIP_SHOVE, Some("All-in")),
        SeatAction::Call => (SEAT_IN, CHIP_CALL, Some("Call")),
        SeatAction::ToAct if is_hero => (Color32::from_rgb(255, 224, 120), FELT_EDGE, Some("Hero")),
        SeatAction::ToAct => (Color32::from_rgb(255, 224, 120), FELT_EDGE, None),
        SeatAction::Waiting if is_hero => {
            (Color32::from_rgb(255, 224, 120), FELT_EDGE, Some("Hero"))
        }
        SeatAction::Waiting => (Color32::WHITE, HEADER, None),
    }
}

fn sb_index(n: usize) -> usize {
    if n <= 2 {
        0
    } else {
        n - 2
    }
}

fn bb_index(n: usize) -> usize {
    n.saturating_sub(1)
}

fn toward(from: egui::Pos2, to: egui::Pos2) -> egui::Vec2 {
    let delta = to - from;
    let len = delta.length().max(1.0);
    delta / len
}

fn tangent_of(inward: egui::Vec2) -> egui::Vec2 {
    egui::vec2(-inward.y, inward.x)
}

fn felt_in_front(
    center: egui::Pos2,
    rx: f32,
    ry: f32,
    seat_pos: egui::Pos2,
    scale: f32,
) -> egui::Pos2 {
    ellipse_point(center, rx, ry, seat_pos, scale.clamp(0.55, 0.94))
}

fn ellipse_point(
    center: egui::Pos2,
    rx: f32,
    ry: f32,
    toward_pos: egui::Pos2,
    scale: f32,
) -> egui::Pos2 {
    let delta = toward_pos - center;
    let angle = delta.y.atan2(delta.x);
    egui::pos2(
        center.x + rx * scale * angle.cos(),
        center.y + ry * scale * angle.sin(),
    )
}

fn draw_dealer_button(painter: &egui::Painter, pos: egui::Pos2) {
    painter.circle_filled(pos, 11.0, Color32::WHITE);
    painter.circle_stroke(
        pos,
        11.0,
        Stroke::new(1.6_f32, Color32::from_rgb(40, 40, 40)),
    );
    painter.text(
        pos,
        Align2::CENTER_CENTER,
        "D",
        FontId::proportional(12.0),
        TEXT,
    );
}

fn draw_chip_pile(
    painter: &egui::Painter,
    pos: egui::Pos2,
    count: i32,
    color: Color32,
    label: &str,
) {
    let count = count.clamp(1, 6);
    for i in 0..count {
        let p = pos - egui::vec2(0.0, i as f32 * 3.4);
        painter.circle_filled(p, 9.0, color);
        painter.circle_stroke(p, 9.0, Stroke::new(1.0_f32, Color32::from_black_alpha(90)));
    }
    let top = pos - egui::vec2(0.0, (count - 1) as f32 * 3.4);
    let bright = color.r() as u16 + color.g() as u16 + color.b() as u16 > 420;
    painter.text(
        top,
        Align2::CENTER_CENTER,
        label,
        FontId::proportional(9.0),
        if bright { TEXT } else { Color32::WHITE },
    );
}

fn paint_hole_card(painter: &egui::Painter, rect: egui::Rect, card: &str) {
    let parsed = Card::from_str(card).ok();
    let (rank, suit) = parsed
        .map(|c| (c.rank_char(), c.suit_char()))
        .unwrap_or(('?', '?'));
    let red = matches!(suit, 'h' | 'd');
    let suit_glyph = match suit {
        's' => "♠",
        'h' => "♥",
        'd' => "♦",
        'c' => "♣",
        _ => "?",
    };
    let color = if red { RED } else { TEXT };
    painter.rect_filled(rect, 6.0, Color32::WHITE);
    painter.rect_stroke(rect, 6.0, Stroke::new(1.0_f32, Color32::from_gray(160)));
    painter.text(
        rect.center() - egui::vec2(0.0, 10.0),
        Align2::CENTER_CENTER,
        rank.to_string(),
        FontId::proportional(16.0),
        color,
    );
    painter.text(
        rect.center() + egui::vec2(0.0, 12.0),
        Align2::CENTER_CENTER,
        suit_glyph,
        FontId::proportional(15.0),
        color,
    );
}

fn action_summary(actions: &[(String, bool)], hero_label: &str) -> String {
    if actions.is_empty() {
        return format!("{} ходит первым", display_player(hero_label));
    }
    let line = actions
        .iter()
        .map(|(pos, is_fold)| {
            format!(
                "{} {}",
                display_player(pos),
                if *is_fold { "fold" } else { "shove" }
            )
        })
        .collect::<Vec<_>>()
        .join(" · ");
    format!("{line} → {}", display_player(hero_label))
}

fn score_label(session: &QuizSession) -> String {
    let mut correct = 0;
    let mut mixed = 0;
    let mut wrong = 0;
    for verdict in session.verdicts.iter().flatten() {
        match verdict {
            QuizVerdict::Correct => correct += 1,
            QuizVerdict::Mixed => mixed += 1,
            QuizVerdict::Wrong => wrong += 1,
        }
    }
    format!("Верно {correct} · смешанно {mixed} · ошибки {wrong}")
}

fn hero_stack_bb(case: &QuizCase) -> Option<f64> {
    let index = hero_seat_index(case.player_count, &case.hero_label)?;
    Some(case.stacks.get(index).copied()? / case.big_blind.max(1e-9))
}

fn sync_seat_stacks(settings: &mut QuizSettings) {
    let n = settings.players.clamp(2, 9);
    settings
        .seat_stacks_bb
        .resize(n, settings.stack_bb.max(1.0));
}

fn header_cell(ui: &mut Ui, width: f32, text: &str) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 22.0), Sense::hover());
    ui.painter().rect_filled(rect, 0.0, HEADER);
    ui.painter().text(
        rect.left_center() + egui::vec2(6.0, 0.0),
        Align2::LEFT_CENTER,
        text,
        FontId::proportional(12.0),
        Color32::WHITE,
    );
}

fn draw_case_row(ui: &mut Ui, cols: &[f32], values: &[&str], selected: bool) -> egui::Response {
    let width = cols.iter().sum::<f32>();
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(width.max(ui.available_width()), 22.0),
        Sense::click(),
    );
    let bg = if selected {
        Color32::from_rgb(196, 226, 248)
    } else if response.hovered() {
        Color32::from_rgb(232, 244, 252)
    } else {
        Color32::WHITE
    };
    ui.painter().rect_filled(rect, 0.0, bg);
    let mut x = rect.left();
    for (index, &col) in cols.iter().enumerate() {
        let cell =
            egui::Rect::from_min_size(egui::pos2(x, rect.top()), egui::vec2(col, rect.height()));
        if let Some(text) = values.get(index) {
            ui.painter().text(
                cell.left_center() + egui::vec2(6.0, 0.0),
                Align2::LEFT_CENTER,
                *text,
                FontId::proportional(12.0),
                TEXT,
            );
        }
        x += col;
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_summary_unopened() {
        let text = action_summary(&[], "BTN");
        assert!(text.contains("первым"));
    }

    #[test]
    fn action_summary_shove() {
        let text = action_summary(&[("UTG".into(), true), ("HJ".into(), false)], "CO");
        assert!(text.contains("fold"));
        assert!(text.contains("shove"));
        assert!(text.contains("CO"));
    }

    #[test]
    fn seat_actions_mark_fold_shove_and_call() {
        let statuses = seat_actions(
            6,
            4,
            &[
                ("UTG".into(), true),
                ("HJ".into(), false),
                ("CO".into(), false),
            ],
        );
        assert_eq!(statuses[0], SeatAction::Fold);
        assert_eq!(statuses[1], SeatAction::Shove);
        assert_eq!(statuses[2], SeatAction::Call);
        assert_eq!(statuses[4], SeatAction::ToAct);
        assert_eq!(statuses[5], SeatAction::Waiting);
    }

    #[test]
    fn blinds_and_dealer_for_hu_and_6max() {
        assert_eq!(sb_index(2), 0);
        assert_eq!(bb_index(2), 1);
        assert_eq!(dealer_seat(2), 0);
        assert_eq!(sb_index(6), 4);
        assert_eq!(bb_index(6), 5);
        assert_eq!(dealer_seat(6), 3);
    }

    fn dummy_node(label: &str, is_call: bool, children: Vec<TreeNode>) -> TreeNode {
        TreeNode {
            range_id: 0,
            label: label.to_string(),
            action: if is_call {
                crate::tabs::strategy_tree::Action::Call
            } else {
                crate::tabs::strategy_tree::Action::Raise
            },
            range_pct: 10.0,
            range: [0.2; 169],
            ev_range: None,
            children,
            expanded: false,
            amount_bb: 10.0,
        }
    }

    #[test]
    fn callers_are_hero_call_children() {
        let hero = dummy_node(
            "BTN push (CO fold)",
            false,
            vec![
                dummy_node("SB call (vs BTN)", true, vec![]),
                dummy_node("BB call (vs BTN)", true, vec![]),
            ],
        );
        let callers = callers_of(&hero);
        assert_eq!(callers.len(), 2);
        assert!(callers[0].label.contains("SB"));
        assert!(callers[1].label.contains("BB"));
    }

    #[test]
    fn finds_btn_after_co_fold_with_bu_label() {
        let tree = vec![
            dummy_node(
                "CO push",
                false,
                vec![dummy_node("BTN call (vs CO)", true, vec![])],
            ),
            dummy_node(
                "BTN push (CO fold)",
                false,
                vec![
                    dummy_node("SB call (vs BTN)", true, vec![]),
                    dummy_node("BB call (vs BTN)", true, vec![]),
                ],
            ),
        ];
        let path = find_quiz_hero_path(&tree, &[("CO".into(), true)], "BU");
        assert_eq!(path, Some(vec![1]));
        let node = node_at_path(&tree, &path.unwrap()).unwrap();
        assert_eq!(callers_of(node).len(), 2);
    }

    #[test]
    fn scan_finds_bb_call_vs_btn_when_co_folded() {
        let tree = vec![
            dummy_node(
                "CO push",
                false,
                vec![dummy_node("BB call (vs CO)", true, vec![])],
            ),
            dummy_node(
                "BTN push (CO fold)",
                false,
                vec![dummy_node("BB call (vs BTN)", true, vec![])],
            ),
        ];
        let path = scan_hero_path(&tree, &[("CO".into(), true), ("BTN".into(), false)], "BB");
        assert_eq!(path, Some(vec![1, 0]));
    }
}
