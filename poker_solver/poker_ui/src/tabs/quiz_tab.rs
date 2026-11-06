use std::f32::consts::{FRAC_PI_2, TAU};
use std::str::FromStr;
use std::sync::mpsc;
use std::thread;

use egui::{Align2, Color32, Context, FontId, RichText, Sense, Stroke, Ui};
use poker_core::{solve, table_position_label, Card, EquityCache, SolverOutput};

use crate::quiz::{
    generate_quiz, grade_answer, hero_seat_index, solver_input_for, QuizCase, QuizLibrary,
    QuizSettings, QuizSourceKind, QuizVerdict, DEFAULT_CATEGORY,
};
use crate::tabs::strategy_tree::{
    build_strategy_tree, display_player, find_hero_decision_path, node_at_path,
};
use crate::widgets::{range_matrix_ui, MatrixMode};

const CACHE_BYTES: &[u8] = include_bytes!("../../assets/equity_cache.bin");
const HEADER: Color32 = Color32::from_rgb(70, 141, 196);
const GREEN: Color32 = Color32::from_rgb(46, 160, 67);
const RED: Color32 = Color32::from_rgb(200, 60, 60);
const MIXED: Color32 = Color32::from_rgb(210, 150, 20);
const FELT: Color32 = Color32::from_rgb(26, 92, 52);
const FELT_EDGE: Color32 = Color32::from_rgb(196, 154, 62);
const TEXT: Color32 = Color32::from_rgb(32, 32, 32);

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
    worker_rx: Option<mpsc::Receiver<SolveMessage>>,
    computing: bool,
    solution: Option<SolvedSpot>,
    revealed: bool,
    error: Option<String>,
    matrix_mode: MatrixMode,
}

#[derive(Clone)]
struct SolvedSpot {
    range: [f64; 169],
    evs: Option<[f64; 169]>,
    title: String,
    is_raise: bool,
    freq: f64,
    ev: Option<f64>,
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
    bb: f64,
    hero_label: String,
    title: String,
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
            bb: spot.big_blind,
            hero_label,
            title,
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
                    spots,
                    index: 0,
                    worker_rx: None,
                    computing: false,
                    solution: None,
                    revealed: false,
                    error: None,
                    matrix_mode: MatrixMode::Frequency,
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
        if fold {
            self.answer(false);
        }
        if action {
            self.answer(true);
        }
        if next {
            self.advance(ctx);
        }
    }

    fn answer(&mut self, took_action: bool) {
        let Some(session) = self.session.as_mut() else {
            return;
        };
        if session.computing || session.revealed || session.solution.is_none() {
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
            spots: vec![case],
            index: 0,
            worker_rx: None,
            computing: false,
            solution: None,
            revealed: false,
            error: None,
            matrix_mode: MatrixMode::Frequency,
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
            bb,
            hero_label,
            title,
        } = match PlayView::from_session(self.session.as_ref()) {
            Some(view) => view,
            None => return,
        };

        {
            let max_outline = (ctx.available_rect().height() * 0.46).max(160.0);
            egui::TopBottomPanel::bottom("quiz_answer_panel")
                .resizable(true)
                .default_height(if revealed { 240.0 } else { 88.0 })
                .min_height(72.0)
                .max_height(max_outline)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.add_enabled_ui(can_answer, |ui| {
                            if ui
                                .add_sized(egui::vec2(96.0, 32.0), egui::Button::new("Fold"))
                                .clicked()
                            {
                                self.answer(false);
                            }
                            if ui
                                .add_sized(egui::vec2(96.0, 32.0), egui::Button::new(action_name))
                                .clicked()
                            {
                                self.answer(true);
                            }
                        });
                        if computing {
                            ui.spinner();
                            ui.label("Считаем Nash…");
                        }
                        if revealed {
                            if ui.button("Дальше").clicked() {
                                self.advance(ctx);
                            }
                        }
                    });
                    if let Some(solution) = solution.as_ref() {
                        if revealed {
                            self.draw_reveal(ui, solution, combo);
                        } else {
                            ui.label(RichText::new(&solution.title).small().weak());
                        }
                    }
                });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(error) = error {
                ui.colored_label(Color32::RED, error);
                if ui.button("Пропустить").clicked() {
                    self.skip_question(ctx);
                }
                return;
            }
            ui.label(RichText::new(title).heading());
            ui.label(action_summary(&actions, &hero_label));
            ui.add_space(8.0);
            draw_poker_table(ui, n, &stacks, bb, hero_index, &hero_label);
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                ui.label(format!(
                    "{} · {}",
                    display_player(&hero_label),
                    combo_label_text
                ));
                if let Some(cards) = cards {
                    draw_hole_card(ui, &cards[0]);
                    draw_hole_card(ui, &cards[1]);
                }
            });
        });
    }

    fn draw_reveal(&mut self, ui: &mut Ui, solution: &SolvedSpot, combo: usize) {
        let (verdict, took_action, mut mode) = {
            let Some(session) = self.session.as_ref() else {
                return;
            };
            (
                session.verdicts[session.index],
                session.choices[session.index],
                session.matrix_mode,
            )
        };
        ui.add_space(6.0);
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
        });
        ui.add_space(6.0);
        let mut range = solution.range;
        let available = ui.available_height().max(140.0);
        ui.allocate_ui(
            egui::vec2(ui.available_width().min(420.0), available),
            |ui| {
                range_matrix_ui(
                    ui,
                    &mut range,
                    solution.evs.as_ref(),
                    &mut mode,
                    &solution.title,
                    false,
                    solution.is_raise,
                    true,
                    Some(combo),
                );
            },
        );
        if let Some(session) = self.session.as_mut() {
            session.matrix_mode = mode;
        }
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
                        let user = match choice {
                            Some(true) => "Push/Call",
                            Some(false) => "Fold",
                            None => "—",
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
            .default_size(egui::vec2(520.0, 430.0))
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
    let path = find_hero_decision_path(&tree, &case.actions_before, &case.hero_label)?;
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
    })
}

fn draw_poker_table(
    ui: &mut Ui,
    n: usize,
    stacks: &[f64],
    bb: f64,
    hero_index: usize,
    hero_label: &str,
) {
    let height = 280.0_f32.min(ui.available_height() - 80.0).max(200.0);
    let width = ui.available_width().min(760.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), Sense::hover());
    let painter = ui.painter_at(rect);
    let center = rect.center();
    let rx = rect.width() * 0.28;
    let ry = rect.height() * 0.30;
    let table = egui::Rect::from_center_size(center, egui::vec2(rx * 2.0, ry * 2.0));
    painter.rect_filled(table, ry, FELT);
    painter.rect_stroke(table, ry, Stroke::new(5.0_f32, FELT_EDGE));
    painter.text(
        center,
        Align2::CENTER_CENTER,
        "QUIZ",
        FontId::proportional(13.0),
        Color32::from_white_alpha(40),
    );

    for index in 0..n {
        let rel = (index + n - hero_index) % n;
        let angle = FRAC_PI_2 + rel as f32 * TAU / n as f32;
        let pos = egui::pos2(
            center.x + rx * 1.35 * angle.cos(),
            center.y + ry * 1.45 * angle.sin(),
        );
        let label = display_player(table_position_label(n, index));
        let stack_bb = stacks.get(index).copied().unwrap_or(0.0) / bb.max(1e-9);
        let is_hero = index == hero_index || label == display_player(hero_label);
        let bubble = egui::Rect::from_center_size(pos, egui::vec2(92.0, 46.0));
        let fill = if is_hero {
            Color32::from_rgb(255, 224, 120)
        } else {
            Color32::WHITE
        };
        painter.rect_filled(bubble, 8.0, fill);
        painter.rect_stroke(
            bubble,
            8.0,
            Stroke::new(1.0_f32, if is_hero { FELT_EDGE } else { HEADER }),
        );
        painter.text(
            bubble.center() - egui::vec2(0.0, 8.0),
            Align2::CENTER_CENTER,
            label,
            FontId::proportional(13.0),
            TEXT,
        );
        painter.text(
            bubble.center() + egui::vec2(0.0, 10.0),
            Align2::CENTER_CENTER,
            format!("{stack_bb:.1} bb"),
            FontId::proportional(12.0),
            TEXT,
        );
    }
}

fn draw_hole_card(ui: &mut Ui, card: &str) {
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
    let (rect, _) = ui.allocate_exact_size(egui::vec2(46.0, 64.0), Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 6.0, Color32::WHITE);
    painter.rect_stroke(rect, 6.0, Stroke::new(1.0_f32, Color32::from_gray(160)));
    painter.text(
        rect.center() - egui::vec2(0.0, 10.0),
        Align2::CENTER_CENTER,
        rank.to_string(),
        FontId::proportional(18.0),
        color,
    );
    painter.text(
        rect.center() + egui::vec2(0.0, 12.0),
        Align2::CENTER_CENTER,
        suit_glyph,
        FontId::proportional(16.0),
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
}
