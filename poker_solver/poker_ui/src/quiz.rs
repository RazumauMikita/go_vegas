use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use poker_core::{combo_index, combo_label, table_position_label, Algorithm, Card, SolverInput};
use serde::{Deserialize, Serialize};

pub const DEFAULT_CATEGORY: &str = "Проблемные раздачи";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuizCase {
    pub id: String,
    pub category: String,
    pub source: String,
    pub player_count: usize,
    pub stacks: Vec<f64>,
    pub prize_percents: Vec<String>,
    pub small_blind: f64,
    pub big_blind: f64,
    pub ante: f64,
    pub actions_before: Vec<(String, bool)>,
    pub hero_label: String,
    pub hero_combo: u8,
    pub hero_combo_label: String,
    pub hero_cards: Option<[String; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuizLibrary {
    pub cases: Vec<QuizCase>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuizSourceKind {
    Generated,
    Saved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuizTaskMode {
    Hand,
    Range,
}

#[derive(Debug, Clone)]
pub struct QuizSettings {
    pub source: QuizSourceKind,
    pub category: String,
    pub players_exact: bool,
    pub players: usize,
    pub players_min: usize,
    pub players_max: usize,
    pub stacks_exact: bool,
    pub stack_bb: f64,
    pub stack_min_bb: f64,
    pub stack_max_bb: f64,
    pub per_seat: bool,
    pub seat_stacks_bb: Vec<f64>,
    pub prize_percents: Vec<String>,
    pub question_count: usize,
    pub include_facing_shove: bool,
    pub task_mode: QuizTaskMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuizVerdict {
    Correct,
    Mixed,
    Wrong,
}

pub struct Rng(u64);

impl Default for QuizSettings {
    fn default() -> Self {
        Self {
            source: QuizSourceKind::Generated,
            category: DEFAULT_CATEGORY.to_string(),
            players_exact: false,
            players: 3,
            players_min: 2,
            players_max: 6,
            stacks_exact: false,
            stack_bb: 10.0,
            stack_min_bb: 5.0,
            stack_max_bb: 15.0,
            per_seat: false,
            seat_stacks_bb: vec![10.0; 3],
            prize_percents: default_prize_strings(3),
            question_count: 10,
            include_facing_shove: false,
            task_mode: QuizTaskMode::Hand,
        }
    }
}

impl QuizLibrary {
    pub fn load() -> Self {
        Self::load_from(&library_path()).unwrap_or_default()
    }

    pub fn save(&self) -> Result<(), String> {
        self.save_to(&library_path())
    }

    pub fn load_from(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|err| err.to_string())?;
        serde_json::from_str(&text).map_err(|err| err.to_string())
    }

    pub fn save_to(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
            }
        }
        let text = serde_json::to_string_pretty(self).map_err(|err| err.to_string())?;
        std::fs::write(path, text).map_err(|err| err.to_string())
    }

    pub fn categories(&self) -> Vec<String> {
        let mut categories = Vec::new();
        push_unique(&mut categories, DEFAULT_CATEGORY);
        for case in &self.cases {
            push_unique(&mut categories, &case.category);
        }
        categories
    }

    pub fn add(&mut self, case: QuizCase) {
        self.cases.retain(|existing| existing.id != case.id);
        self.cases.push(case);
    }

    pub fn remove(&mut self, id: &str) {
        self.cases.retain(|case| case.id != id);
    }

    pub fn cases_in(&self, category: &str) -> Vec<&QuizCase> {
        self.cases
            .iter()
            .filter(|case| case.category == category)
            .collect()
    }
}

impl Rng {
    pub fn new() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E3779B97F4A7C15);
        Self::seed(nanos)
    }

    pub fn seed(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    pub fn gen_range(&mut self, lo: u64, hi: u64) -> u64 {
        if hi <= lo {
            return lo;
        }
        lo + self.next_u64() % (hi - lo + 1)
    }

    pub fn gen_f64(&mut self, lo: f64, hi: f64) -> f64 {
        if hi <= lo {
            return lo;
        }
        let unit = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        lo + (hi - lo) * unit
    }
}

pub fn new_id() -> String {
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}-{}", t.as_millis(), t.subsec_nanos())
}

pub fn generate_question(settings: &QuizSettings, rng: &mut Rng) -> Result<QuizCase, String> {
    let n = sample_player_count(settings, rng)?;
    let bb = 100.0;
    let stacks = sample_stacks(settings, n, bb, rng)?;
    let facing = settings.include_facing_shove && n >= 2 && rng.gen_range(0, 1) == 0;
    let hero_index = if facing {
        rng.gen_range(1, (n - 1) as u64) as usize
    } else {
        rng.gen_range(0, n.saturating_sub(2) as u64) as usize
    };
    let hero_label = table_position_label(n, hero_index).to_string();
    let actions_before = prior_actions(n, hero_index, facing, rng);
    let (cards, combo, label) = random_hole_cards(rng);
    Ok(QuizCase {
        id: new_id(),
        category: "Сгенерировано".into(),
        source: "generated".into(),
        player_count: n,
        stacks,
        prize_percents: prizes_for(n, &settings.prize_percents),
        small_blind: 50.0,
        big_blind: bb,
        ante: 0.0,
        actions_before,
        hero_label,
        hero_combo: combo,
        hero_combo_label: label,
        hero_cards: Some([cards[0].to_string(), cards[1].to_string()]),
    })
}

pub fn questions_from_library(
    library: &QuizLibrary,
    category: &str,
    count: usize,
    rng: &mut Rng,
) -> Result<Vec<QuizCase>, String> {
    let mut cases: Vec<QuizCase> = library.cases_in(category).into_iter().cloned().collect();
    if cases.is_empty() {
        return Err(format!("в категории «{category}» нет сохранённых раздач"));
    }
    shuffle(&mut cases, rng);
    if count > 0 && count < cases.len() {
        cases.truncate(count);
    }
    Ok(cases)
}

pub fn generate_quiz(
    settings: &QuizSettings,
    library: &QuizLibrary,
) -> Result<Vec<QuizCase>, String> {
    let mut rng = Rng::new();
    match settings.source {
        QuizSourceKind::Saved => questions_from_library(
            library,
            &settings.category,
            settings.question_count,
            &mut rng,
        ),
        QuizSourceKind::Generated => {
            let count = settings.question_count.max(1);
            let mut spots = Vec::with_capacity(count);
            for _ in 0..count {
                spots.push(generate_question(settings, &mut rng)?);
            }
            Ok(spots)
        }
    }
}

pub fn solver_input_for(case: &QuizCase) -> Result<SolverInput, String> {
    let n = case.player_count;
    if !(2..=9).contains(&n) {
        return Err("солвер поддерживает 2–9 игроков".into());
    }
    if case.stacks.len() != n {
        return Err("количество стеков не совпадает с числом игроков".into());
    }
    if case.stacks.iter().any(|&stack| stack <= 0.0) {
        return Err("стеки должны быть больше 0".into());
    }
    let payouts = parse_prize_payouts(&case.prize_percents)?;
    Ok(SolverInput {
        stacks: case.stacks.clone(),
        payouts,
        small_blind: case.small_blind,
        big_blind: case.big_blind.max(1e-9),
        ante: case.ante.max(0.0),
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

pub fn grade_range(user: &[f64; 169], nash: &[f64; 169]) -> (QuizVerdict, f64) {
    let mut abs = 0.0;
    let mut total = 0.0;
    for index in 0..169 {
        let weight = combo_weight(index);
        abs += (user[index].clamp(0.0, 1.0) - nash[index].clamp(0.0, 1.0)).abs() * weight;
        total += weight;
    }
    let accuracy = if total <= 0.0 {
        0.0
    } else {
        (1.0 - abs / total).clamp(0.0, 1.0)
    };
    let verdict = if accuracy >= 0.90 {
        QuizVerdict::Correct
    } else if accuracy >= 0.75 {
        QuizVerdict::Mixed
    } else {
        QuizVerdict::Wrong
    };
    (verdict, accuracy)
}

fn combo_weight(index: usize) -> f64 {
    if index < 13 {
        6.0
    } else if index < 91 {
        4.0
    } else {
        12.0
    }
}

pub fn grade_answer(freq: f64, ev: Option<f64>, took_action: bool) -> QuizVerdict {
    let nash_action = if let Some(ev) = ev {
        if ev.abs() < 0.10 {
            return QuizVerdict::Mixed;
        }
        ev > 0.0
    } else {
        if (0.15..0.85).contains(&freq) {
            return QuizVerdict::Mixed;
        }
        freq >= 0.5
    };
    if took_action == nash_action {
        QuizVerdict::Correct
    } else {
        QuizVerdict::Wrong
    }
}

pub fn hero_seat_index(player_count: usize, hero_label: &str) -> Option<usize> {
    (0..player_count)
        .find(|&index| same_seat(table_position_label(player_count, index), hero_label))
}

fn same_seat(left: &str, right: &str) -> bool {
    fn norm(name: &str) -> &str {
        match name {
            "BU" => "BTN",
            other => other,
        }
    }
    norm(left) == norm(right)
}

pub fn default_prize_strings(players: usize) -> Vec<String> {
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

fn sample_player_count(settings: &QuizSettings, rng: &mut Rng) -> Result<usize, String> {
    let n = if settings.players_exact {
        settings.players
    } else {
        let lo = settings.players_min.min(settings.players_max).max(2);
        let hi = settings.players_max.max(settings.players_min).min(9);
        rng.gen_range(lo as u64, hi as u64) as usize
    };
    if !(2..=9).contains(&n) {
        return Err("число игроков должно быть от 2 до 9".into());
    }
    Ok(n)
}

fn sample_stacks(
    settings: &QuizSettings,
    n: usize,
    bb: f64,
    rng: &mut Rng,
) -> Result<Vec<f64>, String> {
    if settings.stacks_exact {
        if settings.per_seat && settings.players_exact {
            if settings.seat_stacks_bb.len() < n {
                return Err("не заданы стеки по позициям".into());
            }
            return Ok(settings.seat_stacks_bb[..n]
                .iter()
                .map(|stack_bb| (stack_bb.max(1.0)) * bb)
                .collect());
        }
        let stack = settings.stack_bb.max(1.0) * bb;
        return Ok(vec![stack; n]);
    }
    let lo = settings.stack_min_bb.min(settings.stack_max_bb).max(1.0);
    let hi = settings.stack_max_bb.max(settings.stack_min_bb).max(lo);
    Ok((0..n)
        .map(|_| round_half_bb(rng.gen_f64(lo, hi)) * bb)
        .collect())
}

fn prior_actions(n: usize, hero_index: usize, facing: bool, rng: &mut Rng) -> Vec<(String, bool)> {
    if hero_index == 0 {
        return Vec::new();
    }
    let shover = if facing {
        Some(rng.gen_range(0, (hero_index - 1) as u64) as usize)
    } else {
        None
    };
    (0..hero_index)
        .map(|index| {
            let label = table_position_label(n, index).to_string();
            (label, shover != Some(index))
        })
        .collect()
}

fn random_hole_cards(rng: &mut Rng) -> ([Card; 2], u8, String) {
    let first = rng.gen_range(0, 51) as usize;
    let mut second = rng.gen_range(0, 50) as usize;
    if second >= first {
        second += 1;
    }
    let cards = [card_from_deck(first), card_from_deck(second)];
    let combo = combo_index(cards);
    (cards, combo, combo_label(combo))
}

fn card_from_deck(index: usize) -> Card {
    let rank = (index / 4) as u8 + 2;
    let suit = (index % 4) as u8;
    Card::new(rank, suit)
}

fn prizes_for(n: usize, template: &[String]) -> Vec<String> {
    let mut out = template.to_vec();
    out.truncate(n);
    while out.len() < n {
        out.push(String::new());
    }
    if out.iter().all(|text| text.trim().is_empty()) {
        return default_prize_strings(n);
    }
    out
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

pub fn dealer_seat(players: usize) -> usize {
    button_index_for(players)
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
        _ => 80,
    }
}

fn round_half_bb(value: f64) -> f64 {
    (value * 2.0).round() / 2.0
}

fn shuffle<T>(items: &mut [T], rng: &mut Rng) {
    for i in (1..items.len()).rev() {
        let j = rng.gen_range(0, i as u64) as usize;
        items.swap(i, j);
    }
}

fn push_unique(items: &mut Vec<String>, value: &str) {
    if !items.iter().any(|item| item == value) {
        items.push(value.to_string());
    }
}

fn library_path() -> PathBuf {
    if let Ok(appdata) = std::env::var("APPDATA") {
        return PathBuf::from(appdata)
            .join("poker_solver")
            .join("quiz_library.json");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".poker_solver")
            .join("quiz_library.json");
    }
    PathBuf::from("quiz_library.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_roundtrip() {
        let dir = std::env::temp_dir().join(format!("quiz-lib-{}", new_id()));
        let path = dir.join("quiz_library.json");
        let mut library = QuizLibrary::default();
        library.add(sample_case("Проблемные раздачи"));
        library.save_to(&path).expect("save");
        let loaded = QuizLibrary::load_from(&path).expect("load");
        assert_eq!(loaded.cases.len(), 1);
        assert_eq!(loaded.cases[0].hero_label, "BTN");
        assert!(loaded.categories().contains(&DEFAULT_CATEGORY.to_string()));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn generated_player_count_stays_in_range() {
        let mut settings = QuizSettings::default();
        settings.players_exact = false;
        settings.players_min = 3;
        settings.players_max = 5;
        let mut rng = Rng::seed(7);
        for _ in 0..40 {
            let spot = generate_question(&settings, &mut rng).unwrap();
            assert!((3..=5).contains(&spot.player_count));
            assert_eq!(spot.stacks.len(), spot.player_count);
        }
    }

    #[test]
    fn generated_stacks_stay_in_range() {
        let mut settings = QuizSettings::default();
        settings.players_exact = true;
        settings.players = 3;
        settings.stacks_exact = false;
        settings.stack_min_bb = 8.0;
        settings.stack_max_bb = 12.0;
        let mut rng = Rng::seed(11);
        let spot = generate_question(&settings, &mut rng).unwrap();
        for stack in spot.stacks {
            let bb = stack / 100.0;
            assert!(bb >= 8.0 - 1e-9 && bb <= 12.0 + 1e-9);
        }
    }

    #[test]
    fn exact_equal_stacks() {
        let mut settings = QuizSettings::default();
        settings.players_exact = true;
        settings.players = 4;
        settings.stacks_exact = true;
        settings.per_seat = false;
        settings.stack_bb = 10.0;
        let mut rng = Rng::seed(3);
        let spot = generate_question(&settings, &mut rng).unwrap();
        assert_eq!(spot.player_count, 4);
        assert!(spot
            .stacks
            .iter()
            .all(|&stack| (stack - 1000.0).abs() < 1e-9));
    }

    #[test]
    fn per_seat_stacks() {
        let mut settings = QuizSettings::default();
        settings.players_exact = true;
        settings.players = 3;
        settings.stacks_exact = true;
        settings.per_seat = true;
        settings.seat_stacks_bb = vec![12.0, 8.0, 6.5];
        let mut rng = Rng::seed(5);
        let spot = generate_question(&settings, &mut rng).unwrap();
        assert_eq!(spot.stacks, vec![1200.0, 800.0, 650.0]);
    }

    #[test]
    fn unopened_hero_is_never_bb() {
        let mut settings = QuizSettings::default();
        settings.include_facing_shove = false;
        settings.players_exact = false;
        settings.players_min = 2;
        settings.players_max = 6;
        let mut rng = Rng::seed(99);
        for _ in 0..80 {
            let spot = generate_question(&settings, &mut rng).unwrap();
            let hero = hero_seat_index(spot.player_count, &spot.hero_label).unwrap();
            assert_ne!(hero, spot.player_count - 1);
        }
    }

    #[test]
    fn hero_seat_accepts_bu_alias() {
        assert_eq!(hero_seat_index(3, "BU"), Some(0));
        assert_eq!(hero_seat_index(4, "BU"), Some(1));
        assert_eq!(hero_seat_index(2, "SB"), Some(0));
    }

    #[test]
    fn unopened_actions_are_folds() {
        let mut settings = QuizSettings::default();
        settings.include_facing_shove = false;
        let mut rng = Rng::seed(21);
        for _ in 0..20 {
            let spot = generate_question(&settings, &mut rng).unwrap();
            assert!(spot.actions_before.iter().all(|(_, is_fold)| *is_fold));
        }
    }

    #[test]
    fn facing_shove_has_one_shove() {
        let mut settings = QuizSettings::default();
        settings.players_exact = true;
        settings.players = 6;
        settings.include_facing_shove = true;
        let mut rng = Rng::seed(42);
        let mut saw_shove = false;
        for _ in 0..30 {
            let spot = generate_question(&settings, &mut rng).unwrap();
            let shoves = spot
                .actions_before
                .iter()
                .filter(|(_, is_fold)| !*is_fold)
                .count();
            if shoves == 1 {
                saw_shove = true;
                let hero = hero_seat_index(spot.player_count, &spot.hero_label).unwrap();
                assert!(hero >= 1);
            }
        }
        assert!(saw_shove);
    }

    #[test]
    fn grades_range_overlap() {
        let nash = [1.0; 169];
        let user = [1.0; 169];
        let (verdict, accuracy) = grade_range(&user, &nash);
        assert_eq!(verdict, QuizVerdict::Correct);
        assert!(accuracy > 0.99);

        let empty = [0.0; 169];
        let (verdict, accuracy) = grade_range(&empty, &nash);
        assert_eq!(verdict, QuizVerdict::Wrong);
        assert!(accuracy < 0.05);
    }

    #[test]
    fn grades_mixed_and_clear_spots() {
        assert_eq!(grade_answer(0.9, Some(1.4), true), QuizVerdict::Correct);
        assert_eq!(grade_answer(0.9, Some(1.4), false), QuizVerdict::Wrong);
        assert_eq!(grade_answer(0.4, Some(-0.8), false), QuizVerdict::Correct);
        assert_eq!(grade_answer(0.5, Some(0.02), true), QuizVerdict::Mixed);
        assert_eq!(grade_answer(0.4, None, true), QuizVerdict::Mixed);
    }

    #[test]
    fn dealer_seat_matches_solver_order() {
        assert_eq!(dealer_seat(2), 0);
        assert_eq!(dealer_seat(3), 0);
        assert_eq!(dealer_seat(6), 3);
        assert_eq!(dealer_seat(9), 6);
    }

    #[test]
    fn solver_input_matches_case() {
        let input = solver_input_for(&sample_case("X")).unwrap();
        assert_eq!(input.num_players, 3);
        assert_eq!(input.stacks.len(), 3);
        assert_eq!(input.payouts, vec![0.5, 0.3, 0.2]);
    }

    fn sample_case(category: &str) -> QuizCase {
        QuizCase {
            id: "case-1".into(),
            category: category.into(),
            source: "403 / 1".into(),
            player_count: 3,
            stacks: vec![1000.0, 1000.0, 1000.0],
            prize_percents: vec!["50".into(), "30".into(), "20".into()],
            small_blind: 50.0,
            big_blind: 100.0,
            ante: 0.0,
            actions_before: Vec::new(),
            hero_label: "BTN".into(),
            hero_combo: 0,
            hero_combo_label: "AA".into(),
            hero_cards: Some(["Ah".into(), "Ad".into()]),
        }
    }
}
