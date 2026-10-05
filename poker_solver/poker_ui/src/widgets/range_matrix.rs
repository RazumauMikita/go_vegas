use poker_core::{combo_index, combo_label, index_to_ranks, Card};

const CELL_MIN: f32 = 14.0;
const GRID_GAP: f32 = 1.0;
const BAR_H: f32 = 20.0;
const HEADER_H: f32 = 24.0;
const IN_RANGE_EPS: f64 = 0.5;

const HRC_GREEN: egui::Color32 = egui::Color32::from_rgb(110, 198, 82);
const HRC_RED: egui::Color32 = egui::Color32::from_rgb(240, 148, 148);
const HRC_WHITE: egui::Color32 = egui::Color32::from_rgb(252, 252, 252);
const HRC_PURPLE: egui::Color32 = egui::Color32::from_rgb(186, 88, 186);
const HRC_CALL: egui::Color32 = egui::Color32::from_rgb(196, 86, 86);
const HRC_FOLD: egui::Color32 = egui::Color32::from_rgb(196, 196, 196);
const HRC_HEADER: egui::Color32 = egui::Color32::from_rgb(70, 141, 196);
const TEXT_DARK: egui::Color32 = egui::Color32::from_rgb(32, 32, 32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatrixMode {
    #[default]
    Ev,
    Frequency,
}

pub fn combo_share(range: &[f64; 169]) -> f64 {
    const TOTAL_COMBOS: f64 = 1326.0;
    let weighted: f64 = range
        .iter()
        .enumerate()
        .map(|(idx, &freq)| freq.clamp(0.0, 1.0) * combo_weight(idx as u8) as f64)
        .sum();
    weighted / TOTAL_COMBOS
}

pub fn rank_hands_for_slider(range: &[f64; 169], evs: Option<&[f64; 169]>) -> Vec<usize> {
    let mut ranks: Vec<usize> = (0..169).collect();
    ranks.sort_by(|&a, &b| {
        if let Some(evs) = evs {
            let order = evs[b]
                .partial_cmp(&evs[a])
                .unwrap_or(std::cmp::Ordering::Equal);
            if order != std::cmp::Ordering::Equal {
                return order;
            }
        }
        let in_range = range[b]
            .partial_cmp(&range[a])
            .unwrap_or(std::cmp::Ordering::Equal);
        if in_range != std::cmp::Ordering::Equal {
            return in_range;
        }
        hand_class_rank(b).cmp(&hand_class_rank(a))
    });
    ranks
}

pub fn fill_range_to_share(ranking: &[usize], target_share: f64) -> [f64; 169] {
    const TOTAL_COMBOS: f64 = 1326.0;
    let target = target_share.clamp(0.0, 1.0) * TOTAL_COMBOS;
    let mut range = [0.0; 169];
    let mut used = 0.0;
    for &idx in ranking {
        if idx >= 169 {
            continue;
        }
        let weight = combo_weight(idx as u8) as f64;
        if used >= target - 1e-9 {
            break;
        }
        if used + weight <= target + 1e-9 {
            range[idx] = 1.0;
            used += weight;
        } else {
            range[idx] = ((target - used) / weight).clamp(0.0, 1.0);
            break;
        }
    }
    range
}

pub fn range_notation(range: &[f64; 169]) -> String {
    if range.iter().all(|&freq| freq >= IN_RANGE_EPS) {
        return "Any two".to_string();
    }
    if range.iter().all(|&freq| freq < IN_RANGE_EPS) {
        return String::new();
    }

    let mut parts = Vec::new();
    parts.extend(format_pairs(range));
    for high in (3..=14).rev() {
        parts.extend(format_nonpairs(range, high, true));
        parts.extend(format_nonpairs(range, high, false));
    }
    parts.join(" ")
}

fn hand_class_rank(idx: usize) -> (u8, u8, u8) {
    let (high, low, suited) = index_to_ranks(idx as u8);
    let class = if high == low {
        2
    } else if suited {
        1
    } else {
        0
    };
    (class, high, low)
}

pub fn range_matrix_ui(
    ui: &mut egui::Ui,
    range: &mut [f64; 169],
    hand_evs: Option<&[f64; 169]>,
    mode: &mut MatrixMode,
    title: &str,
    editable: bool,
    is_raise: bool,
) -> bool {
    let share = combo_share(range);
    let width = ui.available_width().max(13.0 * CELL_MIN);

    ui.horizontal(|ui| {
        ui.selectable_value(mode, MatrixMode::Ev, "EV");
        ui.selectable_value(mode, MatrixMode::Frequency, "Frequency");
    });

    if !title.is_empty() {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(width, HEADER_H), egui::Sense::hover());
        ui.painter().rect_filled(rect, 0.0, HRC_HEADER);
        let title_font = (rect.height() * 0.55).clamp(11.0, 16.0);
        ui.painter().text(
            rect.left_center() + egui::vec2(8.0, 0.0),
            egui::Align2::LEFT_CENTER,
            title,
            egui::FontId::proportional(title_font),
            egui::Color32::WHITE,
        );
    }

    ui.add_space(2.0);
    let grid_h = (ui.available_height() - BAR_H - 4.0).max(13.0 * CELL_MIN);
    let grid_w = ui.available_width().max(13.0 * CELL_MIN);
    let (grid_rect, _) = ui.allocate_exact_size(egui::vec2(grid_w, grid_h), egui::Sense::hover());

    let cell_w = ((grid_rect.width() - GRID_GAP * 12.0) / 13.0).max(1.0);
    let cell_h = ((grid_rect.height() - GRID_GAP * 12.0) / 13.0).max(1.0);
    let label_font = (cell_h * 0.30).clamp(8.0, 22.0);
    let value_font = (cell_h * 0.24).clamp(7.0, 18.0);

    let paint_id = ui.id().with("range_paint");
    let mut changed = false;
    let can_edit = editable && *mode == MatrixMode::Frequency;
    let sense = if can_edit {
        egui::Sense::click_and_drag()
    } else {
        egui::Sense::hover()
    };

    for row in 0..13_u8 {
        for col in 0..13_u8 {
            let idx = matrix_combo_index(row, col) as usize;
            let label = combo_label(matrix_combo_index(row, col));
            let rect = cell_rect(grid_rect, row, col, cell_w, cell_h);
            let id = ui.id().with(("range_cell", row, col));
            let response = ui.interact(rect, id, sense);

            if can_edit {
                if response.clicked() || response.drag_started() {
                    let next = if range[idx] > 0.5 { 0.0 } else { 1.0 };
                    ui.memory_mut(|mem| mem.data.insert_temp(paint_id, next));
                    if (range[idx] - next).abs() > 1e-12 {
                        range[idx] = next;
                        changed = true;
                    }
                } else if response.hovered() && ui.input(|i| i.pointer.primary_down()) {
                    if let Some(next) = ui.memory(|mem| mem.data.get_temp::<f64>(paint_id)) {
                        if (range[idx] - next).abs() > 1e-12 {
                            range[idx] = next;
                            changed = true;
                        }
                    }
                }
            }

            let (bg, value) = match *mode {
                MatrixMode::Frequency => {
                    let freq = range[idx].clamp(0.0, 1.0);
                    let color = lerp_color(
                        HRC_WHITE,
                        if is_raise { HRC_PURPLE } else { HRC_CALL },
                        freq as f32,
                    );
                    (color, format!("{:.0}", freq * 100.0))
                }
                MatrixMode::Ev => {
                    let ev = hand_evs.map(|evs| evs[idx]).unwrap_or(0.0);
                    (ev_color(ev), format!("{ev:+.2}"))
                }
            };

            ui.painter().rect_filled(rect, 0.0, bg);
            ui.painter().text(
                egui::pos2(rect.center().x, rect.top() + cell_h * 0.32),
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::proportional(label_font),
                TEXT_DARK,
            );
            ui.painter().text(
                egui::pos2(rect.center().x, rect.bottom() - cell_h * 0.28),
                egui::Align2::CENTER_CENTER,
                value,
                egui::FontId::proportional(value_font),
                TEXT_DARK,
            );
        }
    }

    ui.add_space(2.0);
    action_bar_ui(ui, share, is_raise, grid_w);
    changed
}

fn cell_rect(grid: egui::Rect, row: u8, col: u8, cell_w: f32, cell_h: f32) -> egui::Rect {
    let x = grid.left() + col as f32 * (cell_w + GRID_GAP);
    let y = grid.top() + row as f32 * (cell_h + GRID_GAP);
    egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(cell_w, cell_h))
}

fn action_bar_ui(ui: &mut egui::Ui, share: f64, is_raise: bool, width: f32) {
    let play = share.clamp(0.0, 1.0) as f32;
    let fold = 1.0 - play;
    let height = BAR_H;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let fold_w = rect.width() * fold;
    let play_w = rect.width() * play;
    let fold_rect = egui::Rect::from_min_size(rect.min, egui::vec2(fold_w, height));
    let play_rect =
        egui::Rect::from_min_size(rect.min + egui::vec2(fold_w, 0.0), egui::vec2(play_w, height));

    ui.painter().rect_filled(fold_rect, 0.0, HRC_FOLD);
    ui.painter().rect_filled(
        play_rect,
        0.0,
        if is_raise { HRC_PURPLE } else { HRC_CALL },
    );

    if fold > 0.08 {
        ui.painter().text(
            fold_rect.center(),
            egui::Align2::CENTER_CENTER,
            format!("{:.1}", fold * 100.0),
            egui::FontId::proportional(11.0),
            TEXT_DARK,
        );
    }
    if play > 0.08 {
        ui.painter().text(
            play_rect.center(),
            egui::Align2::CENTER_CENTER,
            format!("{:.1}", play * 100.0),
            egui::FontId::proportional(11.0),
            egui::Color32::WHITE,
        );
    }
}

fn ev_color(ev: f64) -> egui::Color32 {
    if ev.abs() < 5e-3 {
        return HRC_WHITE;
    }
    let t = ((ev.abs() as f32 / 1.2).clamp(0.0, 1.0)).sqrt();
    if ev > 0.0 {
        lerp_color(HRC_WHITE, HRC_GREEN, t)
    } else {
        lerp_color(HRC_WHITE, HRC_RED, t)
    }
}

fn lerp_color(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    egui::Color32::from_rgb(
        (a.r() as f32 + (b.r() as f32 - a.r() as f32) * t).round() as u8,
        (a.g() as f32 + (b.g() as f32 - a.g() as f32) * t).round() as u8,
        (a.b() as f32 + (b.b() as f32 - a.b() as f32) * t).round() as u8,
    )
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

fn matrix_combo_index(row: u8, col: u8) -> u8 {
    let rank_row = 14 - row;
    let rank_col = 14 - col;
    if row == col {
        combo_index([Card::new(rank_row, 0), Card::new(rank_row, 1)])
    } else if row < col {
        combo_index([Card::new(rank_row, 0), Card::new(rank_col, 0)])
    } else {
        combo_index([Card::new(rank_row, 0), Card::new(rank_col, 1)])
    }
}

fn combo_idx(high: u8, low: u8, suited: bool) -> usize {
    let idx = if high == low {
        combo_index([Card::new(high, 0), Card::new(high, 1)])
    } else if suited {
        combo_index([Card::new(high, 0), Card::new(low, 0)])
    } else {
        combo_index([Card::new(high, 0), Card::new(low, 1)])
    };
    idx as usize
}

fn rank_char(rank: u8) -> char {
    Card::new(rank, 0).rank_char()
}

fn pair_label(rank: u8) -> String {
    format!("{}{}", rank_char(rank), rank_char(rank))
}

fn format_pairs(range: &[f64; 169]) -> Vec<String> {
    let selected: Vec<bool> = (0..13)
        .map(|i| {
            let rank = 14 - i as u8;
            range[combo_idx(rank, rank, false)] >= IN_RANGE_EPS
        })
        .collect();

    let mut parts = Vec::new();
    let mut i = 0;
    while i < 13 {
        if !selected[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < 13 && selected[i] {
            i += 1;
        }
        let end = i - 1;
        let high_rank = 14 - start as u8;
        let low_rank = 14 - end as u8;
        if start == 0 {
            parts.push(format!("{}+", pair_label(low_rank)));
        } else if end == 12 {
            parts.push(format!("{}-", pair_label(high_rank)));
        } else if start == end {
            parts.push(pair_label(high_rank));
        } else {
            parts.push(format!(
                "{}-{}",
                pair_label(high_rank),
                pair_label(low_rank)
            ));
        }
    }
    parts
}

fn format_nonpairs(range: &[f64; 169], high: u8, suited: bool) -> Vec<String> {
    let suffix = if suited { "s" } else { "o" };
    let kickers: Vec<u8> = (2..high).rev().collect();
    let selected: Vec<bool> = kickers
        .iter()
        .map(|&low| range[combo_idx(high, low, suited)] >= IN_RANGE_EPS)
        .collect();

    let mut parts = Vec::new();
    let mut i = 0;
    while i < selected.len() {
        if !selected[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < selected.len() && selected[i] {
            i += 1;
        }
        let end = i - 1;
        let start_low = kickers[start];
        let end_low = kickers[end];
        let high_c = rank_char(high);
        if start == 0 {
            if start == end {
                parts.push(format!("{high_c}{}{suffix}", rank_char(start_low)));
            } else {
                parts.push(format!("{high_c}{}{suffix}+", rank_char(end_low)));
            }
        } else if start == end {
            parts.push(format!("{high_c}{}{suffix}", rank_char(start_low)));
        } else {
            parts.push(format!(
                "{high_c}{}{suffix}-{high_c}{}{suffix}",
                rank_char(start_low),
                rank_char(end_low)
            ));
        }
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notation_any_two() {
        assert_eq!(range_notation(&[1.0; 169]), "Any two");
    }

    #[test]
    fn notation_empty() {
        assert_eq!(range_notation(&[0.0; 169]), "");
    }

    #[test]
    fn notation_all_pairs() {
        let mut range = [0.0; 169];
        for rank in 2..=14 {
            range[combo_idx(rank, rank, false)] = 1.0;
        }
        assert_eq!(range_notation(&range), "22+");
    }

    #[test]
    fn notation_low_pairs() {
        let mut range = [0.0; 169];
        for rank in 2..=10 {
            range[combo_idx(rank, rank, false)] = 1.0;
        }
        assert_eq!(range_notation(&range), "TT-");
    }

    #[test]
    fn notation_ace_suited_plus() {
        let mut range = [0.0; 169];
        for low in 2..14 {
            range[combo_idx(14, low, true)] = 1.0;
        }
        assert_eq!(range_notation(&range), "A2s+");
    }

    #[test]
    fn notation_single_hand() {
        let mut range = [0.0; 169];
        range[combo_idx(14, 13, true)] = 1.0;
        assert_eq!(range_notation(&range), "AKs");
    }

    #[test]
    fn default_matrix_mode_is_ev() {
        assert_eq!(MatrixMode::default(), MatrixMode::Ev);
    }
}
