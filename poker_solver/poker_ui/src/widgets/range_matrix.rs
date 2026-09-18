use poker_core::{combo_index, combo_label, index_to_ranks, Card};

const CELL_SIZE: f32 = 40.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatrixMode {
    #[default]
    Frequency,
    Ev,
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

pub fn range_bar_ui(ui: &mut egui::Ui, share: f64) {
    let pct = share.clamp(0.0, 1.0);
    ui.horizontal(|ui| {
        ui.label(format!("Range: {:.1}%", pct * 100.0));
        ui.add(
            egui::ProgressBar::new(pct as f32)
                .desired_width(200.0)
                .fill(egui::Color32::from_rgb(220, 80, 80)),
        );
    });
}

pub fn range_matrix_ui(
    ui: &mut egui::Ui,
    range: &[f64; 169],
    hand_evs: Option<&[f64; 169]>,
    mode: &mut MatrixMode,
    title: &str,
) {
    ui.label(egui::RichText::new(title).strong());
    ui.horizontal(|ui| {
        ui.selectable_value(mode, MatrixMode::Frequency, "Frequency");
        ui.selectable_value(mode, MatrixMode::Ev, "EV");
    });
    ui.add_space(4.0);

    let max_abs_ev = hand_evs
        .map(|evs| evs.iter().map(|ev| ev.abs()).fold(0.0_f64, f64::max))
        .unwrap_or(1.0)
        .max(1e-9) as f32;

    egui::Grid::new(format!("range_matrix_{title}"))
        .spacing(egui::vec2(1.0, 1.0))
        .show(ui, |ui| {
            for row in 0..13_u8 {
                for col in 0..13_u8 {
                    let idx = matrix_combo_index(row, col) as usize;
                    let label = combo_label(matrix_combo_index(row, col));

                    let (rect, _response) = ui.allocate_exact_size(
                        egui::vec2(CELL_SIZE, CELL_SIZE),
                        egui::Sense::hover(),
                    );

                    let (bg, text) = match *mode {
                        MatrixMode::Frequency => {
                            let freq = range[idx].clamp(0.0, 1.0);
                            let t = freq as f32;
                            let color = egui::Color32::from_rgb(
                                255,
                                (255.0 * (1.0 - t)) as u8,
                                (255.0 * (1.0 - t)) as u8,
                            );
                            (color, format!("{label}\n{freq:.1}"))
                        }
                        MatrixMode::Ev => {
                            let ev = hand_evs.map(|evs| evs[idx]).unwrap_or(0.0);
                            (ev_color(ev, max_abs_ev), format!("{label}\n{ev:+.2}"))
                        }
                    };

                    ui.painter().rect_filled(rect, 2.0, bg);
                    ui.painter().rect_stroke(
                        rect,
                        2.0,
                        egui::Stroke::new(1.0_f32, egui::Color32::from_gray(180)),
                    );
                    ui.painter().text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        text,
                        egui::FontId::proportional(9.0),
                        egui::Color32::BLACK,
                    );
                }
                ui.end_row();
            }
        });

    ui.add_space(4.0);
    range_bar_ui(ui, combo_share(range));
}

fn ev_color(ev: f64, max_abs: f32) -> egui::Color32 {
    if ev.abs() < 1e-9 {
        return egui::Color32::WHITE;
    }
    let t = (ev.abs() / max_abs as f64).clamp(0.0, 1.0) as f32;
    if ev > 0.0 {
        let rb = (255.0 * (1.0 - t * 0.75)) as u8;
        egui::Color32::from_rgb(rb, 255, rb)
    } else {
        let gb = (255.0 * (1.0 - t * 0.65)) as u8;
        egui::Color32::from_rgb(255, gb, (gb + 30).min(255))
    }
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
