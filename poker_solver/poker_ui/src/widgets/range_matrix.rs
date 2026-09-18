use poker_core::{combo_index, combo_label, Card};

const CELL_SIZE: f32 = 40.0;

pub fn range_matrix_ui(ui: &mut egui::Ui, range: &[f64; 169], title: &str) {
    ui.label(egui::RichText::new(title).strong());
    ui.add_space(4.0);

    egui::Grid::new(format!("range_matrix_{title}"))
        .spacing(egui::vec2(1.0, 1.0))
        .show(ui, |ui| {
            for row in 0..13_u8 {
                for col in 0..13_u8 {
                    let idx = matrix_combo_index(row, col) as usize;
                    let freq = range[idx].clamp(0.0, 1.0);
                    let label = combo_label(matrix_combo_index(row, col));

                    let (rect, _response) = ui.allocate_exact_size(
                        egui::vec2(CELL_SIZE, CELL_SIZE),
                        egui::Sense::hover(),
                    );

                    let t = freq as f32;
                    let bg = egui::Color32::from_rgb(
                        255,
                        (255.0 * (1.0 - t)) as u8,
                        (255.0 * (1.0 - t)) as u8,
                    );

                    ui.painter().rect_filled(rect, 2.0, bg);
                    ui.painter().rect_stroke(
                        rect,
                        2.0,
                        egui::Stroke::new(1.0_f32, egui::Color32::from_gray(180)),
                    );

                    let text = format!("{label}\n{freq:.1}");
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
