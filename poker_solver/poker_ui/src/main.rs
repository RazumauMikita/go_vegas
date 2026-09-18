#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![forbid(unsafe_code)]

mod app;
mod tabs;
mod util;
mod widgets;

use app::PokerApp;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("Poker Solver")
            .with_icon(
                eframe::icon_data::from_png_bytes(include_bytes!("../assets/icon.png"))
                    .expect("icon"),
            ),
        ..Default::default()
    };

    eframe::run_native(
        "Poker Solver",
        options,
        Box::new(|_cc| Ok(Box::new(PokerApp::default()))),
    )
}
