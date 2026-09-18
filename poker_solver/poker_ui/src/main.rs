#![forbid(unsafe_code)]

mod app;
mod tabs;
mod util;
mod widgets;

use app::PokerApp;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 800.0])
            .with_title("Poker Solver"),
        ..Default::default()
    };

    eframe::run_native(
        "Poker Solver",
        options,
        Box::new(|_cc| Ok(Box::new(PokerApp::default()))),
    )
}
