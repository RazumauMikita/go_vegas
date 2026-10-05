use egui::{Context, Ui, ViewportCommand};
use poker_core::ThreeWayRankCache;

use crate::tabs::{EquityTab, EvalTab, IcmTab, SolverTab};

const THREE_WAY_RANK_CACHE_BYTES: &[u8] = include_bytes!("../assets/three_way_rank_cache.bin");
const PREFERRED_INNER_SIZE: egui::Vec2 = egui::vec2(1400.0, 900.0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Solver,
    Equity,
    Icm,
    Eval,
}

pub struct PokerApp {
    active_tab: Tab,
    solver: SolverTab,
    equity: EquityTab,
    icm: IcmTab,
    eval: EvalTab,
    window_fitted: bool,
}

impl Default for PokerApp {
    fn default() -> Self {
        ThreeWayRankCache::install_embedded(THREE_WAY_RANK_CACHE_BYTES);
        Self {
            active_tab: Tab::Solver,
            solver: SolverTab::default(),
            equity: EquityTab::default(),
            icm: IcmTab::default(),
            eval: EvalTab::default(),
            window_fitted: false,
        }
    }
}

impl eframe::App for PokerApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        if !self.window_fitted {
            self.window_fitted = fit_native_window(ctx);
        }

        self.solver.poll(ctx);

        egui::TopBottomPanel::top("app_header").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("Poker Solver");
                ui.add_space(12.0);
                self.draw_tabs(ui);
            });
        });

        match self.active_tab {
            Tab::Solver => self.solver.ui(ctx),
            Tab::Equity => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| self.equity.ui(ui));
                });
            }
            Tab::Icm => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| self.icm.ui(ui));
                });
            }
            Tab::Eval => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| self.eval.ui(ui));
                });
            }
        }
    }
}

impl PokerApp {
    fn draw_tabs(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            tab_button(ui, &mut self.active_tab, Tab::Solver, "Solver");
            tab_button(ui, &mut self.active_tab, Tab::Equity, "Equity");
            tab_button(ui, &mut self.active_tab, Tab::Icm, "ICM");
            tab_button(ui, &mut self.active_tab, Tab::Eval, "Eval");
        });
    }
}

fn tab_button(ui: &mut Ui, active: &mut Tab, tab: Tab, label: &str) {
    if ui.selectable_label(*active == tab, label).clicked() {
        *active = tab;
    }
}

fn fit_native_window(ctx: &Context) -> bool {
    let (monitor, inner, outer) = ctx.input(|i| {
        let viewport = i.viewport();
        (viewport.monitor_size, viewport.inner_rect, viewport.outer_rect)
    });
    let Some(monitor) = monitor else {
        return false;
    };
    let Some(inner) = inner else {
        return false;
    };
    let Some(outer) = outer else {
        return false;
    };
    if monitor.x < 64.0 || monitor.y < 64.0 {
        return false;
    }

    let mut chrome = egui::vec2(
        (outer.width() - inner.width()).max(0.0),
        (outer.height() - inner.height()).max(0.0),
    );
    if chrome.y < 8.0 {
        chrome.y = 32.0;
    }
    if chrome.x < 1.0 {
        chrome.x = 16.0;
    }
    let bottom_reserve = (monitor.y * 0.08).clamp(48.0, 88.0);
    let side_reserve = 16.0;
    let work = egui::vec2(
        (monitor.x - side_reserve * 2.0).max(640.0),
        (monitor.y - bottom_reserve).max(480.0),
    );
    let max_inner = egui::vec2(
        (work.x - chrome.x).max(640.0),
        (work.y - chrome.y).max(480.0),
    );
    let inner_size = egui::vec2(
        PREFERRED_INNER_SIZE.x.min(max_inner.x),
        PREFERRED_INNER_SIZE.y.min(max_inner.y),
    );
    let outer_size = inner_size + chrome;
    let pos = egui::pos2(
        ((monitor.x - outer_size.x) * 0.5).max(0.0),
        ((work.y - outer_size.y) * 0.5).max(0.0),
    );

    ctx.send_viewport_cmd(ViewportCommand::InnerSize(inner_size));
    ctx.send_viewport_cmd(ViewportCommand::OuterPosition(pos));
    ctx.request_repaint();
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tab_is_solver() {
        let app = PokerApp::default();
        assert_eq!(app.active_tab, Tab::Solver);
    }

    #[test]
    fn tab_switching() {
        let mut app = PokerApp::default();
        app.active_tab = Tab::Equity;
        assert_eq!(app.active_tab, Tab::Equity);
        app.active_tab = Tab::Icm;
        assert_eq!(app.active_tab, Tab::Icm);
        app.active_tab = Tab::Eval;
        assert_eq!(app.active_tab, Tab::Eval);
        app.active_tab = Tab::Solver;
        assert_eq!(app.active_tab, Tab::Solver);
    }
}
