use egui::{Context, Ui};

use crate::tabs::{EquityTab, EvalTab, IcmTab, SolverTab};

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
}

impl Default for PokerApp {
    fn default() -> Self {
        Self {
            active_tab: Tab::Solver,
            solver: SolverTab::default(),
            equity: EquityTab::default(),
            icm: IcmTab::default(),
            eval: EvalTab::default(),
        }
    }
}

impl eframe::App for PokerApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        self.solver.poll(ctx);

        egui::TopBottomPanel::top("app_header").show(ctx, |ui| {
            ui.heading("Poker Solver");
            ui.add_space(4.0);
            self.draw_tabs(ui);
            ui.separator();
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
        ui.horizontal(|ui| {
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
