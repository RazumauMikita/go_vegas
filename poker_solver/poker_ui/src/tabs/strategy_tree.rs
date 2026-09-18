use egui::{Color32, RichText, Ui};
use poker_core::{SolverOutput, ThreeMaxHandEvs, ThreeMaxRanges};

use crate::widgets::combo_share;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Raise,
    Call,
    CallSpecial,
}

#[derive(Clone)]
pub struct TreeNode {
    pub label: String,
    pub action: Action,
    pub range_pct: f64,
    pub range: [f64; 169],
    pub ev_range: Option<[f64; 169]>,
    pub children: Vec<TreeNode>,
    pub expanded: bool,
}

pub type NodePath = Vec<usize>;

pub fn build_strategy_tree(output: &SolverOutput) -> Vec<TreeNode> {
    if let Some(ranges) = output.three_max.as_ref() {
        let hand_evs = output.three_max_hand_evs.as_ref();
        build_3max_tree(ranges, hand_evs)
    } else {
        build_hu_tree(output)
    }
}

pub fn node_at_path<'a>(tree: &'a [TreeNode], path: &[usize]) -> Option<&'a TreeNode> {
    if path.is_empty() {
        return None;
    }
    let mut current: &TreeNode = tree.get(path[0])?;
    for &index in path.iter().skip(1) {
        current = current.children.get(index)?;
    }
    Some(current)
}

pub fn draw_strategy_tree(
    ui: &mut Ui,
    tree: &mut [TreeNode],
    selected_path: &mut Option<NodePath>,
    path_prefix: &[usize],
    depth: usize,
) {
    for (index, node) in tree.iter_mut().enumerate() {
        let current_path: NodePath = path_prefix
            .iter()
            .chain(std::iter::once(&index))
            .copied()
            .collect();
        let is_selected = selected_path.as_ref() == Some(&current_path);

        let indent = depth as f32 * 14.0;
        ui.horizontal(|ui| {
            ui.add_space(indent);

            if node.children.is_empty() {
                ui.add_space(18.0);
            } else {
                let icon = if node.expanded { "▼" } else { "▶" };
                if ui.small_button(icon).clicked() {
                    node.expanded = !node.expanded;
                }
            }

            let marker = if is_selected { "●" } else { "○" };
            let action_char = match node.action {
                Action::Raise => "R",
                Action::Call | Action::CallSpecial => "C",
            };
            let row_text = format!(
                "{marker} {action_char} {} {:.1}%",
                node.label, node.range_pct
            );

            let action_color = action_color(node.action);
            let bg = if is_selected {
                Color32::from_rgb(100, 150, 220)
            } else {
                ui.visuals().widgets.noninteractive.bg_fill
            };

            let response = ui.add(
                egui::Button::new(RichText::new(row_text).color(action_color))
                    .fill(bg)
                    .stroke(egui::Stroke::NONE),
            );
            if response.clicked() {
                *selected_path = Some(current_path.clone());
            }
        });

        if node.expanded && !node.children.is_empty() {
            draw_strategy_tree(
                ui,
                &mut node.children,
                selected_path,
                &current_path,
                depth + 1,
            );
        }
    }
}

fn action_color(action: Action) -> Color32 {
    match action {
        Action::Raise => Color32::from_rgb(80, 200, 80),
        Action::Call => Color32::from_rgb(220, 80, 80),
        Action::CallSpecial => Color32::from_rgb(100, 150, 220),
    }
}

fn make_node(
    label: &str,
    action: Action,
    range: [f64; 169],
    ev_range: Option<[f64; 169]>,
    children: Vec<TreeNode>,
) -> TreeNode {
    TreeNode {
        label: label.to_string(),
        action,
        range_pct: combo_share(&range) * 100.0,
        range,
        ev_range,
        children,
        expanded: true,
    }
}

fn build_3max_tree(ranges: &ThreeMaxRanges, hand_evs: Option<&ThreeMaxHandEvs>) -> Vec<TreeNode> {
    let evs = hand_evs;
    vec![
        make_node(
            "BTN push",
            Action::Raise,
            ranges.btn_push,
            evs.map(|e| e.btn_push),
            vec![make_node(
                "SB call",
                Action::Call,
                ranges.sb_call_vs_btn,
                evs.map(|e| e.sb_call_vs_btn),
                vec![
                    make_node(
                        "BB call (vs BTN+SB)",
                        Action::CallSpecial,
                        ranges.bb_call_vs_btn_and_sb,
                        evs.map(|e| e.bb_call_vs_btn_and_sb),
                        vec![],
                    ),
                    make_node(
                        "BB call (vs BTN)",
                        Action::Call,
                        ranges.bb_call_vs_btn,
                        evs.map(|e| e.bb_call_vs_btn),
                        vec![],
                    ),
                ],
            )],
        ),
        make_node(
            "SB push (BTN fold)",
            Action::Raise,
            ranges.sb_push,
            evs.map(|e| e.sb_push),
            vec![make_node(
                "BB call (vs SB)",
                Action::Call,
                ranges.bb_call_vs_sb,
                evs.map(|e| e.bb_call_vs_sb),
                vec![],
            )],
        ),
    ]
}

fn build_hu_tree(output: &SolverOutput) -> Vec<TreeNode> {
    vec![make_node(
        "SB push",
        Action::Raise,
        output.push_ranges[0],
        output.hand_evs.get(0).copied(),
        vec![make_node(
            "BB call",
            Action::Call,
            output.call_ranges[1],
            output.hand_evs.get(1).copied(),
            vec![],
        )],
    )]
}
