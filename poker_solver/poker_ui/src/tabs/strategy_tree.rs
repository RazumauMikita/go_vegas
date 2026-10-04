use egui::{Color32, RichText, Ui};
use poker_core::{
    five_node, six_node, FiveMaxRanges, FourMaxRanges, SixMaxRanges, SolverOutput,
    ThreeMaxHandEvs, ThreeMaxRanges,
};

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

pub fn build_strategy_tree(output: &SolverOutput, button_index: usize) -> Vec<TreeNode> {
    if let Some(ranges) = output.six_max.as_ref() {
        build_6max_tree(ranges, output, button_index)
    } else if let Some(ranges) = output.five_max.as_ref() {
        build_5max_tree(ranges, output, button_index)
    } else if let Some(ranges) = output.four_max.as_ref() {
        build_4max_tree(ranges, output, button_index)
    } else if let Some(ranges) = output.three_max.as_ref() {
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

fn four_max_seat(button_index: usize, role: usize) -> usize {
    let button = button_index % 4;
    match role {
        0 => (button + 3) % 4,
        1 => button,
        2 => (button + 1) % 4,
        _ => (button + 2) % 4,
    }
}

fn seat_hand_ev(output: &SolverOutput, button_index: usize, role: usize) -> Option<[f64; 169]> {
    output
        .hand_evs
        .get(four_max_seat(button_index, role))
        .copied()
}

fn build_4max_tree(
    ranges: &FourMaxRanges,
    output: &SolverOutput,
    button_index: usize,
) -> Vec<TreeNode> {
    let ev = |role: usize| seat_hand_ev(output, button_index, role);
    vec![
        make_node(
            "CO push",
            Action::Raise,
            ranges.utg_push,
            ev(0),
            vec![
                make_node(
                    "BTN call (vs CO)",
                    Action::Call,
                    ranges.btn_call_vs_push,
                    None,
                    vec![
                        make_node(
                            "SB call (vs CO+BTN)",
                            Action::Call,
                            ranges.sb_call_vs_utg_btn,
                            None,
                            vec![make_node(
                                "BB call (4-way)",
                                Action::CallSpecial,
                                ranges.bb_call_4way,
                                ev(3),
                                vec![],
                            )],
                        ),
                        make_node(
                            "BB call (vs CO+BTN)",
                            Action::Call,
                            ranges.bb_call_vs_utg_btn,
                            None,
                            vec![],
                        ),
                    ],
                ),
                make_node(
                    "SB call (vs CO)",
                    Action::Call,
                    ranges.sb_call_vs_utg,
                    None,
                    vec![make_node(
                        "BB call (vs CO+SB)",
                        Action::Call,
                        ranges.bb_call_vs_utg_sb,
                        None,
                        vec![],
                    )],
                ),
                make_node(
                    "BB call (vs CO)",
                    Action::Call,
                    ranges.bb_call_vs_utg,
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            "BTN push (CO fold)",
            Action::Raise,
            ranges.btn_push,
            ev(1),
            vec![
                make_node(
                    "SB call (vs BTN)",
                    Action::Call,
                    ranges.sb_call_vs_btn,
                    None,
                    vec![make_node(
                        "BB call (vs BTN+SB)",
                        Action::CallSpecial,
                        ranges.bb_call_vs_btn_sb,
                        None,
                        vec![],
                    )],
                ),
                make_node(
                    "BB call (vs BTN)",
                    Action::Call,
                    ranges.bb_call_vs_btn,
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            "SB push (CO+BTN fold)",
            Action::Raise,
            ranges.sb_push,
            ev(2),
            vec![make_node(
                "BB call (vs SB)",
                Action::Call,
                ranges.bb_call_vs_sb,
                None,
                vec![],
            )],
        ),
    ]
}

fn five_max_seat(button_index: usize, role: usize) -> usize {
    let button = button_index % 5;
    match role {
        0 => (button + 3) % 5,
        1 => (button + 4) % 5,
        2 => button,
        3 => (button + 1) % 5,
        _ => (button + 2) % 5,
    }
}

fn five_seat_ev(output: &SolverOutput, button_index: usize, role: usize) -> Option<[f64; 169]> {
    output
        .hand_evs
        .get(five_max_seat(button_index, role))
        .copied()
}

fn freq_node(ranges: &FiveMaxRanges, actor: usize, mask: u32) -> [f64; 169] {
    ranges.freq[five_node(actor, mask)]
}

fn build_5max_tree(
    ranges: &FiveMaxRanges,
    output: &SolverOutput,
    button_index: usize,
) -> Vec<TreeNode> {
    let ev = |role: usize| five_seat_ev(output, button_index, role);
    let f = |actor: usize, mask: u32| freq_node(ranges, actor, mask);
    vec![
        make_node(
            "HJ push",
            Action::Raise,
            f(0, 0),
            ev(0),
            vec![
                make_node(
                    "CO call (vs HJ)",
                    Action::Call,
                    f(1, 1),
                    None,
                    vec![
                        make_node(
                            "BTN call (vs HJ+CO)",
                            Action::Call,
                            f(2, 3),
                            None,
                            vec![
                                make_node(
                                    "SB call (vs HJ+CO+BTN)",
                                    Action::Call,
                                    f(3, 7),
                                    None,
                                    vec![make_node(
                                        "BB call (5-way)",
                                        Action::CallSpecial,
                                        f(4, 15),
                                        None,
                                        vec![],
                                    )],
                                ),
                                make_node(
                                    "BB call (vs HJ+CO+BTN)",
                                    Action::Call,
                                    f(4, 7),
                                    None,
                                    vec![],
                                ),
                            ],
                        ),
                        make_node(
                            "SB call (vs HJ+CO)",
                            Action::Call,
                            f(3, 3),
                            None,
                            vec![make_node(
                                "BB call (vs HJ+CO+SB)",
                                Action::Call,
                                f(4, 11),
                                None,
                                vec![],
                            )],
                        ),
                        make_node(
                            "BB call (vs HJ+CO)",
                            Action::Call,
                            f(4, 3),
                            None,
                            vec![],
                        ),
                    ],
                ),
                make_node(
                    "BTN call (vs HJ)",
                    Action::Call,
                    f(2, 1),
                    None,
                    vec![
                        make_node(
                            "SB call (vs HJ+BTN)",
                            Action::Call,
                            f(3, 5),
                            None,
                            vec![make_node(
                                "BB call (vs HJ+BTN+SB)",
                                Action::Call,
                                f(4, 13),
                                None,
                                vec![],
                            )],
                        ),
                        make_node(
                            "BB call (vs HJ+BTN)",
                            Action::Call,
                            f(4, 5),
                            None,
                            vec![],
                        ),
                    ],
                ),
                make_node(
                    "SB call (vs HJ)",
                    Action::Call,
                    f(3, 1),
                    None,
                    vec![make_node(
                        "BB call (vs HJ+SB)",
                        Action::Call,
                        f(4, 9),
                        None,
                        vec![],
                    )],
                ),
                make_node(
                    "BB call (vs HJ)",
                    Action::Call,
                    f(4, 1),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            "CO push (HJ fold)",
            Action::Raise,
            f(1, 0),
            ev(1),
            vec![
                make_node(
                    "BTN call (vs CO)",
                    Action::Call,
                    f(2, 2),
                    None,
                    vec![
                        make_node(
                            "SB call (vs CO+BTN)",
                            Action::Call,
                            f(3, 6),
                            None,
                            vec![make_node(
                                "BB call (vs CO+BTN+SB)",
                                Action::Call,
                                f(4, 14),
                                None,
                                vec![],
                            )],
                        ),
                        make_node(
                            "BB call (vs CO+BTN)",
                            Action::Call,
                            f(4, 6),
                            None,
                            vec![],
                        ),
                    ],
                ),
                make_node(
                    "SB call (vs CO)",
                    Action::Call,
                    f(3, 2),
                    None,
                    vec![make_node(
                        "BB call (vs CO+SB)",
                        Action::Call,
                        f(4, 10),
                        None,
                        vec![],
                    )],
                ),
                make_node(
                    "BB call (vs CO)",
                    Action::Call,
                    f(4, 2),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            "BTN push (HJ+CO fold)",
            Action::Raise,
            f(2, 0),
            ev(2),
            vec![
                make_node(
                    "SB call (vs BTN)",
                    Action::Call,
                    f(3, 4),
                    None,
                    vec![make_node(
                        "BB call (vs BTN+SB)",
                        Action::CallSpecial,
                        f(4, 12),
                        None,
                        vec![],
                    )],
                ),
                make_node(
                    "BB call (vs BTN)",
                    Action::Call,
                    f(4, 4),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            "SB push (folds)",
            Action::Raise,
            f(3, 0),
            ev(3),
            vec![make_node(
                "BB call (vs SB)",
                Action::Call,
                f(4, 8),
                ev(4),
                vec![],
            )],
        ),
    ]
}

fn six_max_seat(button_index: usize, role: usize) -> usize {
    let button = button_index % 6;
    match role {
        0 => (button + 3) % 6,
        1 => (button + 4) % 6,
        2 => (button + 5) % 6,
        3 => button,
        4 => (button + 1) % 6,
        _ => (button + 2) % 6,
    }
}

fn six_seat_ev(output: &SolverOutput, button_index: usize, role: usize) -> Option<[f64; 169]> {
    output
        .hand_evs
        .get(six_max_seat(button_index, role))
        .copied()
}

fn freq_node6(ranges: &SixMaxRanges, actor: usize, mask: u32) -> [f64; 169] {
    ranges.freq[six_node(actor, mask)]
}

fn build_6max_tree(
    ranges: &SixMaxRanges,
    output: &SolverOutput,
    button_index: usize,
) -> Vec<TreeNode> {
    let ev = |role: usize| six_seat_ev(output, button_index, role);
    let f = |actor: usize, mask: u32| freq_node6(ranges, actor, mask);
    vec![
        make_node(
            "UTG push",
            Action::Raise,
            f(0, 0),
            ev(0),
            vec![
                make_node("HJ call (vs UTG)", Action::Call, f(1, 1), None, vec![]),
                make_node("CO call (vs UTG)", Action::Call, f(2, 1), None, vec![]),
                make_node("BTN call (vs UTG)", Action::Call, f(3, 1), None, vec![]),
                make_node("SB call (vs UTG)", Action::Call, f(4, 1), None, vec![]),
                make_node("BB call (vs UTG)", Action::Call, f(5, 1), None, vec![]),
            ],
        ),
        make_node(
            "HJ push (UTG fold)",
            Action::Raise,
            f(1, 0),
            ev(1),
            vec![
                make_node("CO call (vs HJ)", Action::Call, f(2, 2), None, vec![]),
                make_node("BTN call (vs HJ)", Action::Call, f(3, 2), None, vec![]),
                make_node("SB call (vs HJ)", Action::Call, f(4, 2), None, vec![]),
                make_node("BB call (vs HJ)", Action::Call, f(5, 2), None, vec![]),
            ],
        ),
        make_node(
            "CO push (folds)",
            Action::Raise,
            f(2, 0),
            ev(2),
            vec![
                make_node("BTN call (vs CO)", Action::Call, f(3, 4), None, vec![]),
                make_node("SB call (vs CO)", Action::Call, f(4, 4), None, vec![]),
                make_node("BB call (vs CO)", Action::Call, f(5, 4), None, vec![]),
            ],
        ),
        make_node(
            "BTN push (folds)",
            Action::Raise,
            f(3, 0),
            ev(3),
            vec![
                make_node("SB call (vs BTN)", Action::Call, f(4, 8), None, vec![]),
                make_node("BB call (vs BTN)", Action::Call, f(5, 8), None, vec![]),
            ],
        ),
        make_node(
            "SB push (folds)",
            Action::Raise,
            f(4, 0),
            ev(4),
            vec![make_node(
                "BB call (vs SB)",
                Action::Call,
                f(5, 16),
                ev(5),
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
