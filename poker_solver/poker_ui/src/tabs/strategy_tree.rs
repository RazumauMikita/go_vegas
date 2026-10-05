use std::collections::HashMap;

use egui::{Color32, FontId, Pos2, Rect, Sense, Ui};
use poker_core::{
    eight_node, five_node, nine_node, seven_node, six_node, EightMaxRanges, FiveMaxRanges,
    FourMaxRanges, NineMaxRanges, SevenMaxRanges, SixMaxRanges, SolverOutput, ThreeMaxHandEvs,
    ThreeMaxRanges,
};

use crate::widgets::{combo_share, range_notation};

const COL_ACTION_W: f32 = 108.0;
const COL_AMT_W: f32 = 64.0;
const COL_PLAYER_W: f32 = 52.0;
const ROW_H: f32 = 22.0;
const TREE_HEADER: Color32 = Color32::from_rgb(70, 141, 196);
const TREE_SELECTED: Color32 = Color32::from_rgb(196, 226, 248);
const TREE_HOVER: Color32 = Color32::from_rgb(232, 244, 252);
const TREE_BG: Color32 = Color32::WHITE;
const TREE_TEXT: Color32 = Color32::from_rgb(32, 32, 32);
const DOT_RAISE: Color32 = Color32::from_rgb(76, 175, 80);
const DOT_CALL: Color32 = Color32::from_rgb(196, 80, 70);
const DOT_CALL_NESTED: Color32 = Color32::from_rgb(66, 133, 244);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Raise,
    Call,
    CallSpecial,
}

#[derive(Clone)]
pub struct TreeNode {
    pub range_id: usize,
    pub label: String,
    pub action: Action,
    pub range_pct: f64,
    pub range: [f64; 169],
    pub ev_range: Option<[f64; 169]>,
    pub children: Vec<TreeNode>,
    pub expanded: bool,
}

pub type NodePath = Vec<usize>;

impl TreeNode {
    pub fn player(&self) -> &str {
        self.label.split_whitespace().next().unwrap_or("")
    }

    pub fn action_code(&self) -> &'static str {
        match self.action {
            Action::Raise => "R",
            Action::Call | Action::CallSpecial => "C",
        }
    }
}

pub fn build_strategy_tree(output: &SolverOutput, button_index: usize) -> Vec<TreeNode> {
    let mut tree = if let Some(ranges) = output.nine_max.as_ref() {
        build_9max_tree(ranges, output, button_index)
    } else if let Some(ranges) = output.eight_max.as_ref() {
        build_8max_tree(ranges, output, button_index)
    } else if let Some(ranges) = output.seven_max.as_ref() {
        build_7max_tree(ranges, output, button_index)
    } else if let Some(ranges) = output.six_max.as_ref() {
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
    };
    expand_one_level(&mut tree);
    tree
}

fn expand_one_level(tree: &mut [TreeNode]) {
    for node in tree {
        node.expanded = !node.children.is_empty();
        collapse_all(&mut node.children);
    }
}

fn collapse_all(tree: &mut [TreeNode]) {
    for node in tree {
        node.expanded = false;
        collapse_all(&mut node.children);
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

pub fn node_at_path_mut<'a>(tree: &'a mut [TreeNode], path: &[usize]) -> Option<&'a mut TreeNode> {
    if path.is_empty() {
        return None;
    }
    let mut current: &mut TreeNode = tree.get_mut(path[0])?;
    for &index in path.iter().skip(1) {
        current = current.children.get_mut(index)?;
    }
    Some(current)
}

pub fn draw_strategy_tree_header(ui: &mut Ui) {
    let width = ui.available_width();
    let cols = tree_columns(width);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, ROW_H), Sense::hover());
    ui.painter().rect_filled(rect, 0.0, TREE_HEADER);
    let font = FontId::proportional(13.0);
    let y = rect.center().y;
    paint_col_text(ui, rect.left() + 8.0, y, "Action", &font, Color32::WHITE, None);
    paint_col_text(
        ui,
        rect.left() + cols.action,
        y,
        "Amt [BB]",
        &font,
        Color32::WHITE,
        None,
    );
    paint_col_text(
        ui,
        rect.left() + cols.action + cols.amt,
        y,
        "Player",
        &font,
        Color32::WHITE,
        None,
    );
    paint_col_text(
        ui,
        rect.left() + cols.action + cols.amt + cols.player,
        y,
        "Range",
        &font,
        Color32::WHITE,
        None,
    );
}

pub fn draw_strategy_tree(
    ui: &mut Ui,
    tree: &mut [TreeNode],
    selected_path: &mut Option<NodePath>,
    editor_path: &mut Option<NodePath>,
    locked_ids: &HashMap<usize, [f64; 169]>,
    stacks_bb: &HashMap<String, f64>,
    path_prefix: &[usize],
    depth: usize,
    in_call_line: bool,
) {
    for (index, node) in tree.iter_mut().enumerate() {
        let current_path: NodePath = path_prefix
            .iter()
            .chain(std::iter::once(&index))
            .copied()
            .collect();
        let is_selected = selected_path.as_ref() == Some(&current_path);
        let is_locked = locked_ids.contains_key(&node.range_id);
        let has_children = !node.children.is_empty();
        let amount = stacks_bb.get(node.player()).copied().unwrap_or(0.0);
        let notation = range_notation(&node.range);
        let range_text = if notation.is_empty() {
            format!("{:.1}%", node.range_pct)
        } else {
            format!("{:.1}%, {notation}", node.range_pct)
        };

        let width = ui.available_width();
        let cols = tree_columns(width);
        let (rect, response) = ui.allocate_exact_size(egui::vec2(width, ROW_H), Sense::click());
        let hovered = response.hovered();
        let bg = if is_selected {
            TREE_SELECTED
        } else if hovered {
            TREE_HOVER
        } else {
            TREE_BG
        };
        ui.painter().rect_filled(rect, 0.0, bg);
        if is_locked {
            ui.painter().rect_stroke(
                rect,
                0.0,
                egui::Stroke::new(1.0_f32, Color32::from_rgb(255, 196, 40)),
            );
        }

        let indent = depth as f32 * 14.0;
        let action_left = rect.left() + 4.0 + indent;
        let mut expand_clicked = false;
        if has_children {
            let icon = if node.expanded { "▼" } else { "▶" };
            let icon_pos = Pos2::new(action_left + 8.0, rect.center().y);
            let icon_rect = Rect::from_center_size(icon_pos, egui::vec2(16.0, ROW_H));
            ui.painter().text(
                icon_pos,
                egui::Align2::CENTER_CENTER,
                icon,
                FontId::proportional(12.0),
                TREE_TEXT,
            );
            if response.clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    if icon_rect.contains(pos) {
                        node.expanded = !node.expanded;
                        expand_clicked = true;
                    }
                }
            }
        }

        let is_call = matches!(node.action, Action::Call | Action::CallSpecial);
        let dot_x = action_left + if has_children { 22.0 } else { 18.0 };
        let dot_color = if in_call_line {
            DOT_CALL_NESTED
        } else if is_call {
            DOT_CALL
        } else {
            DOT_RAISE
        };
        let in_call_line = in_call_line || is_call;
        ui.painter()
            .circle_filled(Pos2::new(dot_x, rect.center().y), 4.5, dot_color);
        ui.painter().text(
            Pos2::new(dot_x + 12.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            node.action_code(),
            FontId::proportional(13.0),
            TREE_TEXT,
        );

        let font = FontId::proportional(12.0);
        paint_col_text(
            ui,
            rect.left() + cols.action,
            rect.center().y,
            &format!("{amount:.2}"),
            &font,
            TREE_TEXT,
            None,
        );
        paint_col_text(
            ui,
            rect.left() + cols.action + cols.amt,
            rect.center().y,
            node.player(),
            &font,
            TREE_TEXT,
            None,
        );
        paint_col_text(
            ui,
            rect.left() + cols.action + cols.amt + cols.player,
            rect.center().y,
            &range_text,
            &font,
            TREE_TEXT,
            Some(rect.right() - 4.0),
        );

        if response.clicked() && !expand_clicked {
            *selected_path = Some(current_path.clone());
        }
        if response.double_clicked() {
            *editor_path = Some(current_path.clone());
        }

        if node.expanded && !node.children.is_empty() {
            draw_strategy_tree(
                ui,
                &mut node.children,
                selected_path,
                editor_path,
                locked_ids,
                stacks_bb,
                &current_path,
                depth + 1,
                in_call_line,
            );
        }
    }
}

fn tree_columns(width: f32) -> TreeColumns {
    let scale = (width / 420.0).clamp(0.62, 1.0);
    TreeColumns {
        action: COL_ACTION_W * scale,
        amt: COL_AMT_W * scale,
        player: COL_PLAYER_W * scale,
    }
}

struct TreeColumns {
    action: f32,
    amt: f32,
    player: f32,
}

fn paint_col_text(
    ui: &Ui,
    x: f32,
    y: f32,
    text: &str,
    font: &FontId,
    color: Color32,
    clip_right: Option<f32>,
) {
    if let Some(right) = clip_right {
        ui.painter()
            .with_clip_rect(Rect::from_min_max(
                Pos2::new(x, y - ROW_H * 0.5),
                Pos2::new(right, y + ROW_H * 0.5),
            ))
            .text(
                Pos2::new(x, y),
                egui::Align2::LEFT_CENTER,
                text,
                font.clone(),
                color,
            );
    } else {
        ui.painter().text(
            Pos2::new(x, y),
            egui::Align2::LEFT_CENTER,
            text,
            font.clone(),
            color,
        );
    }
}

fn make_node(
    range_id: usize,
    label: &str,
    action: Action,
    range: [f64; 169],
    ev_range: Option<[f64; 169]>,
    children: Vec<TreeNode>,
) -> TreeNode {
    TreeNode {
        range_id,
        label: label.to_string(),
        action,
        range_pct: combo_share(&range) * 100.0,
        range,
        ev_range,
        children,
        expanded: false,
    }
}

fn build_3max_tree(ranges: &ThreeMaxRanges, hand_evs: Option<&ThreeMaxHandEvs>) -> Vec<TreeNode> {
    let evs = hand_evs;
    vec![
        make_node(
            0,
            "BTN push",
            Action::Raise,
            ranges.btn_push,
            evs.map(|e| e.btn_push),
            vec![make_node(
                1,
                "SB call",
                Action::Call,
                ranges.sb_call_vs_btn,
                evs.map(|e| e.sb_call_vs_btn),
                vec![
                    make_node(
                        3,
                        "BB call (vs BTN+SB)",
                        Action::CallSpecial,
                        ranges.bb_call_vs_btn_and_sb,
                        evs.map(|e| e.bb_call_vs_btn_and_sb),
                        vec![],
                    ),
                    make_node(
                        2,
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
            4,
            "SB push (BTN fold)",
            Action::Raise,
            ranges.sb_push,
            evs.map(|e| e.sb_push),
            vec![make_node(
                5,
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
            0,
            "CO push",
            Action::Raise,
            ranges.utg_push,
            ev(0),
            vec![
                make_node(
                    1,
                    "BTN call (vs CO)",
                    Action::Call,
                    ranges.btn_call_vs_push,
                    None,
                    vec![
                        make_node(
                            3,
                            "SB call (vs CO+BTN)",
                            Action::Call,
                            ranges.sb_call_vs_utg_btn,
                            None,
                            vec![make_node(
                                7,
                                "BB call (4-way)",
                                Action::CallSpecial,
                                ranges.bb_call_4way,
                                ev(3),
                                vec![],
                            )],
                        ),
                        make_node(
                            8,
                            "BB call (vs CO+BTN)",
                            Action::Call,
                            ranges.bb_call_vs_utg_btn,
                            None,
                            vec![],
                        ),
                    ],
                ),
                make_node(
                    4,
                    "SB call (vs CO)",
                    Action::Call,
                    ranges.sb_call_vs_utg,
                    None,
                    vec![make_node(
                        9,
                        "BB call (vs CO+SB)",
                        Action::Call,
                        ranges.bb_call_vs_utg_sb,
                        None,
                        vec![],
                    )],
                ),
                make_node(
                    10,
                    "BB call (vs CO)",
                    Action::Call,
                    ranges.bb_call_vs_utg,
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            2,
            "BTN push (CO fold)",
            Action::Raise,
            ranges.btn_push,
            ev(1),
            vec![
                make_node(
                    5,
                    "SB call (vs BTN)",
                    Action::Call,
                    ranges.sb_call_vs_btn,
                    None,
                    vec![make_node(
                        11,
                        "BB call (vs BTN+SB)",
                        Action::CallSpecial,
                        ranges.bb_call_vs_btn_sb,
                        None,
                        vec![],
                    )],
                ),
                make_node(
                    12,
                    "BB call (vs BTN)",
                    Action::Call,
                    ranges.bb_call_vs_btn,
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            6,
            "SB push (CO+BTN fold)",
            Action::Raise,
            ranges.sb_push,
            ev(2),
            vec![make_node(
                13,
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
            five_node(0, 0),
            "HJ push",
            Action::Raise,
            f(0, 0),
            ev(0),
            vec![
                make_node(
                    five_node(1, 1),
                    "CO call (vs HJ)",
                    Action::Call,
                    f(1, 1),
                    None,
                    vec![
                        make_node(
                            five_node(2, 3),
                            "BTN call (vs HJ+CO)",
                            Action::Call,
                            f(2, 3),
                            None,
                            vec![
                                make_node(
                                    five_node(3, 7),
                                    "SB call (vs HJ+CO+BTN)",
                                    Action::Call,
                                    f(3, 7),
                                    None,
                                    vec![make_node(
                                        five_node(4, 15),
                                        "BB call (5-way)",
                                        Action::CallSpecial,
                                        f(4, 15),
                                        None,
                                        vec![],
                                    )],
                                ),
                                make_node(
                                    five_node(4, 7),
                                    "BB call (vs HJ+CO+BTN)",
                                    Action::Call,
                                    f(4, 7),
                                    None,
                                    vec![],
                                ),
                            ],
                        ),
                        make_node(
                            five_node(3, 3),
                            "SB call (vs HJ+CO)",
                            Action::Call,
                            f(3, 3),
                            None,
                            vec![make_node(
                                five_node(4, 11),
                                "BB call (vs HJ+CO+SB)",
                                Action::Call,
                                f(4, 11),
                                None,
                                vec![],
                            )],
                        ),
                        make_node(
                            five_node(4, 3),
                            "BB call (vs HJ+CO)",
                            Action::Call,
                            f(4, 3),
                            None,
                            vec![],
                        ),
                    ],
                ),
                make_node(
                    five_node(2, 1),
                    "BTN call (vs HJ)",
                    Action::Call,
                    f(2, 1),
                    None,
                    vec![
                        make_node(
                            five_node(3, 5),
                            "SB call (vs HJ+BTN)",
                            Action::Call,
                            f(3, 5),
                            None,
                            vec![make_node(
                                five_node(4, 13),
                                "BB call (vs HJ+BTN+SB)",
                                Action::Call,
                                f(4, 13),
                                None,
                                vec![],
                            )],
                        ),
                        make_node(
                            five_node(4, 5),
                            "BB call (vs HJ+BTN)",
                            Action::Call,
                            f(4, 5),
                            None,
                            vec![],
                        ),
                    ],
                ),
                make_node(
                    five_node(3, 1),
                    "SB call (vs HJ)",
                    Action::Call,
                    f(3, 1),
                    None,
                    vec![make_node(
                        five_node(4, 9),
                        "BB call (vs HJ+SB)",
                        Action::Call,
                        f(4, 9),
                        None,
                        vec![],
                    )],
                ),
                make_node(
                    five_node(4, 1),
                    "BB call (vs HJ)",
                    Action::Call,
                    f(4, 1),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            five_node(1, 0),
            "CO push (HJ fold)",
            Action::Raise,
            f(1, 0),
            ev(1),
            vec![
                make_node(
                    five_node(2, 2),
                    "BTN call (vs CO)",
                    Action::Call,
                    f(2, 2),
                    None,
                    vec![
                        make_node(
                            five_node(3, 6),
                            "SB call (vs CO+BTN)",
                            Action::Call,
                            f(3, 6),
                            None,
                            vec![make_node(
                                five_node(4, 14),
                                "BB call (vs CO+BTN+SB)",
                                Action::Call,
                                f(4, 14),
                                None,
                                vec![],
                            )],
                        ),
                        make_node(
                            five_node(4, 6),
                            "BB call (vs CO+BTN)",
                            Action::Call,
                            f(4, 6),
                            None,
                            vec![],
                        ),
                    ],
                ),
                make_node(
                    five_node(3, 2),
                    "SB call (vs CO)",
                    Action::Call,
                    f(3, 2),
                    None,
                    vec![make_node(
                        five_node(4, 10),
                        "BB call (vs CO+SB)",
                        Action::Call,
                        f(4, 10),
                        None,
                        vec![],
                    )],
                ),
                make_node(
                    five_node(4, 2),
                    "BB call (vs CO)",
                    Action::Call,
                    f(4, 2),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            five_node(2, 0),
            "BTN push (HJ+CO fold)",
            Action::Raise,
            f(2, 0),
            ev(2),
            vec![
                make_node(
                    five_node(3, 4),
                    "SB call (vs BTN)",
                    Action::Call,
                    f(3, 4),
                    None,
                    vec![make_node(
                        five_node(4, 12),
                        "BB call (vs BTN+SB)",
                        Action::CallSpecial,
                        f(4, 12),
                        None,
                        vec![],
                    )],
                ),
                make_node(
                    five_node(4, 4),
                    "BB call (vs BTN)",
                    Action::Call,
                    f(4, 4),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            five_node(3, 0),
            "SB push (folds)",
            Action::Raise,
            f(3, 0),
            ev(3),
            vec![make_node(
                five_node(4, 8),
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
            six_node(0, 0),
            "UTG push",
            Action::Raise,
            f(0, 0),
            ev(0),
            vec![
                make_node(
                    six_node(1, 1),
                    "HJ call (vs UTG)",
                    Action::Call,
                    f(1, 1),
                    None,
                    vec![],
                ),
                make_node(
                    six_node(2, 1),
                    "CO call (vs UTG)",
                    Action::Call,
                    f(2, 1),
                    None,
                    vec![],
                ),
                make_node(
                    six_node(3, 1),
                    "BTN call (vs UTG)",
                    Action::Call,
                    f(3, 1),
                    None,
                    vec![],
                ),
                make_node(
                    six_node(4, 1),
                    "SB call (vs UTG)",
                    Action::Call,
                    f(4, 1),
                    None,
                    vec![],
                ),
                make_node(
                    six_node(5, 1),
                    "BB call (vs UTG)",
                    Action::Call,
                    f(5, 1),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            six_node(1, 0),
            "HJ push (UTG fold)",
            Action::Raise,
            f(1, 0),
            ev(1),
            vec![
                make_node(
                    six_node(2, 2),
                    "CO call (vs HJ)",
                    Action::Call,
                    f(2, 2),
                    None,
                    vec![],
                ),
                make_node(
                    six_node(3, 2),
                    "BTN call (vs HJ)",
                    Action::Call,
                    f(3, 2),
                    None,
                    vec![],
                ),
                make_node(
                    six_node(4, 2),
                    "SB call (vs HJ)",
                    Action::Call,
                    f(4, 2),
                    None,
                    vec![],
                ),
                make_node(
                    six_node(5, 2),
                    "BB call (vs HJ)",
                    Action::Call,
                    f(5, 2),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            six_node(2, 0),
            "CO push (folds)",
            Action::Raise,
            f(2, 0),
            ev(2),
            vec![
                make_node(
                    six_node(3, 4),
                    "BTN call (vs CO)",
                    Action::Call,
                    f(3, 4),
                    None,
                    vec![],
                ),
                make_node(
                    six_node(4, 4),
                    "SB call (vs CO)",
                    Action::Call,
                    f(4, 4),
                    None,
                    vec![],
                ),
                make_node(
                    six_node(5, 4),
                    "BB call (vs CO)",
                    Action::Call,
                    f(5, 4),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            six_node(3, 0),
            "BTN push (folds)",
            Action::Raise,
            f(3, 0),
            ev(3),
            vec![
                make_node(
                    six_node(4, 8),
                    "SB call (vs BTN)",
                    Action::Call,
                    f(4, 8),
                    None,
                    vec![],
                ),
                make_node(
                    six_node(5, 8),
                    "BB call (vs BTN)",
                    Action::Call,
                    f(5, 8),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            six_node(4, 0),
            "SB push (folds)",
            Action::Raise,
            f(4, 0),
            ev(4),
            vec![make_node(
                six_node(5, 16),
                "BB call (vs SB)",
                Action::Call,
                f(5, 16),
                ev(5),
                vec![],
            )],
        ),
    ]
}

fn seven_max_seat(button_index: usize, role: usize) -> usize {
    let button = button_index % 7;
    match role {
        0 => (button + 3) % 7,
        1 => (button + 4) % 7,
        2 => (button + 5) % 7,
        3 => (button + 6) % 7,
        4 => button,
        5 => (button + 1) % 7,
        _ => (button + 2) % 7,
    }
}

fn seven_seat_ev(output: &SolverOutput, button_index: usize, role: usize) -> Option<[f64; 169]> {
    output
        .hand_evs
        .get(seven_max_seat(button_index, role))
        .copied()
}

fn freq_node7(ranges: &SevenMaxRanges, actor: usize, mask: u32) -> [f64; 169] {
    ranges.freq[seven_node(actor, mask)]
}

fn build_7max_tree(
    ranges: &SevenMaxRanges,
    output: &SolverOutput,
    button_index: usize,
) -> Vec<TreeNode> {
    let ev = |role: usize| seven_seat_ev(output, button_index, role);
    let f = |actor: usize, mask: u32| freq_node7(ranges, actor, mask);
    vec![
        make_node(
            seven_node(0, 0),
            "UTG push",
            Action::Raise,
            f(0, 0),
            ev(0),
            vec![
                make_node(
                    seven_node(1, 1),
                    "MP call (vs UTG)",
                    Action::Call,
                    f(1, 1),
                    None,
                    vec![],
                ),
                make_node(
                    seven_node(2, 1),
                    "HJ call (vs UTG)",
                    Action::Call,
                    f(2, 1),
                    None,
                    vec![],
                ),
                make_node(
                    seven_node(3, 1),
                    "CO call (vs UTG)",
                    Action::Call,
                    f(3, 1),
                    None,
                    vec![],
                ),
                make_node(
                    seven_node(4, 1),
                    "BTN call (vs UTG)",
                    Action::Call,
                    f(4, 1),
                    None,
                    vec![],
                ),
                make_node(
                    seven_node(5, 1),
                    "SB call (vs UTG)",
                    Action::Call,
                    f(5, 1),
                    None,
                    vec![],
                ),
                make_node(
                    seven_node(6, 1),
                    "BB call (vs UTG)",
                    Action::Call,
                    f(6, 1),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            seven_node(1, 0),
            "MP push (UTG fold)",
            Action::Raise,
            f(1, 0),
            ev(1),
            vec![
                make_node(
                    seven_node(2, 2),
                    "HJ call (vs MP)",
                    Action::Call,
                    f(2, 2),
                    None,
                    vec![],
                ),
                make_node(
                    seven_node(3, 2),
                    "CO call (vs MP)",
                    Action::Call,
                    f(3, 2),
                    None,
                    vec![],
                ),
                make_node(
                    seven_node(4, 2),
                    "BTN call (vs MP)",
                    Action::Call,
                    f(4, 2),
                    None,
                    vec![],
                ),
                make_node(
                    seven_node(5, 2),
                    "SB call (vs MP)",
                    Action::Call,
                    f(5, 2),
                    None,
                    vec![],
                ),
                make_node(
                    seven_node(6, 2),
                    "BB call (vs MP)",
                    Action::Call,
                    f(6, 2),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            seven_node(2, 0),
            "HJ push (folds)",
            Action::Raise,
            f(2, 0),
            ev(2),
            vec![
                make_node(
                    seven_node(3, 4),
                    "CO call (vs HJ)",
                    Action::Call,
                    f(3, 4),
                    None,
                    vec![],
                ),
                make_node(
                    seven_node(4, 4),
                    "BTN call (vs HJ)",
                    Action::Call,
                    f(4, 4),
                    None,
                    vec![],
                ),
                make_node(
                    seven_node(5, 4),
                    "SB call (vs HJ)",
                    Action::Call,
                    f(5, 4),
                    None,
                    vec![],
                ),
                make_node(
                    seven_node(6, 4),
                    "BB call (vs HJ)",
                    Action::Call,
                    f(6, 4),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            seven_node(3, 0),
            "CO push (folds)",
            Action::Raise,
            f(3, 0),
            ev(3),
            vec![
                make_node(
                    seven_node(4, 8),
                    "BTN call (vs CO)",
                    Action::Call,
                    f(4, 8),
                    None,
                    vec![],
                ),
                make_node(
                    seven_node(5, 8),
                    "SB call (vs CO)",
                    Action::Call,
                    f(5, 8),
                    None,
                    vec![],
                ),
                make_node(
                    seven_node(6, 8),
                    "BB call (vs CO)",
                    Action::Call,
                    f(6, 8),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            seven_node(4, 0),
            "BTN push (folds)",
            Action::Raise,
            f(4, 0),
            ev(4),
            vec![
                make_node(
                    seven_node(5, 16),
                    "SB call (vs BTN)",
                    Action::Call,
                    f(5, 16),
                    None,
                    vec![],
                ),
                make_node(
                    seven_node(6, 16),
                    "BB call (vs BTN)",
                    Action::Call,
                    f(6, 16),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            seven_node(5, 0),
            "SB push (folds)",
            Action::Raise,
            f(5, 0),
            ev(5),
            vec![make_node(
                seven_node(6, 32),
                "BB call (vs SB)",
                Action::Call,
                f(6, 32),
                ev(6),
                vec![],
            )],
        ),
    ]
}

fn eight_max_seat(button_index: usize, role: usize) -> usize {
    let button = button_index % 8;
    match role {
        0 => (button + 3) % 8,
        1 => (button + 4) % 8,
        2 => (button + 5) % 8,
        3 => (button + 6) % 8,
        4 => (button + 7) % 8,
        5 => button,
        6 => (button + 1) % 8,
        _ => (button + 2) % 8,
    }
}

fn eight_seat_ev(output: &SolverOutput, button_index: usize, role: usize) -> Option<[f64; 169]> {
    output
        .hand_evs
        .get(eight_max_seat(button_index, role))
        .copied()
}

fn freq_node8(ranges: &EightMaxRanges, actor: usize, mask: u32) -> [f64; 169] {
    ranges.freq[eight_node(actor, mask)]
}

fn build_8max_tree(
    ranges: &EightMaxRanges,
    output: &SolverOutput,
    button_index: usize,
) -> Vec<TreeNode> {
    let ev = |role: usize| eight_seat_ev(output, button_index, role);
    let f = |actor: usize, mask: u32| freq_node8(ranges, actor, mask);
    vec![
        make_node(
            eight_node(0, 0),
            "UTG push",
            Action::Raise,
            f(0, 0),
            ev(0),
            vec![
                make_node(
                    eight_node(1, 1),
                    "EP call (vs UTG)",
                    Action::Call,
                    f(1, 1),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(2, 1),
                    "MP call (vs UTG)",
                    Action::Call,
                    f(2, 1),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(3, 1),
                    "HJ call (vs UTG)",
                    Action::Call,
                    f(3, 1),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(4, 1),
                    "CO call (vs UTG)",
                    Action::Call,
                    f(4, 1),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(5, 1),
                    "BTN call (vs UTG)",
                    Action::Call,
                    f(5, 1),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(6, 1),
                    "SB call (vs UTG)",
                    Action::Call,
                    f(6, 1),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(7, 1),
                    "BB call (vs UTG)",
                    Action::Call,
                    f(7, 1),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            eight_node(1, 0),
            "EP push (UTG fold)",
            Action::Raise,
            f(1, 0),
            ev(1),
            vec![
                make_node(
                    eight_node(2, 2),
                    "MP call (vs EP)",
                    Action::Call,
                    f(2, 2),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(3, 2),
                    "HJ call (vs EP)",
                    Action::Call,
                    f(3, 2),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(4, 2),
                    "CO call (vs EP)",
                    Action::Call,
                    f(4, 2),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(5, 2),
                    "BTN call (vs EP)",
                    Action::Call,
                    f(5, 2),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(6, 2),
                    "SB call (vs EP)",
                    Action::Call,
                    f(6, 2),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(7, 2),
                    "BB call (vs EP)",
                    Action::Call,
                    f(7, 2),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            eight_node(2, 0),
            "MP push (folds)",
            Action::Raise,
            f(2, 0),
            ev(2),
            vec![
                make_node(
                    eight_node(3, 4),
                    "HJ call (vs MP)",
                    Action::Call,
                    f(3, 4),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(4, 4),
                    "CO call (vs MP)",
                    Action::Call,
                    f(4, 4),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(5, 4),
                    "BTN call (vs MP)",
                    Action::Call,
                    f(5, 4),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(6, 4),
                    "SB call (vs MP)",
                    Action::Call,
                    f(6, 4),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(7, 4),
                    "BB call (vs MP)",
                    Action::Call,
                    f(7, 4),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            eight_node(3, 0),
            "HJ push (folds)",
            Action::Raise,
            f(3, 0),
            ev(3),
            vec![
                make_node(
                    eight_node(4, 8),
                    "CO call (vs HJ)",
                    Action::Call,
                    f(4, 8),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(5, 8),
                    "BTN call (vs HJ)",
                    Action::Call,
                    f(5, 8),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(6, 8),
                    "SB call (vs HJ)",
                    Action::Call,
                    f(6, 8),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(7, 8),
                    "BB call (vs HJ)",
                    Action::Call,
                    f(7, 8),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            eight_node(4, 0),
            "CO push (folds)",
            Action::Raise,
            f(4, 0),
            ev(4),
            vec![
                make_node(
                    eight_node(5, 16),
                    "BTN call (vs CO)",
                    Action::Call,
                    f(5, 16),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(6, 16),
                    "SB call (vs CO)",
                    Action::Call,
                    f(6, 16),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(7, 16),
                    "BB call (vs CO)",
                    Action::Call,
                    f(7, 16),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            eight_node(5, 0),
            "BTN push (folds)",
            Action::Raise,
            f(5, 0),
            ev(5),
            vec![
                make_node(
                    eight_node(6, 32),
                    "SB call (vs BTN)",
                    Action::Call,
                    f(6, 32),
                    None,
                    vec![],
                ),
                make_node(
                    eight_node(7, 32),
                    "BB call (vs BTN)",
                    Action::Call,
                    f(7, 32),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            eight_node(6, 0),
            "SB push (folds)",
            Action::Raise,
            f(6, 0),
            ev(6),
            vec![make_node(
                eight_node(7, 64),
                "BB call (vs SB)",
                Action::Call,
                f(7, 64),
                ev(7),
                vec![],
            )],
        ),
    ]
}

fn nine_max_seat(button_index: usize, role: usize) -> usize {
    let button = button_index % 9;
    match role {
        0 => (button + 3) % 9,
        1 => (button + 4) % 9,
        2 => (button + 5) % 9,
        3 => (button + 6) % 9,
        4 => (button + 7) % 9,
        5 => (button + 8) % 9,
        6 => button,
        7 => (button + 1) % 9,
        _ => (button + 2) % 9,
    }
}

fn nine_seat_ev(output: &SolverOutput, button_index: usize, role: usize) -> Option<[f64; 169]> {
    output
        .hand_evs
        .get(nine_max_seat(button_index, role))
        .copied()
}

fn freq_node9(ranges: &NineMaxRanges, actor: usize, mask: u32) -> [f64; 169] {
    ranges.freq[nine_node(actor, mask)]
}

fn build_9max_tree(
    ranges: &NineMaxRanges,
    output: &SolverOutput,
    button_index: usize,
) -> Vec<TreeNode> {
    let ev = |role: usize| nine_seat_ev(output, button_index, role);
    let f = |actor: usize, mask: u32| freq_node9(ranges, actor, mask);
    vec![
        make_node(
            nine_node(0, 0),
            "UTG push",
            Action::Raise,
            f(0, 0),
            ev(0),
            vec![
                make_node(
                    nine_node(1, 1),
                    "EP call (vs UTG)",
                    Action::Call,
                    f(1, 1),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(2, 1),
                    "MP1 call (vs UTG)",
                    Action::Call,
                    f(2, 1),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(3, 1),
                    "MP2 call (vs UTG)",
                    Action::Call,
                    f(3, 1),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(4, 1),
                    "HJ call (vs UTG)",
                    Action::Call,
                    f(4, 1),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(5, 1),
                    "CO call (vs UTG)",
                    Action::Call,
                    f(5, 1),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(6, 1),
                    "BTN call (vs UTG)",
                    Action::Call,
                    f(6, 1),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(7, 1),
                    "SB call (vs UTG)",
                    Action::Call,
                    f(7, 1),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(8, 1),
                    "BB call (vs UTG)",
                    Action::Call,
                    f(8, 1),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            nine_node(1, 0),
            "EP push (UTG fold)",
            Action::Raise,
            f(1, 0),
            ev(1),
            vec![
                make_node(
                    nine_node(2, 2),
                    "MP1 call (vs EP)",
                    Action::Call,
                    f(2, 2),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(3, 2),
                    "MP2 call (vs EP)",
                    Action::Call,
                    f(3, 2),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(4, 2),
                    "HJ call (vs EP)",
                    Action::Call,
                    f(4, 2),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(5, 2),
                    "CO call (vs EP)",
                    Action::Call,
                    f(5, 2),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(6, 2),
                    "BTN call (vs EP)",
                    Action::Call,
                    f(6, 2),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(7, 2),
                    "SB call (vs EP)",
                    Action::Call,
                    f(7, 2),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(8, 2),
                    "BB call (vs EP)",
                    Action::Call,
                    f(8, 2),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            nine_node(2, 0),
            "MP1 push (folds)",
            Action::Raise,
            f(2, 0),
            ev(2),
            vec![
                make_node(
                    nine_node(3, 4),
                    "MP2 call (vs MP1)",
                    Action::Call,
                    f(3, 4),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(4, 4),
                    "HJ call (vs MP1)",
                    Action::Call,
                    f(4, 4),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(5, 4),
                    "CO call (vs MP1)",
                    Action::Call,
                    f(5, 4),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(6, 4),
                    "BTN call (vs MP1)",
                    Action::Call,
                    f(6, 4),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(7, 4),
                    "SB call (vs MP1)",
                    Action::Call,
                    f(7, 4),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(8, 4),
                    "BB call (vs MP1)",
                    Action::Call,
                    f(8, 4),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            nine_node(3, 0),
            "MP2 push (folds)",
            Action::Raise,
            f(3, 0),
            ev(3),
            vec![
                make_node(
                    nine_node(4, 8),
                    "HJ call (vs MP2)",
                    Action::Call,
                    f(4, 8),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(5, 8),
                    "CO call (vs MP2)",
                    Action::Call,
                    f(5, 8),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(6, 8),
                    "BTN call (vs MP2)",
                    Action::Call,
                    f(6, 8),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(7, 8),
                    "SB call (vs MP2)",
                    Action::Call,
                    f(7, 8),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(8, 8),
                    "BB call (vs MP2)",
                    Action::Call,
                    f(8, 8),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            nine_node(4, 0),
            "HJ push (folds)",
            Action::Raise,
            f(4, 0),
            ev(4),
            vec![
                make_node(
                    nine_node(5, 16),
                    "CO call (vs HJ)",
                    Action::Call,
                    f(5, 16),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(6, 16),
                    "BTN call (vs HJ)",
                    Action::Call,
                    f(6, 16),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(7, 16),
                    "SB call (vs HJ)",
                    Action::Call,
                    f(7, 16),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(8, 16),
                    "BB call (vs HJ)",
                    Action::Call,
                    f(8, 16),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            nine_node(5, 0),
            "CO push (folds)",
            Action::Raise,
            f(5, 0),
            ev(5),
            vec![
                make_node(
                    nine_node(6, 32),
                    "BTN call (vs CO)",
                    Action::Call,
                    f(6, 32),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(7, 32),
                    "SB call (vs CO)",
                    Action::Call,
                    f(7, 32),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(8, 32),
                    "BB call (vs CO)",
                    Action::Call,
                    f(8, 32),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            nine_node(6, 0),
            "BTN push (folds)",
            Action::Raise,
            f(6, 0),
            ev(6),
            vec![
                make_node(
                    nine_node(7, 64),
                    "SB call (vs BTN)",
                    Action::Call,
                    f(7, 64),
                    None,
                    vec![],
                ),
                make_node(
                    nine_node(8, 64),
                    "BB call (vs BTN)",
                    Action::Call,
                    f(8, 64),
                    None,
                    vec![],
                ),
            ],
        ),
        make_node(
            nine_node(7, 0),
            "SB push (folds)",
            Action::Raise,
            f(7, 0),
            ev(7),
            vec![make_node(
                nine_node(8, 128),
                "BB call (vs SB)",
                Action::Call,
                f(8, 128),
                ev(8),
                vec![],
            )],
        ),
    ]
}

fn build_hu_tree(output: &SolverOutput) -> Vec<TreeNode> {
    vec![make_node(
        0,
        "SB push",
        Action::Raise,
        output.push_ranges[0],
        output.hand_evs.get(0).copied(),
        vec![make_node(
            1,
            "BB call",
            Action::Call,
            output.call_ranges[1],
            output.hand_evs.get(1).copied(),
            vec![],
        )],
    )]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_one_level_keeps_deeper_nodes_collapsed() {
        let mut tree = vec![make_node(
            0,
            "BTN push",
            Action::Raise,
            [0.0; 169],
            None,
            vec![make_node(
                1,
                "SB call",
                Action::Call,
                [0.0; 169],
                None,
                vec![make_node(
                    2,
                    "BB call",
                    Action::Call,
                    [0.0; 169],
                    None,
                    vec![],
                )],
            )],
        )];
        expand_one_level(&mut tree);
        assert!(tree[0].expanded);
        assert!(!tree[0].children[0].expanded);
        assert!(!tree[0].children[0].children[0].expanded);
    }
}

