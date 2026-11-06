use std::collections::HashMap;

use egui::{Color32, FontId, Pos2, Rect, Sense, Ui};
use poker_core::{
    eight_node, five_node, nine_node, seven_node, six_node, table_position_label, FourMaxRanges,
    SolverOutput, ThreeMaxHandEvs, ThreeMaxRanges,
};

use crate::widgets::{combo_share, range_notation};

const COL_AMT_W: f32 = 52.0;
const COL_PLAYER_W: f32 = 44.0;
const COL_ACTION_MIN: f32 = 92.0;
const COL_ACTION_MAX: f32 = 148.0;
const COL_RANGE_MIN: f32 = 72.0;
const INDENT_W: f32 = 16.0;
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
    ThreeBet,
    FourBet,
    Call,
    CallSpecial,
}

impl Action {
    pub fn is_raise(self) -> bool {
        matches!(self, Self::Raise | Self::ThreeBet | Self::FourBet)
    }

    pub fn is_call(self) -> bool {
        matches!(self, Self::Call | Self::CallSpecial)
    }
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
    pub amount_bb: f64,
}

pub type NodePath = Vec<usize>;

impl TreeNode {
    pub fn player(&self) -> &str {
        self.label.split_whitespace().next().unwrap_or("")
    }

    pub fn action_code(&self) -> &'static str {
        match self.action {
            Action::Raise => "R",
            Action::ThreeBet => "3B",
            Action::FourBet => "4B",
            Action::Call | Action::CallSpecial => "C",
        }
    }
}

pub fn build_strategy_tree(
    output: &SolverOutput,
    button_index: usize,
    stacks: &[f64],
    big_blind: f64,
) -> Vec<TreeNode> {
    let mut tree = if let Some(ranges) = output.nine_max.as_ref() {
        let mut tree = build_mask_tree(
            9,
            stacks,
            big_blind,
            |a, m| ranges.freq[nine_node(a, m)],
            nine_node,
        );
        attach_node_evs(&mut tree, &ranges.ev);
        tree
    } else if let Some(ranges) = output.eight_max.as_ref() {
        let mut tree = build_mask_tree(
            8,
            stacks,
            big_blind,
            |a, m| ranges.freq[eight_node(a, m)],
            eight_node,
        );
        attach_node_evs(&mut tree, &ranges.ev);
        tree
    } else if let Some(ranges) = output.seven_max.as_ref() {
        let mut tree = build_mask_tree(
            7,
            stacks,
            big_blind,
            |a, m| ranges.freq[seven_node(a, m)],
            seven_node,
        );
        attach_node_evs(&mut tree, &ranges.ev);
        tree
    } else if let Some(ranges) = output.six_max.as_ref() {
        let mut tree = build_mask_tree(
            6,
            stacks,
            big_blind,
            |a, m| ranges.freq[six_node(a, m)],
            six_node,
        );
        attach_node_evs(&mut tree, &ranges.ev);
        tree
    } else if let Some(ranges) = output.five_max.as_ref() {
        let mut tree = build_mask_tree(
            5,
            stacks,
            big_blind,
            |a, m| ranges.freq[five_node(a, m)],
            five_node,
        );
        attach_node_evs(&mut tree, &ranges.ev);
        tree
    } else if let Some(ranges) = output.four_max.as_ref() {
        let mut tree = build_4max_tree(ranges, output, button_index);
        attach_node_evs(&mut tree, &ranges.ev);
        tree
    } else if let Some(ranges) = output.three_max.as_ref() {
        let hand_evs = output.three_max_hand_evs.as_ref();
        build_3max_tree(ranges, hand_evs)
    } else {
        build_hu_tree(output)
    };
    expand_one_level(&mut tree);
    apply_short_stack_actions(&mut tree, stacks, big_blind);
    tree
}

fn min_raise_chips(big_blind: f64) -> f64 {
    2.0 * big_blind
}

fn can_min_raise(stack: f64, big_blind: f64) -> bool {
    stack + 1e-9 >= min_raise_chips(big_blind)
}

fn completes_only(stack: f64, big_blind: f64) -> bool {
    stack <= big_blind + 1e-9
}

fn apply_short_stack_actions(tree: &mut [TreeNode], stacks: &[f64], big_blind: f64) {
    if stacks.is_empty() || big_blind <= 0.0 {
        return;
    }
    let n = stacks.len();
    let mut by_name = HashMap::new();
    for (index, stack) in stacks.iter().enumerate() {
        by_name.insert(table_position_label(n, index), *stack);
    }
    relabel_unopened(tree, &by_name, big_blind, true);
}

fn relabel_unopened(
    tree: &mut [TreeNode],
    stacks: &HashMap<&'static str, f64>,
    big_blind: f64,
    unopened: bool,
) {
    for node in tree {
        let stack = stacks.get(node.player()).copied().unwrap_or(f64::INFINITY);
        if unopened {
            node.action = if can_min_raise(stack, big_blind) {
                Action::Raise
            } else {
                Action::Call
            };
        }
        let child_unopened = unopened && completes_only(stack, big_blind);
        relabel_unopened(&mut node.children, stacks, big_blind, child_unopened);
    }
}

fn attach_node_evs(tree: &mut [TreeNode], evs: &[[f64; 169]]) {
    for node in tree {
        if let Some(ev) = evs.get(node.range_id).copied() {
            node.ev_range = Some(ev);
        }
        attach_node_evs(&mut node.children, evs);
    }
}

fn expand_one_level(tree: &mut [TreeNode]) {
    collapse_all(tree);
    if let Some(node) = tree.first_mut() {
        node.expanded = !node.children.is_empty();
    }
}

fn collapse_all(tree: &mut [TreeNode]) {
    for node in tree {
        node.expanded = false;
        collapse_all(&mut node.children);
    }
}

pub fn expand_path(tree: &mut [TreeNode], path: &[usize]) {
    if path.is_empty() {
        return;
    }
    collapse_all(tree);
    let mut nodes = tree;
    for (depth, &index) in path.iter().enumerate() {
        let Some(node) = nodes.get_mut(index) else {
            return;
        };
        if depth + 1 < path.len() || path.len() == 1 {
            node.expanded = !node.children.is_empty();
        }
        nodes = node.children.as_mut_slice();
    }
}

pub fn display_player(name: &str) -> &str {
    if name == "BTN" {
        "BU"
    } else {
        name
    }
}

pub fn format_amount_bb(amount: f64) -> String {
    if amount >= 10.0 - 1e-9 {
        format!("{amount:.1}")
    } else {
        format!("{amount:.2}")
    }
}

pub fn display_vs(vs: &str) -> String {
    vs.replace("BTN", "BU")
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

pub fn find_hero_decision_path(
    tree: &[TreeNode],
    actions_before: &[(String, bool)],
    hero_label: &str,
) -> Option<NodePath> {
    walk_spot(tree, Vec::new(), 0, actions_before, hero_label)
}

fn walk_spot(
    siblings: &[TreeNode],
    path: NodePath,
    skip: usize,
    remaining: &[(String, bool)],
    hero: &str,
) -> Option<NodePath> {
    if skip > siblings.len() {
        return None;
    }
    if remaining.is_empty() {
        let view = &siblings[skip..];
        let j = view.iter().position(|node| node.player() == hero)?;
        let mut next = path;
        next.push(skip + j);
        return Some(next);
    }
    let (pos, is_fold) = &remaining[0];
    let view = &siblings[skip..];
    let Some(j) = view.iter().position(|node| node.player() == pos.as_str()) else {
        return walk_spot(siblings, path, skip, &remaining[1..], hero);
    };
    let abs = skip + j;
    if *is_fold {
        let mut next_skip = abs + 1;
        while next_skip < siblings.len() && node_depends_on_player(&siblings[next_skip], pos) {
            next_skip += 1;
        }
        walk_spot(siblings, path, next_skip, &remaining[1..], hero)
    } else if siblings[abs].children.is_empty() {
        None
    } else {
        let mut next = path;
        next.push(abs);
        walk_spot(&siblings[abs].children, next, 0, &remaining[1..], hero)
    }
}

fn node_depends_on_player(node: &TreeNode, player: &str) -> bool {
    if node.player() == player {
        return true;
    }
    let vs = node
        .label
        .strip_prefix(node.player())
        .unwrap_or("")
        .trim()
        .strip_prefix("vs ")
        .unwrap_or("");
    vs.split(", ").any(|name| name == player)
}

pub fn draw_strategy_tree_header(ui: &mut Ui) {
    let width = ui.available_width();
    let cols = tree_columns(width);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, ROW_H), Sense::hover());
    ui.painter().rect_filled(rect, 0.0, TREE_HEADER);
    let font = FontId::proportional(13.0);
    let y = rect.center().y;
    paint_col(
        ui,
        cols.action_rect(rect),
        rect.left() + 8.0,
        y,
        "Action",
        &font,
        Color32::WHITE,
    );
    paint_col(
        ui,
        cols.amt_rect(rect),
        cols.amt_x(rect),
        y,
        "Amt [BB]",
        &font,
        Color32::WHITE,
    );
    paint_col(
        ui,
        cols.player_rect(rect),
        cols.player_x(rect),
        y,
        "Player",
        &font,
        Color32::WHITE,
    );
    paint_col(
        ui,
        cols.range_rect(rect),
        cols.range_x(rect),
        y,
        "Range",
        &font,
        Color32::WHITE,
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
        let amount = if node.amount_bb > 0.0 {
            node.amount_bb
        } else {
            stacks_bb.get(node.player()).copied().unwrap_or(0.0)
        };
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

        let action_rect = cols.action_rect(rect);
        let indent = depth as f32 * INDENT_W;
        let action_left = rect.left() + 4.0 + indent;
        let mut expand_clicked = false;
        if has_children {
            let icon = if node.expanded { "▼" } else { "▶" };
            let icon_pos = Pos2::new(action_left + 8.0, rect.center().y);
            let icon_rect = Rect::from_center_size(icon_pos, egui::vec2(16.0, ROW_H));
            ui.painter().with_clip_rect(action_rect).text(
                icon_pos,
                egui::Align2::CENTER_CENTER,
                icon,
                FontId::proportional(12.0),
                TREE_TEXT,
            );
            if response.clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    if icon_rect.contains(pos) && action_rect.contains(pos) {
                        node.expanded = !node.expanded;
                        expand_clicked = true;
                    }
                }
            }
        }

        let is_call = node.action.is_call();
        let is_nested_raise = matches!(node.action, Action::FourBet);
        let dot_x = action_left + if has_children { 22.0 } else { 18.0 };
        let dot_color = if in_call_line || is_nested_raise {
            DOT_CALL_NESTED
        } else if is_call || matches!(node.action, Action::ThreeBet) {
            DOT_CALL
        } else {
            DOT_RAISE
        };
        let child_call_line =
            in_call_line || matches!(node.action, Action::ThreeBet | Action::FourBet);
        ui.painter().with_clip_rect(action_rect).circle_filled(
            Pos2::new(dot_x, rect.center().y),
            4.5,
            dot_color,
        );
        ui.painter().with_clip_rect(action_rect).text(
            Pos2::new(dot_x + 12.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            node.action_code(),
            FontId::proportional(13.0),
            TREE_TEXT,
        );

        let font = FontId::proportional(12.0);
        paint_col(
            ui,
            cols.amt_rect(rect),
            cols.amt_x(rect),
            rect.center().y,
            &format_amount_bb(amount),
            &font,
            TREE_TEXT,
        );
        paint_col(
            ui,
            cols.player_rect(rect),
            cols.player_x(rect),
            rect.center().y,
            display_player(node.player()),
            &font,
            TREE_TEXT,
        );
        paint_col(
            ui,
            cols.range_rect(rect),
            cols.range_x(rect),
            rect.center().y,
            &range_text,
            &font,
            TREE_TEXT,
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
                child_call_line,
            );
        }
    }
}

fn tree_columns(width: f32) -> TreeColumns {
    let amt = COL_AMT_W;
    let player = COL_PLAYER_W;
    let action = (width - amt - player - COL_RANGE_MIN).clamp(COL_ACTION_MIN, COL_ACTION_MAX);
    TreeColumns {
        action,
        amt,
        player,
    }
}

struct TreeColumns {
    action: f32,
    amt: f32,
    player: f32,
}

impl TreeColumns {
    fn amt_x(&self, rect: Rect) -> f32 {
        rect.left() + self.action
    }

    fn player_x(&self, rect: Rect) -> f32 {
        self.amt_x(rect) + self.amt
    }

    fn range_x(&self, rect: Rect) -> f32 {
        self.player_x(rect) + self.player
    }

    fn action_rect(&self, rect: Rect) -> Rect {
        Rect::from_min_max(
            Pos2::new(rect.left(), rect.top()),
            Pos2::new(self.amt_x(rect), rect.bottom()),
        )
    }

    fn amt_rect(&self, rect: Rect) -> Rect {
        Rect::from_min_max(
            Pos2::new(self.amt_x(rect), rect.top()),
            Pos2::new(self.player_x(rect), rect.bottom()),
        )
    }

    fn player_rect(&self, rect: Rect) -> Rect {
        Rect::from_min_max(
            Pos2::new(self.player_x(rect), rect.top()),
            Pos2::new(self.range_x(rect), rect.bottom()),
        )
    }

    fn range_rect(&self, rect: Rect) -> Rect {
        Rect::from_min_max(
            Pos2::new(self.range_x(rect), rect.top()),
            Pos2::new(rect.right(), rect.bottom()),
        )
    }
}

fn paint_col(ui: &Ui, clip: Rect, x: f32, y: f32, text: &str, font: &FontId, color: Color32) {
    ui.painter().with_clip_rect(clip).text(
        Pos2::new(x + 2.0, y),
        egui::Align2::LEFT_CENTER,
        text,
        font.clone(),
        color,
    );
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
        amount_bb: 0.0,
    }
}

fn build_mask_tree<F, N>(
    n: usize,
    stacks: &[f64],
    big_blind: f64,
    freq: F,
    node_index: N,
) -> Vec<TreeNode>
where
    F: Fn(usize, u32) -> [f64; 169],
    N: Fn(usize, u32) -> usize,
{
    if n < 2 {
        return Vec::new();
    }
    let bb = big_blind.max(1e-9);
    (0..n - 1)
        .map(|actor| first_in_node(n, actor, stacks, bb, &freq, &node_index))
        .collect()
}

fn first_in_node<F, N>(
    n: usize,
    actor: usize,
    stacks: &[f64],
    big_blind: f64,
    freq: &F,
    node_index: &N,
) -> TreeNode
where
    F: Fn(usize, u32) -> [f64; 169],
    N: Fn(usize, u32) -> usize,
{
    let stack = stacks.get(actor).copied().unwrap_or(0.0);
    let puts_in_more_than_bb = stack > big_blind + 1e-9;
    let action = if can_min_raise(stack, big_blind) {
        Action::Raise
    } else {
        Action::Call
    };
    let current_bet = if puts_in_more_than_bb {
        stack
    } else {
        big_blind
    };
    let raise_level = if puts_in_more_than_bb { 1 } else { 0 };
    let children = response_nodes(
        n,
        1 << actor,
        actor + 1,
        raise_level,
        current_bet,
        stacks,
        big_blind,
        true,
        freq,
        node_index,
    );
    let mut node = make_node(
        node_index(actor, 0),
        &mask_label(n, actor, 0),
        action,
        freq(actor, 0),
        None,
        children,
    );
    node.amount_bb = effective_stack_bb(actor, n, 0, stack, stacks, big_blind);
    node
}

fn response_nodes<F, N>(
    n: usize,
    mask: u32,
    next: usize,
    raise_level: u8,
    current_bet: f64,
    stacks: &[f64],
    big_blind: f64,
    nest: bool,
    freq: &F,
    node_index: &N,
) -> Vec<TreeNode>
where
    F: Fn(usize, u32) -> [f64; 169],
    N: Fn(usize, u32) -> usize,
{
    if next >= n {
        return Vec::new();
    }
    let stack = stacks.get(next).copied().unwrap_or(0.0);
    let behind = next + 1 < n;
    let reraises = behind && stack > current_bet + 1e-9;
    let (action, new_level, new_bet) = if reraises {
        let level = raise_level.saturating_add(1);
        (reraise_action(level), level, current_bet.max(stack))
    } else {
        (Action::Call, raise_level, current_bet)
    };
    let continue_mask = mask | (1 << next);
    let nested = if nest {
        response_nodes(
            n,
            continue_mask,
            next + 1,
            new_level,
            new_bet,
            stacks,
            big_blind,
            false,
            freq,
            node_index,
        )
    } else {
        Vec::new()
    };
    let mut continue_node = make_node(
        node_index(next, mask),
        &mask_label(n, next, mask),
        action,
        freq(next, mask),
        None,
        nested,
    );
    continue_node.amount_bb = action_amount_bb(
        next,
        n,
        mask,
        stack,
        current_bet,
        stacks,
        big_blind,
        reraises,
    );
    let mut children = vec![continue_node];
    if reraises {
        children.extend(short_call_leaves(
            n,
            continue_mask,
            next + 1,
            new_bet,
            stacks,
            big_blind,
            freq,
            node_index,
        ));
    }
    children.extend(response_nodes(
        n,
        mask,
        next + 1,
        raise_level,
        current_bet,
        stacks,
        big_blind,
        nest,
        freq,
        node_index,
    ));
    children
}

fn short_call_leaves<F, N>(
    n: usize,
    mask: u32,
    from: usize,
    current_bet: f64,
    stacks: &[f64],
    big_blind: f64,
    freq: &F,
    node_index: &N,
) -> Vec<TreeNode>
where
    F: Fn(usize, u32) -> [f64; 169],
    N: Fn(usize, u32) -> usize,
{
    (from..n)
        .filter(|&player| {
            let stack = stacks.get(player).copied().unwrap_or(0.0);
            !can_min_raise(stack, big_blind) && stack <= current_bet + 1e-9
        })
        .map(|player| {
            let stack = stacks.get(player).copied().unwrap_or(0.0);
            let mut node = make_node(
                node_index(player, mask),
                &mask_label(n, player, mask),
                Action::Call,
                freq(player, mask),
                None,
                Vec::new(),
            );
            node.amount_bb = action_amount_bb(
                player,
                n,
                mask,
                stack,
                current_bet,
                stacks,
                big_blind,
                false,
            );
            node
        })
        .collect()
}

fn reraise_action(level: u8) -> Action {
    match level {
        0 | 1 => Action::Raise,
        2 => Action::ThreeBet,
        _ => Action::FourBet,
    }
}

fn action_amount_bb(
    actor: usize,
    n: usize,
    mask: u32,
    stack: f64,
    current_bet: f64,
    stacks: &[f64],
    big_blind: f64,
    is_reraise: bool,
) -> f64 {
    let bb = big_blind.max(1e-9);
    if is_reraise {
        return effective_stack_bb(actor, n, mask, stack, stacks, big_blind);
    }
    let posted = posted_chips(actor, n, big_blind);
    ((stack.min(current_bet) - posted) / bb).max(0.0)
}

fn posted_chips(actor: usize, n: usize, big_blind: f64) -> f64 {
    if actor + 1 == n {
        big_blind
    } else if actor + 2 == n {
        big_blind * 0.5
    } else {
        0.0
    }
}

fn effective_stack_bb(
    actor: usize,
    n: usize,
    mask: u32,
    stack: f64,
    stacks: &[f64],
    big_blind: f64,
) -> f64 {
    let bb = big_blind.max(1e-9);
    let mut max_other: f64 = 0.0;
    for seat in 0..n {
        if seat == actor {
            continue;
        }
        let alive = (mask & (1 << seat)) != 0 || seat > actor;
        if alive {
            max_other = max_other.max(stacks.get(seat).copied().unwrap_or(0.0));
        }
    }
    if max_other <= 0.0 {
        stack / bb
    } else {
        stack.min(max_other) / bb
    }
}

fn mask_label(n: usize, actor: usize, mask: u32) -> String {
    let player = table_position_label(n, actor);
    let vs = vs_labels(n, mask);
    if vs.is_empty() {
        player.to_string()
    } else {
        format!("{player} vs {vs}")
    }
}

fn vs_labels(n: usize, mask: u32) -> String {
    (0..n)
        .filter(|seat| mask & (1 << seat) != 0)
        .map(|seat| table_position_label(n, seat))
        .collect::<Vec<_>>()
        .join(", ")
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
            vec![
                make_node(
                    1,
                    "SB call",
                    Action::Call,
                    ranges.sb_call_vs_btn,
                    evs.map(|e| e.sb_call_vs_btn),
                    vec![make_node(
                        3,
                        "BB call (vs BTN+SB)",
                        Action::CallSpecial,
                        ranges.bb_call_vs_btn_and_sb,
                        evs.map(|e| e.bb_call_vs_btn_and_sb),
                        vec![],
                    )],
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
    use poker_core::{solve, Algorithm, EquityCache, SolverInput};

    fn empty_3max_ranges() -> ThreeMaxRanges {
        ThreeMaxRanges {
            btn_push: [0.0; 169],
            sb_call_vs_btn: [0.0; 169],
            bb_call_vs_btn: [0.0; 169],
            bb_call_vs_btn_and_sb: [0.0; 169],
            sb_push: [0.0; 169],
            bb_call_vs_sb: [0.0; 169],
        }
    }

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

    #[test]
    fn three_max_tree_matches_hrc_nesting() {
        let tree = build_3max_tree(&empty_3max_ranges(), None);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree[0].label, "BTN push");
        assert_eq!(tree[0].children.len(), 2);
        assert_eq!(tree[0].children[0].label, "SB call");
        assert_eq!(tree[0].children[1].label, "BB call (vs BTN)");
        assert_eq!(tree[0].children[0].children.len(), 1);
        assert_eq!(tree[0].children[0].children[0].label, "BB call (vs BTN+SB)");
        assert_eq!(tree[1].label, "SB push (BTN fold)");
        assert_eq!(tree[1].children[0].label, "BB call (vs SB)");
    }

    #[test]
    fn finds_hero_spot_after_btn_fold() {
        let tree = build_3max_tree(&empty_3max_ranges(), None);
        let path = find_hero_decision_path(&tree, &[("BTN".to_string(), true)], "SB");
        assert_eq!(path, Some(vec![1]));
    }

    #[test]
    fn finds_bb_call_vs_btn() {
        let tree = build_3max_tree(&empty_3max_ranges(), None);
        let path = find_hero_decision_path(
            &tree,
            &[("BTN".to_string(), false), ("SB".to_string(), true)],
            "BB",
        );
        assert_eq!(path, Some(vec![0, 1]));
    }

    #[test]
    fn short_unopened_stack_is_call_and_later_players_still_raise() {
        let mut tree = vec![make_node(
            0,
            "UTG push",
            Action::Raise,
            [0.0; 169],
            None,
            vec![make_node(
                1,
                "EP call (vs UTG)",
                Action::Call,
                [0.0; 169],
                None,
                vec![],
            )],
        )];
        apply_short_stack_actions(
            &mut tree,
            &[81.0, 523.0, 810.0, 2507.0, 820.0, 1707.0, 1266.0, 1287.0],
            100.0,
        );
        assert!(matches!(tree[0].action, Action::Call));
        assert!(matches!(tree[0].children[0].action, Action::Raise));
    }

    #[test]
    fn stack_below_min_raise_is_unopened_call() {
        let stacks = [2980.0, 1320.0, 3020.0, 885.0, 2140.0, 273.0, 1462.0, 1420.0];
        let mut labeled = build_mask_tree(8, &stacks, 150.0, |_, _| [0.0; 169], eight_node);
        apply_short_stack_actions(&mut labeled, &stacks, 150.0);

        let btn = labeled
            .iter()
            .find(|node| node.player() == "BTN")
            .expect("BU");
        assert_eq!(btn.action, Action::Call);
        assert!((btn.amount_bb - 1.82).abs() < 0.02);

        let hj = labeled
            .iter()
            .find(|node| node.player() == "HJ")
            .expect("HJ");
        assert_eq!(hj.action, Action::Raise);
        assert!((hj.amount_bb - 5.90).abs() < 0.02);

        let hj_kids: Vec<_> = hj
            .children
            .iter()
            .map(|node| {
                (
                    node.player(),
                    node.action_code(),
                    node.label.clone(),
                    node.children.is_empty(),
                )
            })
            .collect();
        assert!(
            hj_kids.iter().any(|(p, a, label, leaf)| {
                *p == "BTN" && *a == "C" && label.contains("CO") && *leaf
            }),
            "HRC shows a leaf BU call vs HJ+CO, got {hj_kids:?}"
        );

        let bu_vs_hj = hj
            .children
            .iter()
            .find(|node| {
                node.player() == "BTN" && node.label.contains("HJ") && !node.label.contains("CO")
            })
            .expect("BU vs HJ");
        assert_eq!(bu_vs_hj.action, Action::Call);
        assert!((bu_vs_hj.amount_bb - 1.82).abs() < 0.02);
        assert!(
            !bu_vs_hj.children.is_empty(),
            "after short call SB/BB still act"
        );
        assert_eq!(bu_vs_hj.children[0].player(), "SB");
        assert_eq!(bu_vs_hj.children[0].action, Action::ThreeBet);

        let folded_to_bu = find_hero_decision_path(
            &labeled,
            &[
                ("UTG".to_string(), true),
                ("EP".to_string(), true),
                ("MP".to_string(), true),
                ("HJ".to_string(), true),
                ("CO".to_string(), true),
            ],
            "BTN",
        );
        let unopened =
            node_at_path(&labeled, &folded_to_bu.expect("unopened BU")).expect("unopened node");
        assert_eq!(unopened.player(), "BTN");
        assert_eq!(unopened.action, Action::Call);
        assert!(unopened.label.find("vs").is_none(), "{}", unopened.label);

        let vs_hj = find_hero_decision_path(
            &labeled,
            &[
                ("UTG".to_string(), true),
                ("EP".to_string(), true),
                ("MP".to_string(), true),
                ("HJ".to_string(), false),
                ("CO".to_string(), true),
            ],
            "BTN",
        );
        let facing = node_at_path(&labeled, &vs_hj.expect("BU vs HJ")).expect("vs HJ node");
        assert_eq!(facing.player(), "BTN");
        assert_eq!(facing.action, Action::Call);
        assert!(facing.label.contains("HJ"), "{}", facing.label);
        assert!(
            !facing.label.contains("CO"),
            "CO folded, should not be vs HJ+CO: {}",
            facing.label
        );
    }

    #[test]
    fn attach_node_evs_fills_call_nodes() {
        let mut tree = vec![make_node(
            1,
            "BTN call (vs CO)",
            Action::Call,
            [0.0; 169],
            None,
            vec![],
        )];
        let mut evs = vec![[0.0; 169]; 2];
        evs[1][0] = 1.5;
        attach_node_evs(&mut tree, &evs);
        assert_eq!(tree[0].ev_range.unwrap()[0], 1.5);
    }

    #[test]
    fn seven_max_tree_nests_threebet_and_bb_call() {
        let stacks = [975.0, 695.0, 1560.0, 2515.0, 3340.0, 1505.0, 2910.0];
        let tree = build_mask_tree(7, &stacks, 200.0, |_, _| [0.0; 169], seven_node);
        assert_eq!(tree[0].player(), "UTG");
        assert!(tree[0].action.is_raise());
        let mp = &tree[0].children[0];
        assert_eq!(mp.player(), "MP");
        assert!(mp.action.is_call());
        assert_eq!(
            mp.children
                .iter()
                .map(|node| (node.player(), node.action, node.children.is_empty()))
                .collect::<Vec<_>>(),
            vec![
                ("HJ", Action::ThreeBet, true),
                ("CO", Action::ThreeBet, true),
                ("BTN", Action::ThreeBet, true),
                ("SB", Action::ThreeBet, true),
                ("BB", Action::Call, true),
            ]
        );
        assert!((mp.children[4].amount_bb - 3.88).abs() < 0.02);
        let hj = &tree[0].children[1];
        assert_eq!(hj.player(), "HJ");
        assert_eq!(hj.action, Action::ThreeBet);
        let path = find_hero_decision_path(
            &tree,
            &[
                ("UTG".to_string(), false),
                ("MP".to_string(), true),
                ("HJ".to_string(), false),
                ("CO".to_string(), true),
                ("BTN".to_string(), true),
                ("SB".to_string(), true),
            ],
            "BB",
        );
        let node = node_at_path(&tree, &path.expect("bb path")).expect("bb node");
        assert_eq!(node.player(), "BB");
        assert!(node.action.is_call());
        assert!(node.children.is_empty());
        assert!((node.amount_bb - 6.80).abs() < 0.02);
        assert_eq!(node.range_id, seven_node(6, 0b101));

        let hj = &tree[2];
        assert_eq!(hj.player(), "HJ");
        assert!(hj.action.is_raise());
        assert_eq!(
            hj.children
                .iter()
                .map(|node| (node.player(), node.action, node.children.is_empty()))
                .collect::<Vec<_>>(),
            vec![
                ("CO", Action::ThreeBet, false),
                ("BTN", Action::ThreeBet, false),
                ("SB", Action::Call, false),
                ("BB", Action::Call, true),
            ]
        );
        assert_eq!(
            hj.children[2]
                .children
                .iter()
                .map(|node| (node.player(), node.action, node.children.is_empty()))
                .collect::<Vec<_>>(),
            vec![("BB", Action::Call, true)]
        );
        assert!((hj.children[2].children[0].amount_bb - 6.80).abs() < 0.02);
        let co = &hj.children[0];
        assert_eq!(
            co.children
                .iter()
                .map(|node| (node.player(), node.action, node.children.is_empty()))
                .collect::<Vec<_>>(),
            vec![
                ("BTN", Action::FourBet, true),
                ("SB", Action::Call, true),
                ("BB", Action::Call, true),
            ]
        );
        let btn = &co.children[0];
        assert!((btn.amount_bb - 14.55).abs() < 0.02);
        assert!(btn.children.is_empty());
        assert_eq!(max_depth(&tree), 2);
        let sb = &co.children[1];
        assert!((sb.amount_bb - 7.03).abs() < 0.02);
        assert!((co.children[2].amount_bb - 11.58).abs() < 0.02);

        let mut tree = tree;
        expand_path(&mut tree, &[2, 0, 1]);
        assert!(!tree[0].expanded);
        assert!(!tree[1].expanded);
        assert!(tree[2].expanded);
        assert!(tree[2].children[0].expanded);
        assert!(!tree[2].children[0].children[0].expanded);
        assert!(!tree[2].children[0].children[1].expanded);
        assert!(!tree[3].expanded);
    }

    #[test]
    fn amount_format_matches_hrc_decimals() {
        assert_eq!(format_amount_bb(7.025), "7.03");
        assert_eq!(format_amount_bb(12.575), "12.6");
        assert_eq!(format_amount_bb(14.55), "14.6");
        assert_eq!(format_amount_bb(7.80), "7.80");
    }

    fn max_depth(nodes: &[TreeNode]) -> usize {
        nodes
            .iter()
            .map(|node| {
                if node.children.is_empty() {
                    0
                } else {
                    1 + max_depth(&node.children)
                }
            })
            .max()
            .unwrap_or(0)
    }

    fn dump_tree(nodes: &[TreeNode], depth: usize) {
        for node in nodes {
            println!(
                "{:indent$}{} {:>5} {:<3} {:>5.1}%  id={}  {}",
                "",
                node.action_code(),
                format_amount_bb(node.amount_bb),
                display_player(node.player()),
                node.range_pct,
                node.range_id,
                range_notation(&node.range),
                indent = depth * 2,
            );
            dump_tree(&node.children, depth + 1);
        }
    }

    #[test]
    fn hrc_7max_spot_matches_tree_and_ranges() {
        let cache = EquityCache::from_bytes(include_bytes!("../../assets/equity_cache.bin"))
            .expect("equity cache");
        let input = SolverInput {
            stacks: vec![975.0, 695.0, 1560.0, 2515.0, 3340.0, 1505.0, 2910.0],
            payouts: vec![0.5, 0.3, 0.2],
            small_blind: 100.0,
            big_blind: 200.0,
            ante: 0.0,
            button_index: 4,
            max_iterations: 80,
            tolerance: 0.001,
            num_players: 7,
            verbose_convergence: false,
            profile: false,
            algorithm: Algorithm::Cfr7Max,
            rank_cache_strict: false,
            locked_ranges: HashMap::new(),
        };
        let output = solve(&input, &cache);
        let tree = build_strategy_tree(&output, 4, &input.stacks, 200.0);
        println!("=== our 7-max tree ===");
        dump_tree(&tree, 0);

        let utg = &tree[0];
        assert_eq!(utg.player(), "UTG");
        assert!(utg.action.is_raise());
        let kids: Vec<_> = utg
            .children
            .iter()
            .map(|n| {
                (
                    n.player(),
                    n.action_code(),
                    format_amount_bb(n.amount_bb),
                    n.children.is_empty(),
                    format!("{:.1}", n.range_pct),
                )
            })
            .collect();
        println!("UTG children: {kids:?}");
        println!(
            "HRC UTG: R 4.88 15.2% | C MP 3.48 9.2% | 3B HJ 7.80 4.7% | 3B CO 12.6 7.5% | 3B BU 14.6 7.5% | 3B SB 7.53 10.8% | C BB 3.88 ~20%"
        );
        println!(
            "OURS UTG: R 4.88 {:.1}% | C MP 3.48 {:.1}% | 3B HJ 7.80 {:.1}% | 3B CO 12.6 {:.1}% | 3B BU 14.6 {:.1}% | 3B SB 7.53 {:.1}% | C BB 3.88 {:.1}%",
            utg.range_pct,
            utg.children[0].range_pct,
            utg.children[1].range_pct,
            utg.children[2].range_pct,
            utg.children[3].range_pct,
            utg.children[4].range_pct,
            utg.children[5].range_pct,
        );
        let freq = &output.seven_max.as_ref().expect("7-max").freq;
        for (label, idx) in [
            ("UTG R", seven_node(0, 0)),
            ("MP vs UTG", seven_node(1, 1)),
            ("HJ vs UTG", seven_node(2, 1)),
            ("HJ vs UTG+MP", seven_node(2, 3)),
            ("CO vs UTG+MP", seven_node(3, 3)),
            ("BU vs UTG+MP", seven_node(4, 3)),
            ("SB vs UTG+MP", seven_node(5, 3)),
            ("BB vs UTG+MP", seven_node(6, 3)),
            ("CO vs UTG+HJ", seven_node(3, 0b101)),
            ("BU vs UTG+HJ", seven_node(4, 0b101)),
            ("SB vs UTG+HJ", seven_node(5, 0b101)),
            ("BB vs UTG+HJ", seven_node(6, 0b101)),
            ("CO vs UTG", seven_node(3, 1)),
            ("BB vs UTG", seven_node(6, 1)),
        ] {
            let row = &freq[idx];
            let max = row.iter().copied().fold(0.0, f64::max);
            println!(
                "{label:16} id={idx:3} share={:.1}% max={:.2}",
                combo_share(row) * 100.0,
                max
            );
        }
        assert_eq!(max_depth(&tree), 2);
        assert_eq!(
            kids.iter()
                .map(|(p, a, amt, leaf, _)| (*p, *a, amt.as_str(), *leaf))
                .collect::<Vec<_>>(),
            vec![
                ("MP", "C", "3.48", false),
                ("HJ", "3B", "7.80", false),
                ("CO", "3B", "12.6", false),
                ("BTN", "3B", "14.6", false),
                ("SB", "3B", "7.53", false),
                ("BB", "C", "3.88", true),
            ]
        );
        let mp = &utg.children[0];
        assert!(
            mp.children[0].range_pct > 1.5,
            "HJ vs UTG+MP {:.1}% vs HRC 4.7%",
            mp.children[0].range_pct
        );
        assert!(
            mp.children[1].range_pct > 1.5,
            "CO vs UTG+MP {:.1}% vs HRC 4.7%",
            mp.children[1].range_pct
        );
        assert!(
            utg.children[1].range_pct > 3.0,
            "HJ vs UTG {:.1}% vs HRC 7.5%",
            utg.children[1].range_pct
        );
        assert!(
            utg.children[1].children[0].range_pct > 0.4,
            "4B CO vs UTG+HJ {:.1}% vs HRC 2.1%",
            utg.children[1].children[0].range_pct
        );
        assert!(
            utg.children[1].children[3].range_pct < 35.0,
            "BB vs UTG+HJ {:.1}% vs HRC 3.6%",
            utg.children[1].children[3].range_pct
        );
        assert_eq!(utg.children[1].children[2].action, Action::Call);
        assert!((utg.children[1].children[2].amount_bb - 7.03).abs() < 0.02);
    }
}
