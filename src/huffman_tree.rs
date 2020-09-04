use crate::graph::GraphNode;
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};

#[derive(Debug)]
pub struct HuffmanTree {
    tree_vec: Vec<TreeNode>,
    leaf_id_idx_map: HashMap<i32, usize>,
}

impl HuffmanTree {
    pub fn new<'a, I>(graph: I) -> HuffmanTree
    where
        I: ExactSizeIterator<Item = &'a GraphNode>,
    {
        let mut init_tree_nodes = Vec::with_capacity(graph.len());

        for gn in graph {
            let mut new_node = TreeNode::new(gn.get_weight());
            new_node.set_leaf_id(gn.get_id());
            init_tree_nodes.push(new_node);
        }
        if init_tree_nodes.len() <= 1 {
            panic!("We only have one leaf");
        }

        init_tree_nodes.sort_by(|a, b| b.cmp(a));
        let mut first_queue = VecDeque::with_capacity(init_tree_nodes.len());
        while let Some(node) = init_tree_nodes.pop() {
            first_queue.push_back(node)
        }
        let mut second_queue = VecDeque::with_capacity(init_tree_nodes.len());
        let mut tree_index: usize = 0;

        let mut ht = HuffmanTree {
            tree_vec: Vec::with_capacity((2 * first_queue.len()) + 1),
            leaf_id_idx_map: HashMap::new(),
        };

        while (first_queue.len() > 0) || (second_queue.len() > 1) {
            let mut right_child = get_min_from_queues(&mut first_queue, &mut second_queue);
            let mut left_child = get_min_from_queues(&mut first_queue, &mut second_queue);
            let new_weight = left_child.weight + right_child.weight;

            let left_child_idx = ht.tree_vec.len();
            let right_child_idx = left_child_idx + 1;

            if let Some(node_id) = left_child.leaf_id {
                ht.leaf_id_idx_map.insert(node_id, left_child_idx);
            }
            left_child.is_right_child = Some(false);

            ht.tree_vec_push(left_child);

            if let Some(node_id) = right_child.leaf_id {
                ht.leaf_id_idx_map.insert(node_id, right_child_idx);
            }
            right_child.is_right_child = Some(true);
            ht.tree_vec_push(right_child);

            let mut new_node = TreeNode::new(new_weight);
            new_node.set_children(left_child_idx, right_child_idx);
            new_node.set_tree_index(tree_index);

            second_queue.push_back(new_node);
            tree_index += 1;
        }

        let root_node = second_queue.pop_front().unwrap();
        ht.tree_vec_push(root_node);
        ht.update_parents();
        ht
    }

    fn tree_vec_push(&mut self, tree_node: TreeNode) {
        self.tree_vec.push(tree_node);
    }

    fn update_parents(&mut self) {
        let mut stack = Vec::new();
        let n = self.tree_vec.len();
        stack.push((self.tree_vec[n - 1].left.unwrap(), n - 1));
        stack.push((self.tree_vec[n - 1].right.unwrap(), n - 1));
        while let Some((next, parent)) = stack.pop() {
            self.tree_vec[next].set_parent(parent);
            match (self.tree_vec[next].left, self.tree_vec[next].right) {
                (Some(l), Some(r)) => {
                    stack.push((l, next));
                    stack.push((r, next));
                    ()
                }
                (_, _) => (),
            }
        }
    }

    pub fn get_indices_and_turns(&self, node_id: &i32) -> Vec<(usize, f64)> {
        let is_right_child = self.get_encoding(node_id);
        let node_indexes = self.get_tree_indexes(node_id);
        node_indexes
            .iter()
            .zip(is_right_child.iter())
            .map(|(a, b)| (*a, *b))
            .collect::<Vec<(usize, f64)>>()
    }

    pub fn get_encoding(&self, node_id: &i32) -> Vec<f64> {
        let index = self.leaf_id_idx_map.get(node_id).unwrap();
        let mut encoding = Vec::new();
        let mut curr = &self.tree_vec[*index];

        while let Some(rc) = curr.is_right_child {
            let to_add = if rc { 1.0 } else { -1.0 };
            encoding.push(to_add);
            let parent_id = curr.parent.unwrap();
            curr = &self.tree_vec[parent_id];
        }
        encoding
    }

    fn get_tree_indexes(&self, node_id: &i32) -> Vec<usize> {
        let index = self.leaf_id_idx_map.get(node_id).unwrap();
        let mut indices = Vec::new();
        let mut curr = &self.tree_vec[*index];

        while let Some(p) = curr.parent {
            curr = &self.tree_vec[p];
            indices.push(curr.tree_index.unwrap());
        }
        indices
    }
}
fn get_min_from_queues(
    first_queue: &mut VecDeque<TreeNode>,
    second_queue: &mut VecDeque<TreeNode>,
) -> TreeNode {
    match (first_queue.get(0), second_queue.get(0)) {
        (None, Some(_)) => second_queue.pop_front().unwrap(),
        (Some(_), None) => first_queue.pop_front().unwrap(),
        (Some(a), Some(b)) => {
            let o = if a.weight < b.weight {
                first_queue.pop_front().unwrap()
            } else {
                second_queue.pop_front().unwrap()
            };
            o
        }
        (None, None) => panic!("One of the queues should have entries"),
    }
}
#[derive(Eq, Debug)]
pub struct TreeNode {
    parent: Option<usize>,
    is_right_child: Option<bool>,
    left: Option<usize>,
    right: Option<usize>,
    weight: i32,
    leaf_id: Option<i32>,
    tree_index: Option<usize>,
}

impl PartialOrd for TreeNode {
    fn partial_cmp(&self, other: &TreeNode) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TreeNode {
    fn cmp(&self, other: &TreeNode) -> Ordering {
        self.weight.cmp(&other.weight)
    }
}

impl PartialEq for TreeNode {
    fn eq(&self, other: &TreeNode) -> bool {
        self.weight == other.weight
    }
}

impl TreeNode {
    pub fn new(weight: i32) -> TreeNode {
        TreeNode {
            parent: None,
            is_right_child: None,
            left: None,
            right: None,
            weight: weight,
            leaf_id: None,
            tree_index: None,
        }
    }

    pub fn set_leaf_id(&mut self, id: i32) {
        self.leaf_id = Some(id);
    }

    pub fn set_children(&mut self, left: usize, right: usize) {
        self.left = Some(left);
        self.right = Some(right);
    }

    pub fn set_parent(&mut self, parent: usize) {
        self.parent = Some(parent);
    }

    pub fn set_tree_index(&mut self, index: usize) {
        self.tree_index = Some(index);
    }
}

#[cfg(test)]
mod tree_node_tests {
    use super::*;

    #[test]
    fn test_new_tree() {
        let mut g1 = GraphNode::new(111);
        let mut g2 = GraphNode::new(222);
        let g3 = GraphNode::new(333);
        g1.add_edge(222);
        g2.add_edge(333);
        g1.add_edge(333);

        let test_graph = vec![&g3, &g2, &g1];
        let ht = HuffmanTree::new(test_graph.into_iter());
        assert_eq!(ht.tree_vec.len(), 5);
    }

    #[test]
    fn test_get_path_and_turns() {
        let mut g1 = GraphNode::new(111);
        let mut g2 = GraphNode::new(222);
        let g3 = GraphNode::new(333);
        g1.add_edge(222);
        g2.add_edge(333);
        g1.add_edge(333);

        let test_graph = vec![&g3, &g2, &g1];
        let ht = HuffmanTree::new(test_graph.into_iter());
        assert_eq!(ht.get_indices_and_turns(&333), vec![(0, 1.0), (1, 1.0)]);
    }
}
