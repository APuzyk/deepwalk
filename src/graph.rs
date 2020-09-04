use rand::seq::SliceRandom;
use std::collections::hash_map;
use std::collections::HashMap;
use std::fs;

type NodeID = i32;

#[derive(Debug, PartialEq)]
pub struct GraphNode {
    node_id: NodeID,
    edge_list: Vec<NodeID>,
    num_edges: i32,
}

impl GraphNode {
    pub fn new(node_id: NodeID) -> GraphNode {
        GraphNode {
            node_id,
            edge_list: Vec::new(),
            num_edges: 0,
        }
    }

    pub fn add_edge(&mut self, target: NodeID) {
        self.edge_list.push(target);
        self.num_edges += 1;
    }

    pub fn get_weight(&self) -> i32 {
        self.num_edges
    }

    pub fn get_id(&self) -> NodeID {
        self.node_id
    }

    pub fn random_step(&self) -> Option<&NodeID> {
        self.edge_list.choose(&mut rand::thread_rng())
    }
}

#[derive(Debug)]
pub struct Graph {
    nodes: HashMap<NodeID, GraphNode>,
    node_to_idx_map: HashMap<NodeID, usize>,
}

impl Graph {
    pub fn new() -> Graph {
        let nodes: HashMap<NodeID, GraphNode> = HashMap::new();
        let node_to_idx_map: HashMap<NodeID, usize> = HashMap::new();
        Graph {
            nodes,
            node_to_idx_map,
        }
    }

    pub fn build(&mut self, edge_list: Vec<Vec<NodeID>>) {
        for edge in edge_list {
            let node1 = edge[0];
            let node2 = edge[1];

            self.node_to_idx_map
                .entry(node1)
                .or_insert(self.nodes.len());
            self.nodes.entry(node1).or_insert(GraphNode::new(node1));
            self.node_to_idx_map
                .entry(node2)
                .or_insert(self.nodes.len());
            self.nodes.entry(node2).or_insert(GraphNode::new(node2));

            if let Some(node) = self.nodes.get_mut(&node1) {
                node.add_edge(node2);
            }

            if let Some(node) = self.nodes.get_mut(&node2) {
                node.add_edge(node1);
            }
        }
    }

    pub fn random_walk(&self, starting_node: &NodeID, num_steps: usize) -> Vec<NodeID> {
        let mut curr_node = self
            .get_node(starting_node)
            .expect("Start node does not exist.");
        if curr_node.edge_list.len() == 0 {
            return vec![];
        }
        let mut path = Vec::with_capacity(num_steps);
        for _ in 0..num_steps {
            let next_node = match curr_node.random_step() {
                Some(n) => n,
                None => panic!("The graph is malformed"),
            };
            curr_node = self.get_node(next_node).expect(
                &format!(
                    "Node: {} points to non existant node {}",
                    curr_node.node_id, next_node
                )[..],
            );
            path.push(*next_node);
        }

        path
    }

    pub fn get_node(&self, node_id: &NodeID) -> Option<&GraphNode> {
        self.nodes.get(node_id)
    }

    pub fn build_graph_from_file(&mut self, filename: &str) {
        let contents = fs::read_to_string(filename).expect("Something went wrong reading the file");

        let mut edges = Vec::new();
        for line in contents.lines() {
            let mut edge = Vec::with_capacity(2);
            for node in line.split_whitespace() {
                edge.push(node.parse::<i32>().unwrap());
            }
            edges.push(edge);
        }

        self.build(edges);
    }

    pub fn get_node_iter(&self) -> hash_map::Values<NodeID, GraphNode> {
        self.nodes.values()
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn get_node_idx(&self, node_id: &NodeID) -> Option<&usize> {
        self.node_to_idx_map.get(node_id)
    }
}

#[cfg(test)]
mod graph_tests {
    use super::*;

    #[test]
    fn test_new_graph() {
        let edge_list = vec![vec![111, 222], vec![111, 333], vec![222, 333]];
        let mut g = Graph::new();
        g.build(edge_list);
        let mock_333 = GraphNode {
            node_id: 333,
            edge_list: vec![111, 222],
            num_edges: 2,
        };
        assert_eq!(g.get_node(&333).unwrap(), &mock_333);
        assert_eq!(g.get_node_idx(&111).unwrap(), &0);
    }

    #[test]
    fn test_random_walk() {
        let edge_list = vec![vec![111, 222], vec![111, 333], vec![222, 333]];
        let mut g = Graph::new();
        g.build(edge_list);
        let random_walk = g.random_walk(&111, 5);
        assert_eq!(random_walk.len(), 5);
    }
}
