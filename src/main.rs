mod graph;
mod huffman_tree;
mod model;
mod activation_functions;
use rand::thread_rng;
use rand::seq::SliceRandom;
use std::cmp;

fn main() {
    let mut g = graph::Graph::new();

    g.build_graph_from_file("lastfm_asia_edges.txt");

    let hm = huffman_tree::HuffmanTree::new(g.get_node_iter());
    let mut model = model::Model::new(g.num_nodes(), 2);
    
    let mut node_ids = Vec::new();
    for i in g.get_node_iter() {
        node_ids.push(i.get_id());
    }

    
    let walk_len = 7;
    let window_size = 1;
    let num_iters = 100;


    let mut lr = 0.025;
    let start_lr = 0.025;

    for iter in 0..num_iters {
        let mut rng = thread_rng();
        node_ids.shuffle(&mut rng);
        let mut error = 0.0;
        for node in &node_ids {
            let walk = g.random_walk(node, walk_len);
            for v in 0..walk_len {
                let target = g.get_node_idx(&walk[v]).unwrap();
                let start = if window_size > v {0} else { v - window_size };
                for u in start..v {
                    let outcomes = hm.get_indices_and_turns(&walk[u]);
                    error += model.feed_forward(*target, outcomes, lr);
                }

                for u in (v+1)..cmp::min(v+window_size, walk_len) {
                    let outcomes = hm.get_indices_and_turns(&walk[u]);
                    error += model.feed_forward(*target, outcomes, lr);
                }
            }
        }
        if iter % 50 == 0 {
            println!("Iteration: {}", iter);
            println!("Learning Rate: {}", lr);
            println!("Error: {}", error/(node_ids.len() as f64));
        }
        
        lr = lr - start_lr/(num_iters as f64);

    }

}
