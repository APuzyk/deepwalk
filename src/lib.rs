pub mod huffman_tree;
pub mod graph;
pub mod activation_functions;
pub mod model;
pub mod model_concurrent;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::ThreadPoolBuilder;
use std::cmp;
use std::sync::{Arc, Mutex};

pub fn train(
    model: model_concurrent::ConcurrentModel,
    huffman_tree: huffman_tree::HuffmanTree,
    graph: graph::Graph,
) {
    let mut node_ids = Vec::new();
    for i in graph.get_node_iter() {
        node_ids.push(i.get_id());
    }

    let walk_len = 7;
    let window_size = 1;
    let num_iters = 100;

    let mut lr = 0.025;
    let start_lr = 0.025;
    let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();

    for iter in 0..num_iters {
        let mut rng = thread_rng();
        node_ids.shuffle(&mut rng);
        let error = Arc::new(Mutex::new(0.0));
        for node in &node_ids {
            let walk = graph.random_walk(node, walk_len);

            for v in 0..walk_len {
                let target = graph.get_node_idx(&walk[v]).unwrap();
                let start = if window_size > v { 0 } else { v - window_size };
                for u in start..v {
                    let outcomes = huffman_tree.get_indices_and_turns(&walk[u]);
                    let mut arc_outs = Vec::with_capacity(outcomes.len());
                    for o in outcomes {
                        arc_outs.push((Arc::clone(&model.output_mat[o.0]), o.1));
                    }
                    let arc_node = Arc::clone(&model.weight_mat[*target]);
                    let lr = lr;
                    pool.install(|| {
                        model_concurrent::feed_forward(arc_node, arc_outs, lr, model.vec_dim, Arc::clone(&error))
                    });
                }

                for u in (v + 1)..cmp::min(v + window_size, walk_len) {
                    let outcomes = huffman_tree.get_indices_and_turns(&walk[u]);
                    let mut arc_outs = Vec::with_capacity(outcomes.len());
                    for o in outcomes {
                        arc_outs.push((Arc::clone(&model.output_mat[o.0]), o.1));
                    }
                    let arc_node = Arc::clone(&model.weight_mat[*target]);
                    let lr = lr;
                    pool.install(|| {
                        model_concurrent::feed_forward(arc_node, arc_outs, lr, model.vec_dim, Arc::clone(&error))
                    });
                }
            }
        }
        if iter % 50 == 0 {
            println!("Iteration: {}", iter);
            println!("Learning Rate: {}", lr);
            println!("Error: {}", *error.lock().unwrap() / (node_ids.len() as f64));
        }

        lr = lr - start_lr / (num_iters as f64);
    }
}
