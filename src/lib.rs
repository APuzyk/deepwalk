pub mod activation_functions;
pub mod graph;
pub mod huffman_tree;
pub mod model;
pub mod model_concurrent;
use crossbeam::sync::WaitGroup;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::ThreadPoolBuilder;
use std::cmp;
use std::sync::{Arc, Mutex};

pub fn train(
    mut model: model::Model,
    huffman_tree: huffman_tree::HuffmanTree,
    graph: graph::Graph,
    walk_len: usize,
    window_size: usize,
    num_iterations: usize,
    learning_rate: f64,
) {
    let mut node_ids = Vec::new();
    for i in graph.get_node_iter() {
        node_ids.push(i.get_id());
    }

    let mut lr = learning_rate;
    let start_lr = 0.025;

    for iter in 0..num_iterations {
        let mut rng = thread_rng();
        node_ids.shuffle(&mut rng);
        let mut error = 0.0;
        for node in &node_ids {
            let walk = graph.random_walk(node, walk_len);
            for v in 0..walk_len {
                let target = graph.get_node_idx(&walk[v]).unwrap();
                let start = if window_size > v { 0 } else { v - window_size };
                for u in start..v {
                    let outcomes = huffman_tree.get_indices_and_turns(&walk[u]);
                    error += model.step(*target, outcomes, lr);
                }

                for u in (v + 1)..cmp::min(v + window_size, walk_len) {
                    let outcomes = huffman_tree.get_indices_and_turns(&walk[u]);
                    error += model.step(*target, outcomes, lr);
                }
            }
        }
        if iter % 50 == 0 {
            println!("Iteration: {}", iter);
            println!("Learning Rate: {}", lr);
            println!("Error: {}", error / (node_ids.len() as f64));
        }

        lr = lr - start_lr / (num_iterations as f64);
    }
}

pub fn train_concurrent(
    model: model_concurrent::ConcurrentModel,
    huffman_tree: huffman_tree::HuffmanTree,
    graph: graph::Graph,
    walk_len: usize,
    window_size: usize,
    num_iterations: usize,
    learning_rate: f64,
) {
    let mut node_ids = Vec::new();
    for i in graph.get_node_iter() {
        node_ids.push(i.get_id());
    }
    let vec_dim: usize = model.vec_dim;

    let mut lr = learning_rate;
    let start_lr = 0.025;
    let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();

    for iter in 0..num_iterations {
        let mut rng = thread_rng();
        node_ids.shuffle(&mut rng);
        let error = Arc::new(Mutex::new(0.0));
        //let wg = WaitGroup::new();
        for node in &node_ids {
            let mut targets = Vec::with_capacity(walk_len);
            for v in graph.random_walk(node, walk_len) {
                targets.push((v, graph.get_node_idx(&v).unwrap()));
            }

            let mut outcomes = Vec::with_capacity(walk_len);
            for (node_id, _) in &targets {
                let outs = huffman_tree.get_indices_and_turns(node_id);
                let mut this_encoding = Vec::with_capacity(outs.len());
                for (output_idx, y) in outs {
                    this_encoding.push((Arc::clone(&model.output_mat[output_idx]), y));
                }
                outcomes.push(this_encoding);
            }

            let mut step_weights = Vec::with_capacity(walk_len);
            for (_, &node_idx) in &targets {
                step_weights.push(Arc::clone(&model.weight_mat[node_idx]));
            }
            let error = Arc::clone(&error);
            //let wg = wg.clone();
            for v in 0..walk_len {
                let arc_node = &step_weights[v];
                let start = if window_size > v { 0 } else { v - window_size };
                for u in start..v {
                    let arc_outs = &outcomes[u];
                    let lr = lr;
                    pool.install(|| {
                        model_concurrent::step(arc_node, arc_outs, lr, vec_dim, &error);
                    });
                }

                for u in (v + 1)..cmp::min(v + window_size, walk_len) {
                    let arc_outs = &outcomes[u];
                    let lr = lr;
                    pool.install(|| {
                        model_concurrent::step(arc_node, arc_outs, lr, vec_dim, &error);
                    });
                }
            }
            // pool.install(move || {
            //     for v in 0..walk_len {
            //         let arc_node = &step_weights[v];
            //         let start = if window_size > v { 0 } else { v - window_size };
            //         for u in start..v {
            //             let arc_outs = &outcomes[u];
            //             let lr = lr;
            //             model_concurrent::step(arc_node, arc_outs, lr, vec_dim, &error);
            //         }

            //         for u in (v + 1)..cmp::min(v + window_size, walk_len) {
            //             let arc_outs = &outcomes[u];
            //             let lr = lr;
            //             model_concurrent::step(arc_node, arc_outs, lr, vec_dim, &error);
            //         }
            //     }
            //drop(wg);
            //});
        }
        //wg.wait();
        if iter % 5 == 0 {
            println!("Iteration: {}", iter);
            println!("Learning Rate: {}", lr);
            println!(
                "Error: {}",
                *error.lock().unwrap() / (node_ids.len() as f64)
            );
        }

        lr = lr - start_lr / (num_iterations as f64);
    }
}
