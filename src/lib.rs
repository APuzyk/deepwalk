pub mod activation_functions;
pub mod graph;
pub mod huffman_tree;
pub mod model;
pub mod model_concurrent;
use crossbeam::sync::WaitGroup;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::{cmp, thread};
use std::sync::{Arc, Mutex};
use std::fs::File;
use std::time::Instant;
use std::io::Write;

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
        if iter % 1 == 0 {
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

    let mut f = File::create("perf.txt").expect("Unable to create output file for perf");
    writeln!(f, "iteration learning_rate error time").expect("Unable to write to perf file");
    let now = Instant::now();

    for iter in 0..num_iterations {
        let mut rng = thread_rng();
        node_ids.shuffle(&mut rng);
        let error = Arc::new(Mutex::new(0.0));
        let wg = WaitGroup::new();
        for node in &node_ids {
            let mut targets = Vec::with_capacity(walk_len);
            let mut outcomes = Vec::with_capacity(walk_len);

            for v in graph.random_walk(node, walk_len) {
                let target = graph.get_node_idx(&v).unwrap();
                targets.push(*target);
                let outs = huffman_tree.get_indices_and_turns(&v);
                outcomes.push(outs);
            }

            let weight_mat = model.weight_mat.clone();
            let output_mat = model.output_mat.clone();
            let error = error.clone();
            let wg = wg.clone();
            thread::spawn(move || {
                for v in 0..walk_len {
                    let target = targets[v];
                    let start = if window_size > v { 0 } else { v - window_size };
                    for u in start..v {
                        let outcomes = &outcomes[u];
                        let weight_mat = weight_mat.clone();
                        let output_mat = output_mat.clone();
                        let error = error.clone();
                        model_concurrent::step(weight_mat, output_mat,
                            target, outcomes, learning_rate, error, vec_dim);
                    }

                    for u in (v + 1)..cmp::min(v + window_size, walk_len) {
                        let outcomes = &outcomes[u];
                        let weight_mat = weight_mat.clone();
                        let output_mat = output_mat.clone();
                        let error = error.clone();
                        model_concurrent::step(weight_mat, output_mat,
                            target, outcomes, learning_rate, error, vec_dim);
                    }
                }
                drop(wg);
            });
        }
        wg.wait();
        let err = *error.lock().unwrap() / (node_ids.len() as f64);
        if iter % 1 == 0 {
            println!("Iteration: {}", iter);
            println!("Learning Rate: {}", lr);
            
            println!(
                "Error: {}",
                err
            );
        }
        //"iteration learning_rate error time");
        writeln!(f, "{} {} {} {}",
            iter, lr, err, now.elapsed().as_secs()).expect("Unable to write to perf file");

        lr = lr - start_lr / (num_iterations as f64);
    }
    let f = File::create("test.txt").expect("Unable to create output file for weights");
    model.write_weight_mat(f, graph);
}