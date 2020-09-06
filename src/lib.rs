pub mod activation_functions;
pub mod config;
pub mod graph;
pub mod huffman_tree;
pub mod model;
pub mod model_concurrent;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::cmp;
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub fn deepwalk(args: Vec<String>) {
    let config = config::Config::new(&args[1]);

    println!("Run Starting...");
    let now = Instant::now();
    let mut g = graph::Graph::new();
    g.build_graph_from_file(config.input_file());
    println!("...graph built..");
    let hm = huffman_tree::HuffmanTree::new(g.get_node_iter());
    println!("...huffman tree built...");
    if config.nthreads() > 1 {
        let model = model_concurrent::ConcurrentModel::new(g.num_nodes(), config.vector_dim());
        train_concurrent(model, hm, g, config)
    } else {
        let model = model::Model::new(g.num_nodes(), config.vector_dim());
        train(model, hm, g, config);
    }
    println!("Run took {} seconds", now.elapsed().as_secs());
}
pub fn train(
    mut model: model::Model,
    huffman_tree: huffman_tree::HuffmanTree,
    graph: graph::Graph,
    config: config::Config,
) {
    let walk_len = config.walk_length();
    let window_size = config.window_size();

    let mut node_ids = Vec::new();
    for i in graph.get_node_iter() {
        node_ids.push(i.get_id());
    }

    let mut lr = config.learning_rate();
    let start_lr = 0.025;
    let mut f = File::create("perf.txt").expect("Unable to create output file for perf");
    writeln!(f, "iteration learning_rate error time").expect("Unable to write to perf file");
    let now = Instant::now();

    for iter in 0..config.num_iterations() {
        let mut rng = thread_rng();
        node_ids.shuffle(&mut rng);
        let mut error = 0.0;
        for node in &node_ids {
            let walk = graph.random_walk(node, walk_len);
            for v in 0..walk_len {
                let target = graph.get_node_idx(&walk[v]).unwrap();
                let start = if window_size > v { 0 } else { v - window_size };
                for u in start..cmp::min(v + window_size, walk_len) {
                    if u != v {
                        let outcomes = huffman_tree.get_indices_and_turns(&walk[u]);
                        error += model.step(*target, outcomes, lr);
                    }
                }
            }
        }
        if iter % 1 == 0 {
            println!(
                "Iteration: {}\nLearning Rate: {}\nError: {}",
                iter,
                lr,
                error / (node_ids.len() as f64)
            );
        }

        writeln!(
            f,
            "{} {} {} {}",
            iter,
            lr,
            error / (node_ids.len() as f64),
            now.elapsed().as_secs()
        )
        .expect("Unable to write to perf file");

        lr = lr - start_lr / (config.num_iterations() as f64);
    }
}

pub fn train_concurrent(
    model: model_concurrent::ConcurrentModel,
    huffman_tree: huffman_tree::HuffmanTree,
    graph: graph::Graph,
    config: config::Config,
) {
    let walk_len = config.walk_length();
    let window_size = config.window_size();

    let mut node_ids = Vec::new();
    for i in graph.get_node_iter() {
        node_ids.push(i.get_id());
    }
    let vec_dim: usize = model.vec_dim;

    let mut lr = config.learning_rate();
    let start_lr = 0.025;

    let mut f = File::create(config.perf_file()).expect("Unable to create output file for perf");
    writeln!(f, "iteration learning_rate error time").expect("Unable to write to perf file");
    let now = Instant::now();
    let model = Arc::new(model);
    for iter in 0..config.num_iterations() {
        let mut rng = thread_rng();
        node_ids.shuffle(&mut rng);
        let error = Arc::new(Mutex::new(0.0));
        crossbeam::scope(|scope| {
            for node in &node_ids {
                let mut targets = Vec::with_capacity(walk_len);
                let mut outcomes = Vec::with_capacity(walk_len);

                for v in graph.random_walk(node, walk_len) {
                    let target = graph.get_node_idx(&v).unwrap();
                    targets.push(*target);
                    let outs = huffman_tree.get_indices_and_turns(&v);
                    outcomes.push(outs);
                }

                let model = Arc::clone(&model);
                let error = error.clone();
                let learning_rate = lr;
                scope.spawn(move |_| {
                    for v in 0..walk_len {
                        let target = &targets[v];
                        let start = if window_size > v { 0 } else { v - window_size };
                        for u in start..cmp::min(v + window_size, walk_len) {
                            if u != v {
                                model.step(
                                    *target,
                                    &outcomes[u],
                                    learning_rate,
                                    error.clone(),
                                    vec_dim,
                                );
                            }
                        }
                    }
                });
            }
        })
        .expect("Node run failed");
        let err = *error.lock().unwrap() / (node_ids.len() as f64);
        if iter % 1 == 0 {
            println!("Iteration: {}", iter);
            println!("Learning Rate: {}", lr);
            println!("Error: {}", err);
        }
        //"iteration learning_rate error time");
        writeln!(f, "{} {} {} {}", iter, lr, err, now.elapsed().as_secs())
            .expect("Unable to write to perf file");

        lr = lr - start_lr / (config.num_iterations() as f64);
    }
    let f = File::create(config.weight_file()).expect("Unable to create output file for weights");
    model.write_weight_mat(f, graph);
}
