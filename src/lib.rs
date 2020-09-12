pub mod activation_functions;
pub mod config;
pub mod graph;
pub mod huffman_tree;
pub mod model;
pub mod model_concurrent;

use crossbeam::sync::WaitGroup;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::cmp;
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
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
        train_concurrent(Arc::new(model), Arc::new(hm), Arc::new(g), Arc::new(config));
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
    model: Arc<model_concurrent::ConcurrentModel>,
    huffman_tree: Arc<huffman_tree::HuffmanTree>,
    graph: Arc<graph::Graph>,
    config: Arc<config::Config>,
) {
    let mut node_ids = Vec::new();
    for i in graph.get_node_iter() {
        node_ids.push(i.get_id());
    }
    let mut lr = config.learning_rate();
    let start_lr = lr;

    let mut f = File::create(config.perf_file()).expect("Unable to create output file for perf");
    writeln!(f, "iteration learning_rate error time").expect("Unable to write to perf file");
    let now = Instant::now();

    for iter in 0..config.num_iterations() {
        let mut rng = thread_rng();
        node_ids.shuffle(&mut rng);

        let error = Arc::new(Mutex::new(0.0));
        let tmp_nodes = Arc::new(RwLock::new(node_ids.to_vec()));

        let wg_iter = WaitGroup::new();
        let nthreads = if node_ids.len() > config.nthreads() {
            config.nthreads()
        } else {
            node_ids.len()
        };

        for _ in 0..nthreads {
            let wg_iter = wg_iter.clone();
            let model = Arc::clone(&model);
            let error = Arc::clone(&error);
            let learning_rate = lr;
            let tmp_nodes = Arc::clone(&tmp_nodes);
            let huffman_tree = Arc::clone(&huffman_tree);
            let graph = Arc::clone(&graph);
            let config = Arc::clone(&config);
            thread::spawn(move || {
                let walk_len = config.walk_length();
                let window_size = config.window_size();
                while tmp_nodes.read().unwrap().len() > 0 {
                    let node = tmp_nodes.write().unwrap().pop();
                    match node {
                        Some(node) => {
                            let walk = graph.random_walk(&node, walk_len);
                            for (i, target) in walk.iter().enumerate() {
                                let start = if window_size > i { 0 } else { i - window_size };
                                for u in start..cmp::min(i + window_size, walk_len) {
                                    if u != i {
                                        model.step(
                                            *graph.get_node_idx(target).unwrap(),
                                            &huffman_tree.get_indices_and_turns(&walk[u]),
                                            learning_rate,
                                            error.clone(),
                                            config.vector_dim(),
                                        );
                                    }
                                }
                            }
                        }
                        None => (),
                    }
                }
                drop(wg_iter);
            });
        }
        wg_iter.wait();
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
    model.write_weight_mat(&config.weight_file(), graph);
}
