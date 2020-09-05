use std::time::Instant;

fn main() {

    let file = "karate_network.txt";
    const VEC_DIM: usize = 128;
    const WALK_LENGTH: usize = 7;
    const WINDOW_SIZE: usize = 2;
    const NUM_ITERATIONS: usize = 1000;
    let learning_rate = 0.025;

    println!("Concurrent Run Starting...");
    let now = Instant::now();
    let mut g = deepwalk::graph::Graph::new();
    g.build_graph_from_file(file);
    println!("...graph built..");
    let model = deepwalk::model_concurrent::ConcurrentModel::new(g.num_nodes(), VEC_DIM);
    let hm = deepwalk::huffman_tree::HuffmanTree::new(g.get_node_iter());
    println!("...model and huffman tree built...");
    deepwalk::train_concurrent(
        model,
        hm,
        g,
        WALK_LENGTH,
        WINDOW_SIZE,
        NUM_ITERATIONS,
        learning_rate,
    );
    println!("Concurrent run took {} seconds", now.elapsed().as_secs());

    // println!("Linear Run Starting...");
    // let now = Instant::now();

    // let mut g = deepwalk::graph::Graph::new();
    // g.build_graph_from_file(file);
    // let hm = deepwalk::huffman_tree::HuffmanTree::new(g.get_node_iter());
    // let model = deepwalk::model::Model::new(g.num_nodes(), VEC_DIM);
    // deepwalk::train(
    //     model,
    //     hm,
    //     g,
    //     WALK_LENGTH,
    //     WINDOW_SIZE,
    //     NUM_ITERATIONS,
    //     learning_rate,
    // );
    // println!("Linear run took {} seconds", now.elapsed().as_secs());
}
