use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Config {
    learning_rate: f64,
    vector_dim: usize,
    walk_length: usize,
    window_size: usize,
    num_iterations: usize,
    input_file: String,
    perf_file: String,
    weight_file: String,
    nthreads: u32,
}

impl Config {
    pub fn new<P: AsRef<Path>>(filename: &P) -> Config {
        let file = File::open(filename).expect("Couldn't open config file");
        let reader = BufReader::new(file);
        let config: Config = serde_json::from_reader(reader).expect("Couldn't read config file");
        config
    }

    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }
    pub fn vector_dim(&self) -> usize {
        self.vector_dim
    }
    pub fn walk_length(&self) -> usize {
        self.walk_length
    }
    pub fn window_size(&self) -> usize {
        self.window_size
    }
    pub fn num_iterations(&self) -> usize {
        self.num_iterations
    }
    pub fn input_file(&self) -> &str {
        &self.input_file[..]
    }
    pub fn perf_file(&self) -> &str {
        &self.perf_file[..]
    }
    pub fn weight_file(&self) -> &str {
        &self.weight_file[..]
    }
    pub fn nthreads(&self) -> u32 {
        self.nthreads
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn test_config() {
        let file = r#"{
            "learning_rate": 0.025,
            "vector_dim": 128,
            "walk_length": 10,
            "window_size": 2,
            "num_iterations": 25,
            "input_file": "karate_network.txt",
            "perf_file": "perf.txt",
            "weight_file": "weights.txt",
            "nthreads": 0
        }"#;

        let config: Config = serde_json::from_str(file).unwrap();
        assert_eq!(config.learning_rate(), 0.025);
        assert_eq!(config.vector_dim(), 128);
        assert_eq!(config.walk_length(), 10);
        assert_eq!(config.window_size(), 2);
        assert_eq!(config.input_file(), &"karate_network.txt"[..]);
        assert_eq!(config.perf_file(), &"perf.txt"[..]);
        assert_eq!(config.weight_file(), &"weights.txt"[..]);
        assert_eq!(config.nthreads, 0);
    }
}
