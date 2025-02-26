use crate::activation_functions::sigmoid;
use crate::graph::Graph;
use nalgebra::DVector;
use rand::distributions::Uniform;
use rand::thread_rng;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

type ConcurrentDVecf64 = Arc<RwLock<DVector<f64>>>;

pub struct ConcurrentModel {
    pub weight_mat: Arc<Vec<ConcurrentDVecf64>>,
    pub output_mat: Arc<Vec<ConcurrentDVecf64>>,
    pub vec_dim: usize,
}

impl ConcurrentModel {
    pub fn new(num_nodes: usize, vec_dim: usize) -> ConcurrentModel {
        let runif = Uniform::new(-0.5 / (vec_dim as f64), 0.5 / (vec_dim as f64));
        let mut rng = thread_rng();

        let mut weight_mat = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let dv = DVector::from_distribution(vec_dim, &runif, &mut rng);
            let dv = Arc::new(RwLock::new(dv));
            weight_mat.push(dv);
        }
        let weight_mat = Arc::new(weight_mat);

        let mut output_mat = Vec::with_capacity(num_nodes - 1);
        for _ in 0..(num_nodes - 1) {
            output_mat.push(Arc::new(RwLock::new(DVector::zeros(vec_dim))));
        }

        let output_mat = Arc::new(output_mat);

        ConcurrentModel {
            weight_mat,
            output_mat,
            vec_dim,
        }
    }

    pub fn write_weight_mat<P: AsRef<Path>>(&self, weight_file: &P, graph: Arc<Graph>) {
        let mut f = File::create(weight_file).expect("Unable to create output file for weights");
        for (node_id, node_idx) in graph.get_node_id_to_idx().iter() {
            write!(f, "{}", node_id).expect("Writing to the weight file errored");
            let node_vec = &self.weight_mat[*node_idx].read().unwrap();
            for i in 0..node_vec.shape().0 {
                write!(f, " {}", node_vec[(i, 0)]).expect("Writing to the weight file errored");
            }
            write!(f, "\n").expect("Writing to the weight file errored");
        }
    }

    pub fn step(
        &self,
        node_idx: usize,
        outcomes: &Vec<(usize, f64)>,
        learning_rate: f64,
        error: Arc<Mutex<f64>>,
        vec_dim: usize,
    ) {
        let mut err = 0.0;
        let mut h_update = DVector::from_element(vec_dim, 0.0);
        {
            let node_vec = self.weight_mat[node_idx].read().unwrap();

            for (out_idx, outcome) in outcomes {
                let mut out_vec = self.output_mat[*out_idx].write().unwrap();

                let nv_dot_ov = out_vec.dot(&node_vec);
                let e = sigmoid(outcome * nv_dot_ov).ln();
                err += e;

                let tj = if *outcome == 1.0 { 1.0 } else { 0.0 };
                let de_dvh = sigmoid(nv_dot_ov) - tj;
                h_update.axpy(1.0 * de_dvh, &out_vec, 1.0);
                out_vec.axpy(-1.0 * de_dvh * learning_rate, &node_vec, 1.0);
            }
        }
        {
            let mut node_vec = self.weight_mat[node_idx].write().unwrap();
            node_vec.axpy(-1.0 * learning_rate, &h_update, 1.0);
        }
        {
            let mut error = error.lock().unwrap();
            *error += -1.0 * err;
        }
    }
}

#[cfg(test)]
mod model_tests {
    use super::*;

    #[test]
    fn test_model() {
        let _model = ConcurrentModel::new(3, 5);
    }

    #[test]
    fn test_vec() {
        let v1 = DVector::from_element(3, 1.0);
        let v2 = v1 * 3.0;

        assert_eq!(v2, DVector::from_element(3, 3.0));
    }
}
