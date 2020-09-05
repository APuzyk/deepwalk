use crate::activation_functions::sigmoid;
use nalgebra::DVector;
use rand::distributions::Uniform;
use rand::thread_rng;
use std::sync::{RwLock, Arc, Mutex};

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
}

pub fn step(
    weight_mat: Arc<Vec<ConcurrentDVecf64>>,
    output_mat: Arc<Vec<ConcurrentDVecf64>>,
    node_idx: usize,
    outcomes: &Vec<(usize, f64)>,
    learning_rate: f64,
    error: Arc<Mutex<f64>>,
    vec_dim: usize,

) {
    let mut err = 0.0;
    let mut h_update = DVector::from_element(vec_dim, 0.0);
    {
        let node_vec = weight_mat[node_idx].read().unwrap();

        for (out_idx, outcome) in outcomes {
            let mut out_vec = output_mat[*out_idx].write().unwrap();

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
        let mut node_vec = weight_mat[node_idx].write().unwrap();
        node_vec.axpy(-1.0 * learning_rate, &h_update, 1.0);
    }
    {
        let mut error = error.lock().unwrap();
        *error += -1.0 * err;
    }
}

#[cfg(test)]
mod model_tests {
    use super::*;

    #[test]
    fn test_model() {
        let model = ConcurrentModel::new(3, 5);
    }

    #[test]
    fn test_vec() {
        let v1 = DVector::from_element(3, 1.0);
        let v2 = v1 * 3.0;

        assert_eq!(v2, DVector::from_element(3, 3.0));
    }
}
