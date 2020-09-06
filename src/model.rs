use crate::activation_functions::sigmoid;
use crate::graph::Graph;
use nalgebra::{DVector, Dynamic, Matrix, VecStorage};
use rand::distributions::Uniform;
use rand::thread_rng;
use std::fs::File;
use std::io::Write;
use std::path::Path;

type DMatrixf64 = Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>;

pub struct Model {
    weight_mat: DMatrixf64,
    output_mat: DMatrixf64,
    vec_dim: usize,
}

impl Model {
    pub fn new(num_nodes: usize, vec_dim: usize) -> Model {
        let runif = Uniform::new(-0.5 / (vec_dim as f64), 0.5 / (vec_dim as f64));
        let mut rng = thread_rng();
        let weight_mat = DMatrixf64::from_distribution(vec_dim, num_nodes, &runif, &mut rng);
        let output_mat = DMatrixf64::zeros(vec_dim, num_nodes - 1);

        Model {
            weight_mat,
            output_mat,
            vec_dim,
        }
    }

    pub fn step(
        &mut self,
        node_idx: usize,
        outcomes: Vec<(usize, f64)>,
        learning_rate: f64,
    ) -> f64 {
        let node_vec = self.weight_mat.column(node_idx);
        let mut error = 0.0;
        let mut h_update = DVector::from_element(self.vec_dim, 0.0);
        for (idx, outcome) in outcomes {
            let mut out_vec = self.output_mat.column_mut(idx);
            let nv_dot_ov = out_vec.dot(&node_vec);

            error += sigmoid(outcome * nv_dot_ov).ln();

            // Derivative of error with respect to out_vec*node_vec
            let de_dvh = if outcome == 1.0 {
                sigmoid(nv_dot_ov) - 1.0
            } else {
                sigmoid(nv_dot_ov)
            };

            h_update.axpy(de_dvh, &out_vec, 1.0);
            out_vec.axpy(-1.0 * learning_rate * de_dvh, &node_vec, 1.0);
        }
        let mut node_vec = self.weight_mat.column_mut(node_idx);
        node_vec.axpy(-1.0 * learning_rate, &h_update, 1.0);
        -1.0 * error
    }

    pub fn write_weight_mat<P: AsRef<Path>>(&self, weight_file: &P, graph: Graph) {
        let mut f = File::create(weight_file).expect("Unable to create output file for weights");
        for (node_id, node_idx) in graph.get_node_id_to_idx().iter() {
            write!(f, "{}", node_id).expect("Writing to the weight file errored");
            let node_vec = &self.weight_mat.column(*node_idx);
            for i in 0..node_vec.shape().0 {
                write!(f, " {}", node_vec[(i, 0)]).expect("Writing to the weight file errored");
            }
            write!(f, "\n").expect("Writing to the weight file errored");
        }
    }
}

#[cfg(test)]
mod model_tests {
    use super::*;

    #[test]
    fn test_model() {
        let mut model = Model::new(3, 5);
        model.step(0, vec![(0, 1.0)], 0.5);
    }

    #[test]
    fn test_vec() {
        let v1 = DVector::from_element(3, 1.0);
        let v2 = v1 * 3.0;

        assert_eq!(v2, DVector::from_element(3, 3.0));
    }
}
