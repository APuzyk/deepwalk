use crate::activation_functions::sigmoid;
use nalgebra::{DVector, Dynamic, Matrix, VecStorage};
use rand::distributions::Uniform;
use rand::thread_rng;

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

    pub fn feed_forward(
        &mut self,
        node_idx: usize,
        outcomes: Vec<(usize, f64)>,
        learning_rate: f64,
    ) -> f64 {
        let node_vec = self.weight_mat.column(node_idx);
        let mut error = 0.0;
        //let mut de_dvh_vec = Vec::with_capacity(outcomes.len());
        let mut h_update = DVector::from_element(self.vec_dim, 0.0);
        for (idx, outcome) in outcomes {
            let mut out_vec = self.output_mat.column_mut(idx);

            let nv_dot_ov = out_vec.dot(&node_vec);
            let e = sigmoid(outcome * nv_dot_ov).ln();
            error += e;

            let tj = if outcome == 1.0 { 1.0 } else { 0.0 };
            let de_dvh = sigmoid(nv_dot_ov) - tj;
            //de_dvh_vec.push(sigmoid(nv_dot_ov) - tj);
            let update_out_vec = (sigmoid(nv_dot_ov) - tj) * &node_vec;
            h_update.axpy(1.0, &(de_dvh * &out_vec), 1.0);
            out_vec.axpy(-1.0 * learning_rate, &update_out_vec, 1.0);
        }
        let mut node_vec = self.weight_mat.column_mut(node_idx);
        node_vec.axpy(-1.0 * learning_rate, &h_update, 1.0);
        if error.is_infinite() {
            panic!("haha");
        }
        //println!("Error: {}", -1.0 * error);
        -1.0 * error
    }
}

#[cfg(test)]
mod model_tests {
    use super::*;

    #[test]
    fn test_model() {
        let mut model = Model::new(3, 5);
        model.feed_forward(0, vec![(0, 1.0)], 0.5);
    }

    #[test]
    fn test_vec() {
        let v1 = DVector::from_element(3, 1.0);
        let v2 = v1 * 3.0;

        assert_eq!(v2, DVector::from_element(3, 3.0));
    }
}
