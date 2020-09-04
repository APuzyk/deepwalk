pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-1.0 * x).exp())
}

#[cfg(test)]
mod sigmoid_test {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let x = 0.0;
        assert_eq!(sigmoid(x), 0.5);
    }
}
