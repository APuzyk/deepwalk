use std::env;

fn main() {
    deepwalk::deepwalk(env::args().collect())
}
