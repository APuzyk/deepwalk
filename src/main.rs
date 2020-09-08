use std::env;

/// Runs [deepwalk](https://arxiv.org/abs/1403.6652) on a whitespace separated edge list.
/// 
/// Paramters, including the input file are provided in
/// a json file the location of which is passed as an 
/// argument to the CLI.  
/// 
/// # Examples
/// 
/// ```
/// deepwalk config.json
/// ```
fn main() {
    deepwalk::deepwalk(env::args().collect())
}
