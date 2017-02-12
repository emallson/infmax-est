extern crate bit_set;
extern crate rayon;
extern crate petgraph;
extern crate docopt;
extern crate rand;
extern crate serde_json;
extern crate capngraph;
extern crate ris;
extern crate rustc_serialize;
#[macro_use]
extern crate slog;
extern crate slog_json;
extern crate slog_stream;
extern crate slog_term;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

use ris::*;

use std::collections::BTreeSet;
use petgraph::prelude::*;
use petgraph::graph::node_index;
use rayon::prelude::*;

use std::fs::File;
use slog::{Logger, DrainExt};
use serde_json::to_string as json_string;

#[cfg_attr(rustfmt, rustfmt_skip)]
const USAGE: &'static str = "
Estimate the actual influence of a given seed set.

Usage:
    est-inf <graph> <model> <epsilon> <delta> [--file <seedfile>] [--threads <threads>] [--log <file>]
    est-inf (-h | --help)

Options:
    -h --help               Show this screen.
    --file <seedfile>       Load seeds from file.
    --log <file>            Log results to <file>.
    --threads <threads>     Number of threads to use.
";

#[derive(Debug, RustcDecodable)]
struct Args {
    arg_graph: String,
    arg_model: Model,
    arg_epsilon: f64,
    arg_delta: f64,
    flag_file: Option<String>,
    flag_log: Option<String>,
    flag_threads: Option<usize>,
}

#[derive(Debug, RustcDecodable)]
enum Model {
    IC,
    LT,
}

pub fn load_seeds(fname: String) -> Result<Vec<usize>, std::io::Error> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::str::FromStr;

    File::open(fname.as_str()).and_then(|f| {
        BufReader::new(f)
            .lines()
            .map(|line| Ok(usize::from_str(try!(line).as_str()).unwrap()))
            .collect::<Result<Vec<usize>, _>>()
    })
}

fn estimate_influence(g: Graph<(), f32>,
                      seeds: Vec<usize>,
                      epsilon: f64,
                      delta: f64,
                      model: Model,
                      log: Logger)
                      -> f64 {
    let epsilon_2 = epsilon / 16.0;
    let delta_2 = delta / 16.0;
    let epsilon_2 = epsilon_2 / (1.0 - epsilon_2);

    let seeds = seeds.into_iter().map(|id: usize| node_index(id)).collect::<BTreeSet<NodeIndex>>();

    let upper = 1.0 +
                (2.0 + 2.0 * epsilon_2 / 3.0) * (1.0 + epsilon_2) * (1.0 / delta_2).ln() /
                epsilon_2.powi(2);
    let upper = upper as usize;
    info!(log, "upper bound"; "ub" => upper);
    let mut degree = 0;
    let mut counter = 0;
    let step = 10_000;
    let mut samples: Vec<BTreeSet<NodeIndex>> = Vec::with_capacity(step);
    while degree < upper {
        (0..step)
            .into_par_iter()
            .map(|_| match model {
                Model::IC => IC::new_uniform(&g).collect(),
                Model::LT => LT::new_uniform(&g).collect(),
            })
            .collect_into(&mut samples);

        for sample in &samples {
            counter += 1;
            if sample.intersection(&seeds).take(1).count() > 0 {
                degree += 1;
            }
            if degree >= upper {
                break;
            }
        }

        if counter / step % 10 == 0 {
            debug!(log, "remaining"; "ub" => upper, "deg" => degree, "rem" => upper - degree);
        }
    }

    return degree as f64 * g.node_count() as f64 / counter as f64;
}

fn main() {
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.decode())
        .unwrap_or_else(|e| e.exit());

    let log =
        match args.flag_log {
            Some(filename) => slog::Logger::root(slog::Duplicate::new(slog_term::streamer().color().compact().build(),
                                                                  slog_stream::stream(File::create(filename).unwrap(), slog_json::default())).fuse(), o!("version" => env!("CARGO_PKG_VERSION"))),
            None => {
                slog::Logger::root(slog_term::streamer().color().compact().build().fuse(),
                                   o!("version" => env!("CARGO_PKG_VERSION")))
            }
        };


    let g = capngraph::load_graph(args.arg_graph.as_str()).unwrap();

    if let Some(threads) = args.flag_threads {
        rayon::initialize(rayon::Configuration::new().set_num_threads(threads)).unwrap();
    }

    if let Some(file) = args.flag_file {
        let seeds = load_seeds(file).unwrap();
        info!(log, "seeds"; "input" => json_string(&seeds).unwrap(), "count" => seeds.len());
        let inf = estimate_influence(g,
                                     seeds,
                                     args.arg_epsilon,
                                     args.arg_delta,
                                     args.arg_model,
                                     log.new(o!("section" => "estimate influence")));
        info!(log, "estimated influence"; "inf" => inf);
    } else {
        panic!("No seeds given! Pass either --seeds or --file to provide seeds.");
    }
}
