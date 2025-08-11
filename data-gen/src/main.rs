#![allow(unused)]

use clap::Arg;
use clap::Parser;
use kdam::tqdm;
use kdam::BarExt;
use std::env::args;
use std::iter;
use std::ops::Index;
use std::panic::resume_unwind;
use std::thread;
use std::sync::mpsc;
use std::thread::Thread;
use std::error::Error;
use std::cmp::max;
use std::cmp::min;

/// Program to generate data
#[derive(Parser, Debug, Clone)]
#[command(about, long_about = None)]
struct Args {
    /// Maximum length of a sequence.
    #[arg(short, long, default_value_t = 10)]
    max_word_length: i64,

    /// Group order to use in practice.
    #[arg(short, long, default_value_t = 3)]
    braid_count: i64,

    /// Dataset size.
    #[arg(short, long, default_value_t = 100_000)]
    dataset_size: i64,

    /// Filename to write to.
    /// Don't include file extensions.
    #[arg(short, long, default_value_t = String::from("data"))]
    filename: String,

    /// Number of threads to use.
    /// Ensure threads number divides dataset size.
    #[arg(short, long, default_value_t = 1)]
    threads: i64,

    /// Maximum hypothetical group order that can be used later.
    /// Used for scaling gen only.
    #[arg(short='M', long, default_value_t = 5)]
    braid_count_to_scale_to: i64,

    /// Amount of files to generate.
    /// If this is greater than -1, it will generate this number of files.
    #[arg(short='A', long, default_value_t = -1)]
    amount_to_gen: i64,

    /// Starting number.
    /// If this is greater than -1, files will be numbered starting from this number.
    #[arg(short='s', long, default_value_t = -1)]
    start_index: i64,
}

// finds x^+
fn pos(x: i64) -> i64 {
    return max(x, 0);
}

fn neg(x: i64) -> i64 {
    return min(x, 0);
}

// the next two functions are derived from Thiffeault ch8
// it's kind of a weird piecewise function
// so apologies if it isn't nice to read

/// Updates a dynnikov coordinate based on a generator.
fn dynnikov_pve_sigma_action(coord: &Vec<i64>, sigma: i64) -> Vec<i64> {
    let n = (coord.len() as i64+4)/2;
    let i = sigma as usize;

    // denotes some key places in the coordinate vector
    // lot stands for last of type
    let lot = (n-3) as usize;
    let a_start: usize = 0;
    let b_start: usize = lot + 1;

    let mut new_coord = coord.to_vec();

    // deal with edge cases
    if sigma == 1 {
        let a1 = coord[0];
        let b1 = coord[b_start];

        new_coord[a_start] = -b1 + pos(a1 + pos(b1));
        new_coord[b_start] = a1 + pos(b1);
    } else if sigma == n-1 {
        let a_last = coord[a_start + lot];
        let b_last = coord[b_start + lot];

        new_coord[a_start + lot] = -b_last + neg(a_last + neg(b_last));
        new_coord[b_start + lot] = a_last + neg(b_last);
    } else {
        // the normal case
        let aisub1 = coord[a_start + i - 2];
        let bisub1 = coord[b_start + i - 2];
        let ai = coord[a_start + i - 1];
        let bi = coord[b_start + i - 1];

         // technically this is c_{i-1}, oh well
        let c = aisub1 - ai - pos(bi) + neg(bisub1);

        new_coord[a_start + i - 2] = aisub1 - pos(bisub1) - pos(pos(bi) + c);
        new_coord[b_start + i - 2] = bi + neg(c);
        new_coord[a_start + i - 1] = ai - neg(bi) - neg(neg(bisub1) - c);
        new_coord[b_start + i - 1] = bisub1 - neg(c);
    }

    return new_coord;
}

/// Updates a dynnikov coordinate based on an inverse generator.
fn dynnikov_nve_sigma_action(coord: &Vec<i64>, sigma: i64) -> Vec<i64> {
    let n = (coord.len() as i64+4)/2;
    let i = sigma as usize;

    // denotes some key places in the coordinate vector
    // lot stands for last of type
    let lot = (n-3) as usize;
    let a_start: usize = 0;
    let b_start: usize = lot + 1;

    let mut new_coord = coord.to_vec();

    // deal with edge cases
    if sigma == 1 {
        let a1 = coord[0];
        let b1 = coord[b_start];

        new_coord[a_start] = b1 - pos(pos(b1) - a1);
        new_coord[b_start] = pos(b1) - a1;
    } else if sigma == n-1 {
        let a_last = coord[a_start + lot];
        let b_last = coord[b_start + lot];

        new_coord[a_start + lot] = b_last - neg(neg(b_last) - a_last);
        new_coord[b_start + lot] = neg(b_last) - a_last;
    } else {
        // the normal case
        let aisub1 = coord[a_start + i - 2];
        let bisub1 = coord[b_start + i - 2];
        let ai = coord[a_start + i - 1];
        let bi = coord[b_start + i -1];

        // technically this is d_{i-1}, oh well
        let d = aisub1 - ai + pos(bi) - neg(bisub1);

        new_coord[a_start + i - 2] = aisub1 + pos(bisub1) + pos(pos(bi) - d);
        new_coord[b_start + i - 2] = bi - pos(d);
        new_coord[a_start + i - 1] = ai + neg(bi) + neg(neg(bisub1) + d);
        new_coord[b_start + i - 1] = bisub1 + pos(d);
    }

    return new_coord;
}

/// Computes the action of a braid word on a dynnikov coordinate.
/// Braids act left to right.
fn dynnikov_word(coord: &Vec<i64>, word: &Vec<i64>) -> Vec<i64> {
    let mut new_coord = coord.to_vec();

    for sigma in word.iter() {
        if *sigma == 0 {
            // 0 is the identity, do nothing
            continue;
        } else if *sigma > 0 {
            // +ve version of generators
            new_coord = dynnikov_pve_sigma_action(&new_coord, *sigma);
        } else {
            // -ve version of generators
            new_coord = dynnikov_nve_sigma_action(&new_coord, -*sigma);
        }
    }

    return new_coord;
}

fn get_initial(n: i64) -> Vec<i64> {
    let mut a_part: Vec<i64> = (0..(n-1)).map(|_| 0).collect();
    let b_part: Vec<i64> = (0..(n-1)).map(|_| -1).collect();

    a_part.extend(b_part.iter().cloned());

    return a_part;
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let initial = get_initial(args.braid_count_to_scale_to);
    let word  = vec![1, 3, 4, 3, -4, -3, -4, 2, 1, -2, -1, -2];

    let new_coord = dynnikov_word(&initial, &word);

    println!("initial: {:?}", initial);
    println!("Output: {:?}", new_coord);

    Ok(())
}
