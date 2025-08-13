use std::error::Error;
use clap::Parser;

mod utilities;
mod dynnikov;
mod args;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = args::Args::parse();

    // Change braid count to scale to if it wasn't defined.
    if args.braid_count_to_scale_to < -1 {
        args.braid_count_to_scale_to = args.braid_count;
    }

    let initial = utilities::get_initial(args.braid_count_to_scale_to);
    let word  = vec![1, 3, 4, 3, -4, -3, -4, 2, 1, -2, -1, -2];

    let new_coord = dynnikov::word_action(&initial, &word);

    println!("Output: {:?}", new_coord);

    Ok(())
}
