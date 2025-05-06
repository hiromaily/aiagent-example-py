use clap::Parser;
use std::env;

#[derive(Debug, Parser)]
#[command(author, version, about)]
/// Rust version of `echo`
pub struct Args {
    /// tool
    #[arg(required(true))]
    pub tool: String,

    /// model path
    #[arg(short('m'), default_value = "gpt-4o-mini")]
    pub model: String,

    /// embedding model
    #[arg(short('e'), default_value = "text-embedding-ada-002")]
    pub embedding_model: String,
}

#[allow(dead_code)]
pub fn print_parsed_args() {
    // command line arguments
    // by clap
    let args = Args::parse();
    println!(
        "tool:{}, model:{}, embedding_model:{}",
        args.tool, args.model, args.embedding_model,
    );

    // by std::env
    let args: Vec<String> = env::args().collect();
    println!("{:?}", args);
}

#[allow(dead_code)]
pub fn get_args() -> Args {
    Args::parse()
}
