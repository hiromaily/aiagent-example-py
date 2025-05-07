use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(author, version, about)]
pub struct App {
    /// tool
    #[arg(short('t'), default_value = "openai")]
    pub tool: String,

    /// model path
    #[arg(short('m'), default_value = "gpt-4o-mini")]
    pub model: String,

    /// embedding model
    #[arg(short('e'), default_value = "text-embedding-ada-002")]
    pub embedding_model: String,

    #[command(subcommand)]
    pub command: SubCommand,
}

#[derive(Debug, Subcommand)]
pub enum SubCommand {
    Basic {
        #[clap(long, short = 'q')]
        #[arg(required(true))]
        question: String,
    },
    Prompt {
        #[clap(long, short = 'm')]
        mode: String,
    },
}
