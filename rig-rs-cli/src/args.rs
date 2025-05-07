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
        #[clap(long, short = 'o')]
        opt: bool,
    },
}

// pub fn get_args() {
//     let cli = App::parse();

//     println!("Question: {}", cli.question);
//     println!("Tool: {}", cli.tool);
//     println!("Model: {}", cli.model);
//     println!("Embedding Model: {}", cli.embedding_model);

//     match &cli.command {
//         SubCommand::Basic => {
//             println!("TODO: run basic command");
//         }
//         SubCommand::Prompt { opt } => {
//             println!("TODO: run prompt command:: {}", opt);
//         }
//     }
// }
