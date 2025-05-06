use clap::Parser;
use dotenv::dotenv;
use rig_example::args::{App, SubCommand};
use rig_example::openais::openai::{get_agent, OpenAI, OpenAIImpl};

#[tokio::main]
async fn main() {
    dotenv().ok();

    let cli = App::parse();

    println!("Question: {}", cli.question);
    println!("Tool: {}", cli.tool);
    println!("Model: {}", cli.model);
    println!("Embedding Model: {}", cli.embedding_model);

    // Create OpenAI client and model
    let agent = get_agent(&cli.tool, &cli.model).expect("Failed to get agent");
    let openai_client: Box<dyn OpenAI> = Box::new(OpenAIImpl::new(agent));

    match &cli.command {
        SubCommand::Basic => {
            println!("TODO: run basic command");
            // Call
            let response = openai_client
                .call_prompt(&cli.question)
                .await
                .expect("Failed to prompt");
            println!("Agent: {response}");
        }
        SubCommand::Prompt { opt } => {
            println!("TODO: run prompt command:: {}", opt);
        }
    }
}
