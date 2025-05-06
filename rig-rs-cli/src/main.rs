use dotenv::dotenv;
use rig::{completion::Prompt, providers::openai};

use rig_example::args;

#[tokio::main]
async fn main() {
    dotenv().ok();

    let args_data = args::get_args();
    // debug
    //args::print_parsed_args();
    println!("Tool: {}", args_data.tool);
    println!("Model: {}", args_data.model);
    println!("Embedding Model: {}", args_data.embedding_model);

    // Create OpenAI client and model
    // This requires the `OPENAI_API_KEY` environment variable to be set.
    println!("Tool: {}", args_data.tool);
    let openai_client = if args_data.tool == "openai" {
        openai::Client::from_env()
    } else if args_data.tool == "ollama" {
        openai::Client::from_url("ollama", "http://localhost:11434/v1")
    } else if args_data.tool == "lmstudio" {
        openai::Client::from_url("lm-studio", "http://localhost:1234/v1")
    } else {
        panic!("Unsupported tool: {}", args_data.tool);
    };
    let agent = openai_client.agent(&args_data.model).build();

    // Prompt the model and print its response
    let response = agent
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt");

    println!("Agent: {response}");
}
