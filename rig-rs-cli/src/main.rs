use dotenv::dotenv;
use rig::completion::Prompt;

use rig_example::args;
use rig_example::openai::get_agent;

#[tokio::main]
async fn main() {
    dotenv().ok();

    let args_data = args::get_args();
    // debug
    //args::print_parsed_args();
    println!("Question: {}", args_data.question);
    println!("Tool: {}", args_data.tool);
    println!("Model: {}", args_data.model);
    println!("Embedding Model: {}", args_data.embedding_model);

    // Create OpenAI client and model
    // This requires the `OPENAI_API_KEY` environment variable to be set.
    println!("Tool: {}", args_data.tool);
    let agent = get_agent(&args_data).expect("Failed to get agent");

    // Prompt the model and print its response
    let response = agent
        .prompt(&args_data.question)
        .await
        .expect("Failed to prompt");

    println!("Agent: {response}");
}
