use dotenv::dotenv;
use rig_example::args;
use rig_example::openais::openai::{get_agent, OpenAI, OpenAIImpl};

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
    let agent = get_agent(&args_data.tool, &args_data.model).expect("Failed to get agent");
    let openai_client: Box<dyn OpenAI> = Box::new(OpenAIImpl::new(agent));
    let response = openai_client
        .call_prompt(&args_data.question)
        .await
        .expect("Failed to prompt");

    println!("Agent: {response}");
}
