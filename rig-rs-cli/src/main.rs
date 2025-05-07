use clap::Parser;
use dotenv::dotenv;
use rig_example::args::{App, SubCommand};
use rig_example::registry::Registry;

#[tokio::main]
async fn main() {
    dotenv().ok();

    let cli = App::parse();

    println!("Tool: {}", cli.tool);
    println!("Model: {}", cli.model);
    println!("Embedding Model: {}", cli.embedding_model);

    match &cli.command {
        SubCommand::Basic { question } => {
            println!("run basic command: {}", question);
            let basic_usecase = Registry::new(
                cli.tool.clone(),
                cli.model.clone(),
                cli.embedding_model.clone(),
            )
            .get_basic_usecase();

            // Call
            basic_usecase
                .call_prompt(question)
                .await
                .expect("Failed to call call_prompt");
        }
        SubCommand::Prompt { mode } => {
            println!("run prompt command");
            let prompt_usecase = Registry::new(
                cli.tool.clone(),
                cli.model.clone(),
                cli.embedding_model.clone(),
            )
            .get_prompt_usecase();

            // Call
            match mode.as_str() {
                "zero-shot" => {
                    println!("run zero_shot command");
                    prompt_usecase
                        .zero_shot()
                        .await
                        .expect("Failed to call zero_shot");
                }
                "few-shot" => {
                    println!("run few_shot command");
                    prompt_usecase
                        .few_shot()
                        .await
                        .expect("Failed to call few_shot");
                }
                _ => {
                    println!("Invalid mode: {}", mode);
                    return;
                }
            }
        }
    }
}
