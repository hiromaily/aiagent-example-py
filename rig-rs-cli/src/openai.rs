use crate::args;
use rig::{agent, providers::openai};

pub fn get_agent(args_data: &args::Args) -> Result<agent::Agent<openai::CompletionModel>, String> {
    let openai_client = match args_data.tool.as_str() {
        "openai" => openai::Client::from_env(),
        "ollama" => openai::Client::from_url("ollama", "http://localhost:11434/v1"),
        "lmstudio" => openai::Client::from_url("lm-studio", "http://localhost:1234/v1"),
        _ => return Err(format!("Unsupported tool: {}", args_data.tool)),
    };

    let agent = openai_client.agent(&args_data.model).build();
    Ok(agent)
}
