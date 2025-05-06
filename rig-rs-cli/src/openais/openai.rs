use async_trait::async_trait;
use rig::completion::Prompt;
use rig::{agent, providers::openai};

#[async_trait]
pub trait OpenAI {
    async fn call_prompt(&self, question: &str) -> Result<String, Box<dyn std::error::Error>>;
}

//
// OpenAI
//

pub struct OpenAIImpl {
    pub agent: agent::Agent<openai::CompletionModel>,
}

impl OpenAIImpl {
    pub fn new(agent: agent::Agent<openai::CompletionModel>) -> Self {
        OpenAIImpl { agent }
    }
}

#[async_trait]
impl OpenAI for OpenAIImpl {
    async fn call_prompt(&self, question: &str) -> Result<String, Box<dyn std::error::Error>> {
        // Prompt the model and return its response
        let response = self.agent.prompt(question).await?;
        Ok(response)
    }
}

//
// Dummy OpenAI
//

pub struct DummyOpenAIImpl {}

impl DummyOpenAIImpl {
    pub fn new() -> Self {
        DummyOpenAIImpl {}
    }
}

impl Default for DummyOpenAIImpl {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl OpenAI for DummyOpenAIImpl {
    async fn call_prompt(&self, question: &str) -> Result<String, Box<dyn std::error::Error>> {
        Ok(format!("Dummy response to: {}", question))
    }
}

//
// Utility
//

pub fn get_agent(tool: &str, model: &str) -> Result<agent::Agent<openai::CompletionModel>, String> {
    let openai_client = match tool {
        // This requires the `OPENAI_API_KEY` environment variable to be set.
        "openai" => openai::Client::from_env(),
        "ollama" => openai::Client::from_url("ollama", "http://localhost:11434/v1"),
        "lmstudio" => openai::Client::from_url("lm-studio", "http://localhost:1234/v1"),
        _ => return Err(format!("Unsupported tool: {}", tool)),
    };

    let agent = openai_client.agent(model).build();
    Ok(agent)
}
