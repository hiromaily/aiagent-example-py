use crate::openais::openai::OpenAI;
use async_trait::async_trait;

#[async_trait]
pub trait BasicUsecase {
    async fn call_prompt(&self, question: &str) -> Result<(), Box<dyn std::error::Error>>;
}

//
// BasicUsecase
//

pub struct BasicUsecaseImpl {
    pub openai_client: Box<dyn OpenAI>,
}

impl BasicUsecaseImpl {
    pub fn new(openai_client: Box<dyn OpenAI>) -> Self {
        BasicUsecaseImpl { openai_client }
    }
}

#[async_trait]
impl BasicUsecase for BasicUsecaseImpl {
    async fn call_prompt(&self, question: &str) -> Result<(), Box<dyn std::error::Error>> {
        let response = self.openai_client.call_prompt(question).await?;
        println!("Agent: {response}");
        Ok(())
    }
}

//
// Dummy BasicUsecase
//

pub struct DummyBasicUsecaseImpl {}

impl DummyBasicUsecaseImpl {
    pub fn new() -> Self {
        DummyBasicUsecaseImpl {}
    }
}

impl Default for DummyBasicUsecaseImpl {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BasicUsecase for DummyBasicUsecaseImpl {
    async fn call_prompt(&self, _question: &str) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}
