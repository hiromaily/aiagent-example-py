use crate::openais::openai::OpenAI;
use async_trait::async_trait;

#[async_trait]
pub trait PromptUsecase {
    async fn zero_shot(&self) -> Result<(), Box<dyn std::error::Error>>;
    async fn few_shot(&self) -> Result<(), Box<dyn std::error::Error>>;
}

//
// PromptUsecase
//

pub struct PromptUsecaseImpl {
    pub openai_client: Box<dyn OpenAI>,
}

impl PromptUsecaseImpl {
    pub fn new(openai_client: Box<dyn OpenAI>) -> Self {
        PromptUsecaseImpl { openai_client }
    }
}

#[async_trait]
impl PromptUsecase for PromptUsecaseImpl {
    async fn zero_shot(&self) -> Result<(), Box<dyn std::error::Error>> {
        let prompt = r#"
        "What are the top 10 Python libraries for AI?"
        "#;

        let response = self.openai_client.call_prompt(prompt).await?;
        println!("Agent: {response}");
        Ok(())
    }

    async fn few_shot(&self) -> Result<(), Box<dyn std::error::Error>> {
        let prompt = r#"
        Please classify the emotion (positive or negative) of the following sentences:

        Sentence: "This movie was great!"
        Emotion: Positive

        Sentence: "The service was very slow and I was dissatisfied."
        Emotion: Negative

        Sentence: "The quality was not good for the price."
        Emotion: Negative

        Sentence: "The staff were polite and I was able to shop comfortably."
        Emotion: Positive

        Sentence: "The wait time was too long and I was tired."
        Emotion:
        "#;

        let response = self.openai_client.call_prompt(prompt).await?;
        println!("Agent: {response}");
        Ok(())
    }
}

//
// Dummy PromptUsecase
//

pub struct DummyPromptUsecaseImpl {}

impl DummyPromptUsecaseImpl {
    pub fn new() -> Self {
        DummyPromptUsecaseImpl {}
    }
}

impl Default for DummyPromptUsecaseImpl {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PromptUsecase for DummyPromptUsecaseImpl {
    async fn zero_shot(&self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
    async fn few_shot(&self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}
