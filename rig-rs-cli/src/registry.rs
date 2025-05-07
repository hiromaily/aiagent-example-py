use crate::openais::openai::{get_agent, OpenAI, OpenAIImpl};
use crate::usecases::basic::{BasicUsecase, BasicUsecaseImpl};

pub struct Registry {
    tool: String,
    model: String,
    #[allow(dead_code)]
    embedding_model: String,
}

impl Registry {
    pub fn new(tool: String, model: String, embedding_model: String) -> Registry {
        Registry {
            tool,
            model,
            embedding_model,
        }
    }

    fn build_agent(&self) -> Box<dyn OpenAI> {
        // Create OpenAI client and model
        let agent = get_agent(&self.tool, &self.model).expect("Failed to get agent");
        Box::new(OpenAIImpl::new(agent))
    }

    // Get basic usecase
    pub fn get_basic_usecase(&self) -> Box<dyn BasicUsecase> {
        // build basic usecase
        let openai_client = self.build_agent();
        Box::new(BasicUsecaseImpl::new(openai_client))
    }
}
