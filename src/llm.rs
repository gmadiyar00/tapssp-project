use anyhow::{Result, anyhow};
use llama_rs::{
    Model, ModelParams, InferenceParams, InferenceSession,
    InferenceRequest, InferenceResponse, TokenId
};
use std::{path::PathBuf, sync::Arc};
use std::io::Write;

pub struct LLMConfig {
    pub model_path: Option<PathBuf>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            max_tokens: 1000,
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
        }
    }
}

pub struct LLM {
    model: Arc<Model>,
    config: LLMConfig,
}

impl LLM {
    pub fn new(mut config: LLMConfig) -> Result<Self> {
        // If model path not provided, download and use default model
        if config.model_path.is_none() {
            config.model_path = Some(Self::get_default_model()?);
        }

        let model_path = config.model_path.as_ref()
            .ok_or_else(|| anyhow!("Model path not set"))?;

        if !model_path.exists() {
            return Err(anyhow!("Model file not found at {:?}", model_path));
        }

        let model_params = ModelParams::default();
        let model = Model::load(&model_path, model_params)?;

        Ok(LLM {
            model: Arc::new(model),
            config,
        })
    }

    fn get_default_model() -> Result<PathBuf> {
        let models_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow!("Could not determine cache directory"))?
            .join("tapssp-project")
            .join("models");

        std::fs::create_dir_all(&models_dir)?;
        
        let model_path = models_dir.join("mistral-7b-instruct-v0.1.Q4_K_M.gguf");
        
        if !model_path.exists() {
            println!("Downloading Mistral 7B model...");
            // Download model from HuggingFace
            let url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf";
            let response = reqwest::blocking::get(url)?;
            let mut file = std::fs::File::create(&model_path)?;
            let mut content = std::io::Cursor::new(response.bytes()?);
            std::io::copy(&mut content, &mut file)?;
            println!("Model downloaded successfully!");
        }

        Ok(model_path)
    }

    pub fn generate_response(&self, query: &str, context: Vec<String>) -> Result<String> {
        if query.trim().is_empty() {
            return Err(anyhow!("Query cannot be empty"));
        }

        let prompt = self.construct_prompt(query, context);
        
        let inference_params = InferenceParams {
            n_threads: num_cpus::get(),  // Use all available CPU cores
            n_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            top_p: self.config.top_p,
            repeat_penalty: self.config.repeat_penalty,
            ..InferenceParams::default()
        };

        let mut session = InferenceSession::new(
            self.model.clone(),
            inference_params,
        )?;

        let mut response = String::new();
        let mut tokens = session.infer::<std::io::Stdout>(
            InferenceRequest::from_prompt(prompt),
            |r| match r {
                InferenceResponse::InferredToken(token) => {
                    response.push_str(&token);
                    Ok(())
                }
                InferenceResponse::EotToken => Ok(()),
            },
        )?;

        Ok(response)
    }

    fn construct_prompt(&self, query: &str, context: Vec<String>) -> String {
        let context_str = if context.is_empty() {
            String::new()
        } else {
            format!(
                "Using the following context to answer the question:\n\n{}\n\n",
                context.join("\n\n")
            )
        };

        format!(
            "<s>[INST] {context_str}Question: {query} [/INST]",
        )
    }
}