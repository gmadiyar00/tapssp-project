use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use lazy_static::lazy_static;
use std::env;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::{PaddingParams, Tokenizer};

lazy_static! {
    /// Lazily initialized MiniLM embeddings model
    pub static ref EMB_MODEL: Mutex<Option<(BertModel, Tokenizer)>> = Mutex::new(None);
}

/// Load the MiniLM embeddings model from local files
pub fn load_model() -> Result<(BertModel, Tokenizer)> {
    let path = env::var("TAPSSP_EMBEDDINGS_PATH")
        .unwrap_or_else(|_| "/Users/gulbanumadiyarova/Downloads/Textbooks/tapssp-project/all-MiniLM-L6-v2".into());

    let config_path = env::var("TAPSSP_EMBEDDINGS_CONFIG")
        .unwrap_or_else(|_| format!("{}/config.json", path));
    let tokenizer_path = env::var("TAPSSP_EMBEDDINGS_TOKENIZER")
        .unwrap_or_else(|_| format!("{}/tokenizer.json", path));
    let weights_path = env::var("TAPSSP_EMBEDDINGS_WEIGHTS")
        .unwrap_or_else(|_| format!("{}/pytorch_model.bin", path));

    if !Path::new(&config_path).exists() {
        return Err(anyhow::anyhow!("Embeddings config not found at {}", config_path));
    }
    if !Path::new(&tokenizer_path).exists() {
        return Err(anyhow::anyhow!("Embeddings tokenizer not found at {}", tokenizer_path));
    }
    if !Path::new(&weights_path).exists() {
        return Err(anyhow::anyhow!("Embeddings weights not found at {}", weights_path));
    }

    let config_str = std::fs::read_to_string(config_path)?;
    let config: Config = serde_json::from_str(&config_str)?;

    let mut tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Ensure padding is set for batch longest
    if tokenizer.get_padding().is_none() {
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }

    let vb = VarBuilder::from_pth(&weights_path, DType::F32, &Device::Cpu)?;
    let model = BertModel::load(vb, &config)?;

    Ok((model, tokenizer))
}

/// Get embeddings for a sentence using MiniLM
pub fn get_embeddings(sentence: &str) -> Result<Tensor> {
    // Initialize model if needed
    {
        let mut guard = EMB_MODEL.lock().unwrap();
        if guard.is_none() {
            let loaded = load_model()?;
            *guard = Some(loaded);
        }
    }

    let guard = EMB_MODEL.lock().unwrap();
    let (model, tokenizer) = guard.as_ref().ok_or_else(|| anyhow::anyhow!("Embeddings model not loaded"))?;

    // Tokenize input
    let tokens = tokenizer
        .encode_batch(vec![sentence.to_string()], true)
        .map_err(|e| anyhow::anyhow!("Tokenizer encode_batch failed: {}", e))?;

    // Token IDs as i64 tensor
    let ids: Vec<i64> = tokens[0].get_ids().iter().map(|&v| v as i64).collect();
    let token_ids = Tensor::new(ids.as_slice(), &Device::Cpu)?.unsqueeze(0)?;

    // Attention mask as i64 -> convert to F32
    let mask: Vec<i64> = tokens[0].get_attention_mask().iter().map(|&v| v as i64).collect();
    let attention = Tensor::new(mask.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
    let attention_f32 = attention.to_dtype(DType::F32)?;

    // Token type IDs (all zeros)
    let token_type_ids = token_ids.zeros_like()?;

    // Forward pass
    let embeddings = model.forward(&token_ids, &token_type_ids, None)?;

    // Mean pooling with attention mask
    let mask_expanded = attention_f32.unsqueeze(2)?;
    let masked = embeddings.broadcast_mul(&mask_expanded)?;
    let sum_embeddings = masked.sum(1)?;
    let mask_sum = attention_f32.sum(1)?.unsqueeze(1)?;
    let pooled = sum_embeddings.broadcast_div(&mask_sum)?;

    // L2 normalization
    let sqr = pooled.sqr()?;
    let sum_keepdim = sqr.sum_keepdim(1)?;
    let sqrt = sum_keepdim.sqrt()?;
    let normalized = pooled.broadcast_div(&sqrt)?;

    Ok(normalized)
}
