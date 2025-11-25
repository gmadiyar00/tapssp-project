use anyhow::{anyhow, Result};
use std::env;
use std::process::Command;
use std::time::Duration;
use tokio::task;
use tokio::time;

use crate::vector_db::VectorIndex;

#[derive(Clone, Debug)]
pub struct LLMConfig {
    pub llama_bin: String,
    pub model_path: String,
    pub threads: usize,
    pub max_tokens: usize,
    pub timeout_secs: u64,
}

impl Default for LLMConfig {
    fn default() -> Self {
        let model_path = env::var("TAPSSP_PHI_PATH")
            .unwrap_or_else(|_| "/Users/gulbanumadiyarova/Downloads/Textbooks/tapssp-project/llama.cpp/models/phi-2.Q2_K.gguf".into());

        let default_bin = "/opt/homebrew/bin/llama-simple".to_string();
        let bin = env::var("TAPSSP_LLAMA_BIN").unwrap_or(default_bin);

        let threads = env::var("TAPSSP_LLAMA_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);

        let max_tokens = env::var("TAPSSP_LLAMA_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256);

        let timeout_secs = env::var("TAPSSP_LLAMA_TIMEOUT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60);

        Self {
            llama_bin: bin,
            model_path,
            threads,
            max_tokens,
            timeout_secs,
        }
    }
}

#[derive(Clone)]
pub struct LLM {
    cfg: LLMConfig,
}

impl LLM {
    pub fn new(cfg: LLMConfig) -> Self {
        Self { cfg }
    }

    fn build_prompt(&self, query: &str, references: Option<&[VectorIndex]>) -> String {
        if let Some(refs) = references {
            if !refs.is_empty() {
                let mut ctx = String::new();
                for (i, r) in refs.iter().enumerate() {
                    ctx.push_str(&format!("Reference {}:\n{}\n\n", i + 1, r.content_chunk));
                }
                return format!(
                    "You are a helpful assistant. Use the context below to answer concisely.\n\nContext:\n{}\nQuestion: {}\nAnswer:",
                    ctx, query
                );
            }
        }
        format!("Question: {}\nAnswer:", query.trim())
    }

    pub async fn generate_with_context(
        &self,
        query: &str,
        references: Option<&[VectorIndex]>
    ) -> Result<String> {
        let prompt = self.build_prompt(query, references);
        self.generate_via_subprocess(&prompt).await
    }

    async fn generate_via_subprocess(&self, prompt: &str) -> Result<String> {
        let bin = &self.cfg.llama_bin;
        let model = &self.cfg.model_path;

        if !std::path::Path::new(bin).exists() {
            return Err(anyhow!("llama binary not found: {} (set TAPSSP_LLAMA_BIN)", bin));
        }
        if !std::path::Path::new(model).exists() {
            return Err(anyhow!("GGUF model not found: {} (set TAPSSP_PHI_PATH)", model));
        }

        let bin = bin.clone();
        let model = model.clone();
        let prompt_owned = prompt.to_string();
        let threads = self.cfg.threads;
        let n_str = self.cfg.max_tokens.to_string();
        let timeout_secs = self.cfg.timeout_secs;

        // Run blocking
        let handle = task::spawn_blocking(move || {
            let out = Command::new(&bin)
                .arg("-m").arg(&model)
                .arg("-n").arg(&n_str)
                .arg("--threads").arg(threads.to_string())
                .arg("--color")
                .arg(&prompt_owned)
                .output()
                .map_err(|e| anyhow!("failed to spawn llama subprocess: {}", e))?;

            if !out.status.success() {
                return Err(anyhow!(
                    "llama subprocess failed: status={} stderr={}",
                    out.status,
                    String::from_utf8_lossy(&out.stderr)
                ));
            }
            Ok(String::from_utf8_lossy(&out.stdout).to_string())
        });

        let timeout_dur = Duration::from_secs(timeout_secs);
        let res = time::timeout(timeout_dur, handle)
            .await
            .map_err(|_| anyhow!("llama subprocess timed out after {}s", timeout_secs))?
            .map_err(|e| anyhow!("llama spawn error: {}", e))??;

        Ok(strip_prompt_from_llama_output(prompt, &res))
    }
}

fn strip_prompt_from_llama_output(prompt: &str, stdout: &str) -> String {
    if let Some(idx) = stdout.rfind("Answer:") {
        return stdout[idx + "Answer:".len()..].trim().to_string();
    }
    if let Some(idx) = stdout.rfind("Assistant:") {
        return stdout[idx + "Assistant:".len()..].trim().to_string();
    }

    stdout.replacen(prompt, "", 1).trim().to_string();

    let tail = if prompt.len() > 80 {
        &prompt[prompt.len() - 80..]
    } else {
        prompt
    };

    if let Some(idx) = stdout.rfind(tail) {
        return stdout[idx + tail.len()..].trim().to_string();
    }

    stdout.trim().to_string()
}

pub async fn answer_with_context(
    query: &str,
    references: Vec<VectorIndex>
) -> Result<String> {
    let cfg = LLMConfig::default();
    let client = LLM::new(cfg);

    let refs_opt =
        if references.is_empty() { None } else { Some(references.as_slice()) };

    client.generate_with_context(query, refs_opt).await
}
