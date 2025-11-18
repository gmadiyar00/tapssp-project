use anyhow::{anyhow, Result};
use std::process::{Command, Stdio};
use std::path::Path;
use std::env;
pub struct LLMConfig {
    pub llama_bin: String,
    pub model_path: String,
    pub max_tokens: usize,
    pub threads: Option<usize>,
    /// Extra args to pass to the llama.cpp binary (e.g. ["--device","none"]).
    /// Can be set via the `TAPSSP_LLAMA_ARGS` env var (space-separated).
    pub args: Vec<String>,
    /// Context size (max tokens) for the model; used to trim context to fit the model.
    pub ctx_size: usize,
}

impl Default for LLMConfig {
    fn default() -> Self {
        let home = env::var("HOME").unwrap_or_else(|_| "~".to_string());
        let default_model = format!("{}/.cache/tapssp-project/models/llama-3.1-8b-instruct.Q4_K_M.gguf", home);
        Self {
            // keep default as a relative path but LLM::new will try env/PATH fallbacks
            llama_bin: "./llama.cpp/main".to_string(),
            model_path: default_model,
            max_tokens: 256,
            threads: None,
            args: Vec::new(),
            ctx_size: 4096,
        }
    }
}

pub struct LLM {
    cfg: LLMConfig,
}

impl LLM {
    pub fn new(cfg: LLMConfig) -> Result<Self> {
        // Resolve llama binary: check provided path, then env var, then PATH and common locations
        let mut tried_bins = Vec::new();
        let resolved_bin = find_executable(&cfg.llama_bin, &mut tried_bins);

        let mut tried_models = Vec::new();
        let resolved_model = find_model(&cfg.model_path, &mut tried_models);

        match (resolved_bin, resolved_model) {
            (Some(bin), Some(model)) => {
                let mut cfg = cfg;
                cfg.llama_bin = bin;
                cfg.model_path = model;

                // If the TAPSSP_LLAMA_ARGS env var is set, parse it into args (simple whitespace split).
                if let Ok(extra) = env::var("TAPSSP_LLAMA_ARGS") {
                    let parts: Vec<String> = extra
                        .split_whitespace()
                        .map(|s| s.to_string())
                        .collect();
                    if !parts.is_empty() {
                        cfg.args = parts;
                    }
                }

                Ok(LLM { cfg })
            }
            (None, None) => Err(anyhow!("llama binary not found (tried: {}) and model file not found (tried: {})", tried_bins.join(", "), tried_models.join(", "))),
            (None, _) => Err(anyhow!("llama binary not found. Tried: {}", tried_bins.join(", "))),
            (_, None) => Err(anyhow!("model file not found. Tried: {}", tried_models.join(", "))),
        }
    }

    fn construct_prompt(&self, query: &str, context: Vec<String>) -> String {
        // If no context, just ask the question.
        if context.is_empty() {
            return format!("Question: {}\n\nAnswer:", query);
        }

        // Estimate tokens by characters (approx 4 chars per token). This is coarse
        // but prevents overlong prompts when a true tokenizer isn't available.
        let avg_chars_per_token = 4.0_f32;
        let mut allowed_tokens = if self.cfg.ctx_size > self.cfg.max_tokens {
            self.cfg.ctx_size - self.cfg.max_tokens
        } else {
            // if misconfigured, leave a small room
            64
        };

        // reserve some tokens for prompt overhead
        if allowed_tokens > 64 { allowed_tokens -= 64; }

        let mut included = Vec::new();
        let mut used_tokens: usize = 0;

        for chunk in context.iter() {
            let est = ((chunk.chars().count() as f32) / avg_chars_per_token).ceil() as usize;
            if used_tokens + est > allowed_tokens {
                break; // stop adding more chunks
            }
            included.push(chunk.clone());
            used_tokens += est;
        }

        let context_str = included.join("\n\n");
        format!(
            "Using the following context to answer the question:\n\n{}\n\nQuestion: {}\n\nAnswer:",
            context_str,
            query
        )
    }

    pub fn generate_response(&self, query: &str, context: Vec<String>) -> Result<String> {
        if query.trim().is_empty() {
            return Err(anyhow!("Query cannot be empty"));
        }

        let prompt = self.construct_prompt(query, context);

        let mut cmd = Command::new(&self.cfg.llama_bin);
        cmd.arg("-m").arg(&self.cfg.model_path);
        cmd.arg("-p").arg(&prompt);
        cmd.arg("-n").arg(self.cfg.max_tokens.to_string());

        if let Some(t) = self.cfg.threads {
            cmd.arg("-t").arg(t.to_string());
        }

        // Append any extra args the user provided (e.g. --device none)
        for a in &self.cfg.args {
            cmd.arg(a);
        }

        let output = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("llama binary failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(stdout)
    }
}

fn expand_home(path: &str) -> String {
    if path.starts_with("~") {
        if let Ok(home) = env::var("HOME") {
            return path.replacen("~", &home, 1);
        }
    }
    path.to_string()
}

fn find_executable(provided: &str, tried: &mut Vec<String>) -> Option<String> {
    // 1) Provided path
    let p = expand_home(provided);
    tried.push(p.clone());
    if Path::new(&p).exists() {
        return Some(p);
    }

    // 2) env var TAPSSP_LLAMA_BIN
    if let Ok(env_bin) = env::var("TAPSSP_LLAMA_BIN") {
        let env_bin_exp = expand_home(&env_bin);
        tried.push(env_bin_exp.clone());
        if Path::new(&env_bin_exp).exists() {
            return Some(env_bin_exp);
        }
    }

    // 3) Search PATH for common binary names
    if let Ok(path_var) = env::var("PATH") {
        for dir in path_var.split(':') {
            for name in ["main", "llama", "llama-main"] {
                let cand = format!("{}/{}", dir, name);
                tried.push(cand.clone());
                if Path::new(&cand).exists() {
                    return Some(cand);
                }
            }
        }
    }

    // 4) Common project locations
    for cand in ["./llama.cpp/main", "./build/main", "./main", "./llama"] {
        let cand_exp = expand_home(cand);
        tried.push(cand_exp.clone());
        if Path::new(&cand_exp).exists() {
            return Some(cand_exp);
        }
    }

    None
}

fn find_model(provided: &str, tried: &mut Vec<String>) -> Option<String> {
    // 1) provided
    let p = expand_home(provided);
    tried.push(p.clone());
    if Path::new(&p).exists() {
        return Some(p);
    }

    // 2) env var
    if let Ok(env_m) = env::var("TAPSSP_MODEL_PATH") {
        let m = expand_home(&env_m);
        tried.push(m.clone());
        if Path::new(&m).exists() {
            return Some(m);
        }
    }

    // 3) common cache location
    if let Ok(home) = env::var("HOME") {
        let cand = format!("{}/.cache/tapssp-project/models/llama-3.1-8b-instruct.Q4_K_M.gguf", home);
        tried.push(cand.clone());
        if Path::new(&cand).exists() {
            return Some(cand);
        }
    }

    None
}
