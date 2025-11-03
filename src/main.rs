mod retriever;
mod vector_db;
mod llm;
mod utils;

use anyhow::Result;
use llm::{LLM, LLMConfig};
use retriever::Retriever;
use std::{env, fs};

async fn load_documents(retriever: &mut Retriever, docs_dir: &str) -> Result<()> {
    for entry in fs::read_dir(docs_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().map_or(false, |ext| ext == "txt") {
            let content = fs::read_to_string(path)?;
            retriever.add_to_knowledge_base(content)?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    // Initialize LLM with default config (will download model if needed)
    let config = LLMConfig::default();
    println!("Initializing LLM (first run will download the model)...");
    let llm = LLM::new(config)?;
    
    let mut retriever = Retriever::new();

    // Load documents from a directory
    let docs_dir = env::args()
        .nth(1)
        .unwrap_or_else(|| "docs".to_string());

    println!("Loading documents from '{}'...", docs_dir);
    if let Err(e) = load_documents(&mut retriever, &docs_dir) {
        eprintln!("Warning: Failed to load documents: {}", e);
    }

    println!("RAG System initialized! Enter your questions (Ctrl+C to exit)");
    println!("Using Mistral 7B for local inference - no API key needed!");

    // Interactive query loop
    loop {
        let mut query = String::new();
        print!("> ");
        std::io::Write::flush(&mut std::io::stdout())?;
        
        if std::io::stdin().read_line(&mut query)? == 0 {
            break; // EOF (Ctrl+D)
        }

        let query = query.trim();
        if query.is_empty() {
            continue;
        }

        // Retrieve relevant context
        let relevant_chunks = retriever.retrieve(query, 3);
        
        // Generate and print response
        print!("\nThinking...");
        std::io::stdout().flush()?;
        match llm.generate_response(query, relevant_chunks) {
            Ok(response) => println!("\r{}\n", response),
            Err(e) => eprintln!("\rError: {}\n", e),
        }
    }

    Ok(())
}
