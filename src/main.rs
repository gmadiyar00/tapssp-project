mod retriever;
mod vector_db;
mod llm;
mod utils;
use anyhow::Result;
use llm::{LLM, LLMConfig};
use retriever::Retriever;
use std::env;
use std::io::Write;
use std::path::Path;
use utils::ensure_dir;

fn load_documents(retriever: &mut Retriever, docs_dir: &str) -> Result<()> {
    let texts = utils::load_text_files(docs_dir)?;
    for content in texts {
        if content.chars().count() > 2000 {
            let chunks = utils::split_into_chunks(&content, 2000);
            for chunk in chunks {
                retriever.add_to_knowledge_base(chunk)?;
            }
        } else {
            retriever.add_to_knowledge_base(content)?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let config = LLMConfig::default();
    let mut llm: Option<LLM> = None;

    let data_dir = Path::new("data");
    if let Err(e) = ensure_dir(data_dir) {
        eprintln!("Warning: failed to create data dir: {}", e);
    }

    let db_path = data_dir.join("vector_db.bin");
    let mut retriever = match Retriever::load_from_file(&db_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Warning: failed to load persisted DB: {}\nStarting with empty DB.", e);
            Retriever::new()
        }
    };
    let args: Vec<String> = env::args().collect();
    let mut mode_retrieve = false;
    let mut retrieve_query = String::new();
    let mut retrieve_topk: usize = 3;
    let mut docs_dir = "docs".to_string();

    if args.len() > 1 {
        if args[1] == "retrieve" {
            mode_retrieve = true;
            if args.len() > 2 {
                retrieve_query = args[2].clone();
            } else {
                eprintln!("Usage: {} retrieve \"your query\" [--topk N] [--docs DIR]", args[0]);
                return Ok(());
            }
            // rudimentary flag parsing
            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--topk" => {
                        if i + 1 < args.len() {
                            if let Ok(n) = args[i + 1].parse::<usize>() {
                                retrieve_topk = n;
                            }
                            i += 2;
                        } else { break; }
                    }
                    "--docs" => {
                        if i + 1 < args.len() { docs_dir = args[i + 1].clone(); }
                        i += 2;
                    }
                    _ => { i += 1; }
                }
            }
        } else {
            docs_dir = args[1].clone();
        }
    }

    println!("Loading documents from '{}'...", docs_dir);
    if let Err(e) = load_documents(&mut retriever, &docs_dir) {
        eprintln!("Warning: Failed to load documents: {}", e);
    } else {
        // finalize indexing once after bulk add (computes IDF and real embeddings)
        retriever.finalize_indexing();
        if let Err(e) = retriever.save_to_file(&db_path) {
            eprintln!("Warning: failed to save vector DB: {}", e);
        }
    }

    if mode_retrieve {
        // retrieve-only mode: print top-k chunks and exit
        let results = retriever.retrieve(&retrieve_query, retrieve_topk);
        println!("Top {} results for query: \"{}\"", retrieve_topk, retrieve_query);
        for (i, (score, text)) in results.iter().enumerate() {
            println!("--- result {} (score: {:.4}) ---\n{}\n", i + 1, score, text);
        }
        return Ok(());
    }

    println!("RAG System initialized! Enter your questions (Ctrl+C to exit)");

    loop {
        let mut query = String::new();
        print!("> ");
        std::io::stdout().flush()?;

        if std::io::stdin().read_line(&mut query)? == 0 {
            break; // EOF (Ctrl+D)
        }

        let query = query.trim();
        if query.is_empty() { continue; }

        let relevant = retriever.retrieve(query, 3);

        println!("\nRetrieved {} chunks:", relevant.len());
        for (i, (score, chunk)) in relevant.iter().enumerate() {
            let preview = if chunk.len() > 300 { format!("{}...", &chunk[..300]) } else { chunk.clone() };
            println!("  [{}] score={:.4}\n{}\n", i + 1, score, preview);
        }

        print!("\nThinking...");
        std::io::stdout().flush()?;

        if llm.is_none() {
            println!("\nInitializing LLM (this may take a while)...");
            llm = Some(LLM::new(config.clone())?);
        }

        if let Some(ref model) = llm {
            let context_texts: Vec<String> = relevant.into_iter().map(|(_, t)| t).collect();
            match model.generate_response(query, context_texts) {
                Ok(response) => println!("\r{}\n", response),
                Err(e) => eprintln!("\rError: {}\n", e),
            }
        }
    }

    Ok(())
}