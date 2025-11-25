use clap::Parser;
mod cli;
mod vector_db;
mod embeddings;
mod llm;
mod ingest;
use ingest::{ ingest_via_pdf_file, ingest_via_txt_file, };
use crate::ingest::ingest_via_cli;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    vector_db::load_from_disk()?;

    let args = cli::Cli::parse();

    match args.command {
        cli::Commands::Ask { query } => {
            let references = vector_db::retrieve(&query).await?;

            let answer = llm::answer_with_context(&query, references).await?;

            println!("Answer: {}", answer);
        }

        cli::Commands::Remember { content } => {
            ingest_via_cli(&content).await?;
        }

        cli::Commands::Upload { content_type, path } => match content_type {
            ingest::IngestType::Text => {
                ingest_via_txt_file(path).await?;
            }
            ingest::IngestType::PDF => {
                ingest_via_pdf_file(path).await?;
            }
        },
    }

    Ok(())
}