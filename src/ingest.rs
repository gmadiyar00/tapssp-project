use crate::vector_db::{smart_insert_content};
// use crate::whisper::whisper_decode;
use anyhow::Context;
use clap::ValueEnum;
use std::collections::HashMap;
use chrono::{Utc};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq)]
pub enum IngestType {
    PDF,
    Text,
}

pub async fn ingest_via_cli(content: &str) -> anyhow::Result<()> {
    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), "direct insert".to_string());

    smart_insert_content(
        &format!("Direct insert on {}", Utc::now().date_naive()),
        &content,
        metadata,
    )
    .await?;

    println!("Text is remembered");
    Ok(())
}

pub async fn ingest_via_pdf_file(path: PathBuf) -> anyhow::Result<()> {
    let display = path.display();
    let bytes = std::fs::read(path.clone()).unwrap();
    let out = pdf_extract::extract_text_from_mem(&bytes).unwrap();

    let file_name = path
        .file_name()
        .context("Unable to get file name")?
        .to_str()
        .context("Unable to convert file name to string")?;

    println!("Processing pdf from {}", display);

    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), file_name.to_string());

    let content = smart_insert_content(
        &format!("Contents of {:?}", file_name),
        &out,
        metadata,
    )
    .await?;

    println!("Memorized {}", content.title);

    Ok(())
}

pub async fn ingest_via_txt_file(path: PathBuf) -> anyhow::Result<()> {
    let display = path.display();
    let file_name = path
        .file_name()
        .context("Unable to get file name")?
        .to_str()
        .context("Unable to convert file name to string")?;
    let file = match File::open(&path) {
        Err(why) => panic!("couldn't open {}: {}", display, why),
        Ok(file) => file,
    };
    println!("Processing text file from {}", display);

    let reader = BufReader::new(file);
    // read all lines and create a single string with "\n" as separator
    let content = reader
        .lines()
        .map(|l| l.unwrap())
        .collect::<Vec<String>>()
        .join("\n");

    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), file_name.to_string());

    let content = smart_insert_content(
        &format!("Contents of {:?}", file_name),
        &content,
        metadata,
    )
    .await?;

    println!("Memorized {}", content.title);

    Ok(())
}