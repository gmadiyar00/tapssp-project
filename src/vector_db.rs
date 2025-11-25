use anyhow::anyhow;
use anyhow::{Error, Result};
use chrono::{DateTime, Utc};
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Mutex;
use uuid::Uuid;
use crate::embeddings::get_embeddings;

lazy_static! {
    static ref CONTENTS: Mutex<Vec<Content>> = Mutex::new(Vec::new());
    static ref VECTOR_INDEXES: Mutex<Vec<VectorIndex>> = Mutex::new(Vec::new());
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Id {
    pub id: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Content {
    pub id: Id,
    pub title: String,
    pub text: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VectorIndex {
    pub id: Id,
    pub content_id: Id,
    pub content_chunk: String,
    pub chunk_number: u16,
    pub metadata: HashMap<String, String>,
    pub vector: Vec<f32>,
    pub created_at: DateTime<Utc>,
}

#[derive(Serialize, Deserialize, Debug)]
struct PersistedData {
    contents: Vec<Content>,
    vector_indexes: Vec<VectorIndex>,
}

pub async fn insert_content(title: &str, text: &str) -> Result<Content, Error> {
    let id = Uuid::new_v4().to_string().replace('-', "");
    let content = Content {
        id: Id { id: id.clone() },
        title: title.to_string(),
        text: text.to_string(),
        created_at: Utc::now(),
    };
    CONTENTS.lock().unwrap().push(content.clone());
    save_to_disk()?;
    Ok(content)
}

pub async fn insert_vector_index(
    content_id: Id,
    chunk_number: u16,
    content_chunk: &str,
    metadata: HashMap<String, String>,
) -> Result<VectorIndex, Error> {
    let chunk = content_chunk
        .chars()
        .collect::<String>()
        .trim()
        .to_string();

    if chunk.is_empty() {
        return Err(anyhow!("Content chunk is empty"));
    }

    let id = Uuid::new_v4().to_string().replace('-', "");

    // Compute embeddings for the chunk and store the vector.
    let emb = get_embeddings(&chunk).map_err(|e| anyhow!("Unable to compute embeddings: {}", e))?;
    let vector: Vec<f32> = match emb.to_vec1() {
        Ok(v) => v,
        Err(_) => {
            let mat = emb.to_vec2().map_err(|e| anyhow!("Unable to convert embeddings to Vec<f32>: {}", e))?;
            if mat.is_empty() {
                return Err(anyhow!("Embeddings conversion produced empty matrix"));
            }
            mat.into_iter().next().unwrap()
        }
    };

    let vector_index = VectorIndex {
        id: Id { id: id.clone() },
        content_id: content_id.clone(),
        content_chunk: chunk,
        chunk_number,
        metadata,
        vector,
        created_at: Utc::now(),
    };

    VECTOR_INDEXES.lock().unwrap().push(vector_index.clone());

    Ok(vector_index)
}

pub async fn smart_insert_content(title: &str, text: &str, metadata: HashMap<String, String>) -> Result<Content, Error> {
    let content = insert_content(title, text).await?;

    // Split by 300-character chunks
    let chunk_size = 300;
    let mut offset = 0;
    let chars: Vec<char> = text.chars().collect();

    while offset < chars.len() {
        let end = (offset + chunk_size).min(chars.len());
        let chunk: String = chars[offset..end].iter().collect();

        insert_vector_index(
            content.id.clone(),
            (offset / chunk_size) as u16,
            &chunk,
            metadata.clone()
        ).await?;

        offset = end;
    }

    save_to_disk()?;
    Ok(content)
}

pub async fn retrieve(query: &str) -> Result<Vec<VectorIndex>, Error> {
    // Compute embedding for query (robust conversion)
    let qemb = get_embeddings(query).map_err(|e| anyhow!("Failed to embed query: {}", e))?;
    let query_emb: Vec<f32> = match qemb.to_vec1() {
        Ok(v) => v,
        Err(_) => {
            let mat = qemb.to_vec2().map_err(|e| anyhow!("Embedding vector convert error: {}", e))?;
            if mat.is_empty() {
                return Err(anyhow!("Embedding vector convert error: empty matrix"));
            }
            mat.into_iter().next().unwrap()
        }
    };

    let indexes = VECTOR_INDEXES.lock().unwrap();
    let mut scored: Vec<(f32, VectorIndex)> = indexes
        .iter()
        .map(|v| {
            let score = cosine_similarity(&query_emb, &v.vector);
            (score, v.clone())
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Take top 4
    Ok(scored.into_iter().take(4).map(|x| x.1).collect())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

fn get_db_path() -> String {
    "data/vector_db.bin".to_string()
}

pub fn load_from_disk() -> Result<()> {
    let path = get_db_path();
    if !Path::new(&path).exists() {
        return Ok(()); // No file to load, that's fine
    }

    let data = fs::read(&path)?;
    let persisted: PersistedData = bincode::deserialize(&data)?;

    let mut contents = CONTENTS.lock().unwrap();
    let mut indexes = VECTOR_INDEXES.lock().unwrap();

    *contents = persisted.contents;
    *indexes = persisted.vector_indexes;

    Ok(())
}

pub fn save_to_disk() -> Result<()> {
    let contents = CONTENTS.lock().unwrap().clone();
    let vector_indexes = VECTOR_INDEXES.lock().unwrap().clone();

    let persisted = PersistedData {
        contents,
        vector_indexes,
    };

    let data = bincode::serialize(&persisted)?;
    fs::create_dir_all("data")?;
    fs::write(get_db_path(), data)?;

    Ok(())
}