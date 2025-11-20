use anyhow::Result;
use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::collections::hash_map::DefaultHasher;
use bincode;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub term_freq: HashMap<String, f32>,
    pub content_hash: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VectorDB {
    documents: HashMap<String, Document>,
    postings: HashMap<String, Vec<(String, f32)>>,
    term_doc_freq: HashMap<String, usize>,
    doc_hashes: HashSet<String>,
    doc_count: usize,
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            documents: HashMap::new(),
            postings: HashMap::new(),
            term_doc_freq: HashMap::new(),
            doc_hashes: HashSet::new(),
            doc_count: 0,
        }
    }

    pub fn add_document(&mut self, content: String) -> Result<()> {
        // compute simple hash to dedupe (DefaultHasher)
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        let hash = format!("{:x}", hasher.finish());

        if self.doc_hashes.contains(&hash) {
            return Ok(());
        }

        let id = uuid::Uuid::new_v4().to_string();

        let tokens = Self::tokenize(&content);
        if tokens.is_empty() {
            return Ok(());
        }

        let mut counts: HashMap<String, f32> = HashMap::new();
        for t in tokens.iter() {
            *counts.entry(t.clone()).or_insert(0.0) += 1.0;
        }
        let tokens_count = tokens.len() as f32;
        for v in counts.values_mut() {
            *v /= tokens_count;
        }

        // update postings and df
        for term in counts.keys() {
            self.postings
                .entry(term.clone())
                .or_insert_with(Vec::new)
                .push((id.clone(), *counts.get(term).unwrap_or(&0.0)));
            *self.term_doc_freq.entry(term.clone()).or_insert(0) += 1;
        }

        let doc = Document {
            id: id.clone(),
            content,
            term_freq: counts,
            content_hash: hash.clone(),
        };

        self.documents.insert(id, doc);
        self.doc_hashes.insert(hash);
        self.doc_count = self.documents.len();

        Ok(())
    }

    pub fn finalize_indexing(&mut self) {
        // no-op for incremental implementation
    }

    pub fn search_similar(&self, query: &str, top_k: usize) -> Vec<(f32, String)> {
        let q_tokens = Self::tokenize(query);
        if q_tokens.is_empty() || self.doc_count == 0 {
            return Vec::new();
        }

        let mut q_tf: HashMap<String, f32> = HashMap::new();
        for t in q_tokens.iter() {
            *q_tf.entry(t.clone()).or_insert(0.0) += 1.0;
        }
        let q_len = q_tokens.len() as f32;
        for v in q_tf.values_mut() {
            *v /= q_len;
        }

        let mut q_weights: HashMap<String, f32> = HashMap::new();
        for (term, tf) in q_tf.iter() {
            let df = *self.term_doc_freq.get(term).unwrap_or(&0) as f32;
            let idf = (1.0 + (self.doc_count as f32) / (1.0 + df)).ln();
            q_weights.insert(term.clone(), tf * idf);
        }

        let q_norm = q_weights.values().map(|w| w * w).sum::<f32>().sqrt();
        if q_norm == 0.0 {
            return Vec::new();
        }

        let mut accum: HashMap<String, f32> = HashMap::new();
        for (term, q_w) in q_weights.iter() {
            if let Some(postings) = self.postings.get(term) {
                let df = *self.term_doc_freq.get(term).unwrap_or(&0) as f32;
                let idf = (1.0 + (self.doc_count as f32) / (1.0 + df)).ln();
                for (doc_id, doc_tf) in postings.iter() {
                    let contrib = (*q_w) * (doc_tf * idf);
                    *accum.entry(doc_id.clone()).or_insert(0.0) += contrib;
                }
            }
        }

        let mut results: Vec<(f32, String)> = Vec::new();
        for (doc_id, numer) in accum.into_iter() {
            if let Some(doc) = self.documents.get(&doc_id) {
                let mut doc_norm_sq = 0.0_f32;
                for (term, tf_doc) in doc.term_freq.iter() {
                    let df = *self.term_doc_freq.get(term).unwrap_or(&0) as f32;
                    let idf = (1.0 + (self.doc_count as f32) / (1.0 + df)).ln();
                    let w = tf_doc * idf;
                    doc_norm_sq += w * w;
                }
                let doc_norm = doc_norm_sq.sqrt();
                if doc_norm == 0.0 {
                    continue;
                }
                let score = numer / (q_norm * doc_norm);
                results.push((score, doc.content.clone()));
            }
        }

        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        results.into_iter().take(top_k).collect()
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        bincode::serialize_into(file, self)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let db: VectorDB = bincode::deserialize_from(file)?;
        Ok(db)
    }
    
    fn tokenize(text: &str) -> Vec<String> {
        lazy_static! {
            static ref STOP_WORDS: HashSet<&'static str> = {
                let words = vec![
                    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
                    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
                    "to", "was", "were", "will", "with"
                ];
                words.into_iter().collect()
            };
        }

            let text = text.to_lowercase();
        let re = Regex::new(r"[^\w\s]").unwrap();
        let text = re.replace_all(&text, " ");
        text.split_whitespace()
            .filter(|&token| !STOP_WORDS.contains(token))
            .map(|s| s.to_string())
            .collect()
    }
}