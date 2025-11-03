use anyhow::Result;
use ndarray::Array1;
use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;
use lazy_static::lazy_static;

#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub embedding: Array1<f32>,
}

pub struct VectorDB {
    documents: HashMap<String, Document>,
    vocabulary: FxHashSet<String>,
    idf_values: FxHashMap<String, f32>,
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            documents: HashMap::new(),
            vocabulary: FxHashSet::default(),
            idf_values: FxHashMap::default(),
        }
    }

    pub fn add_document(&mut self, content: String) -> Result<()> {
        let id = uuid::Uuid::new_v4().to_string();
        let tokens = self.tokenize(&content);
        
        // Update vocabulary and document frequencies
        for token in &tokens {
            self.vocabulary.insert(token.clone());
        }
        
        // Calculate TF-IDF embedding
        let embedding = self.calculate_tfidf(&tokens);
        
        let document = Document {
            id: id.clone(),
            content,
            embedding,
        };
        
        self.documents.insert(id, document);
        self.update_idf_values();
        Ok(())
    }

    pub fn search_similar(&self, query: &str, top_k: usize) -> Vec<&Document> {
        let tokens = self.tokenize(query);
        let query_embedding = self.calculate_tfidf(&tokens);

        let mut similarities: Vec<(f32, &Document)> = self
            .documents
            .values()
            .map(|doc| {
                let similarity = self.cosine_similarity(&doc.embedding, &query_embedding);
                (similarity, doc)
            })
            .collect();

        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        similarities
            .into_iter()
            .take(top_k)
            .map(|(_, doc)| doc)
            .collect()
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        lazy_static! {
            static ref STOP_WORDS: FxHashSet<&'static str> = {
                let words = vec![
                    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
                    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
                    "to", "was", "were", "will", "with"
                ];
                words.into_iter().collect()
            };
        }

        // Normalize text
        let text = text.nfc().collect::<String>().to_lowercase();
        
        // Remove special characters and split into tokens
        let re = Regex::new(r"[^\w\s]").unwrap();
        let text = re.replace_all(&text, " ");
        
        text.split_whitespace()
            .filter(|&token| !STOP_WORDS.contains(token))
            .map(|token| token.to_string())
            .collect()
    }

    fn calculate_tfidf(&self, tokens: &[String]) -> Array1<f32> {
        let mut term_freq = FxHashMap::default();
        
        // Calculate term frequencies
        for token in tokens {
            *term_freq.entry(token.clone()).or_insert(0.0) += 1.0;
        }
        
        // Normalize term frequencies
        let tokens_count = tokens.len() as f32;
        for freq in term_freq.values_mut() {
            *freq /= tokens_count;
        }
        
        // Calculate TF-IDF vector
        let vocab_size = self.vocabulary.len();
        let mut tfidf = vec![0.0; vocab_size];
        
        for (i, term) in self.vocabulary.iter().enumerate() {
            if let Some(tf) = term_freq.get(term) {
                if let Some(idf) = self.idf_values.get(term) {
                    tfidf[i] = tf * idf;
                }
            }
        }
        
        Array1::from(tfidf)
    }

    fn update_idf_values(&mut self) {
        let doc_count = self.documents.len() as f32;
        
        for term in &self.vocabulary {
            let doc_freq = self.documents.values()
                .filter(|doc| self.tokenize(&doc.content).contains(term))
                .count() as f32;
            
            let idf = (1.0 + doc_count / (1.0 + doc_freq)).ln();
            self.idf_values.insert(term.clone(), idf);
        }
    }

    fn cosine_similarity(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let dot_product = a.dot(b);
        let norm_a = (a.dot(a)).sqrt();
        let norm_b = (b.dot(b)).sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}