use crate::vector_db::VectorDB;
use anyhow::Result;
use std::path::Path;

pub struct Retriever {
    vector_db: VectorDB,
}

impl Retriever {
    pub fn new() -> Self {
        Retriever {
            vector_db: VectorDB::new(),
        }
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let p = path.as_ref();
        if p.exists() {
            let db = VectorDB::load(p)?;
            Ok(Retriever { vector_db: db })
        } else {
            Ok(Retriever::new())
        }
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.vector_db.save(path)?;
        Ok(())
    }

    /// After bulk additions, finalize to compute IDF and real embeddings
    pub fn finalize_indexing(&mut self) {
        self.vector_db.finalize_indexing();
    }

    pub fn add_to_knowledge_base(&mut self, content: String) -> Result<()> {
        self.vector_db.add_document(content)
    }

    pub fn retrieve(&self, query: &str, top_k: usize) -> Vec<(f32, String)> {
        self.vector_db.search_similar(query, top_k)
    }
}