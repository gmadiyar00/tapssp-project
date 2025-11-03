use crate::vector_db::VectorDB;
use anyhow::Result;

pub struct Retriever {
    vector_db: VectorDB,
}

impl Retriever {
    pub fn new() -> Self {
        Retriever {
            vector_db: VectorDB::new(),
        }
    }

    pub fn add_to_knowledge_base(&mut self, content: String) -> Result<()> {
        self.vector_db.add_document(content)
    }

    pub fn retrieve(&self, query: &str, top_k: usize) -> Vec<String> {
        self.vector_db.search_similar(query, top_k)
            .into_iter()
            .map(|doc| doc.content.clone())
            .collect()
    }
}