# RAG System (Retrieval-Augmented Generation)
Final Project #
Student: Gulbanu Madiyarova #
Student ID: 2105242

Project Overview:
The TAPSSP Project implements a lightweight Retrieval-Augmented Generation (RAG) system in Rust. This project allows users to:
Ingest content from direct input, text files, or PDFs.
Embed content into vector representations using MiniLM embeddings.
Query the system using natural language, retrieving relevant content chunks and generating responses via a local LLM (phi-2.Q2_K.gguf).
The system demonstrates key concepts of modern AI pipelines including vector search, embeddings, prompt augmentation, and integration with local LLMs.

User Input
   │
   ▼
[Vector DB] ←──── Ingested content (text/PDF)
   │
   ▼
Relevant Chunks → [LLM Prompt Builder] → [LLM Subprocess] → Response

Components:

main.rs – CLI interface using clap, supporting:

- ask "question" – queries the RAG system.

- remember "text" – directly stores knowledge.

- upload [text/pdf] path – adds documents to the vector DB.

vector_db.rs – In-memory vector database.

- Stores Content and VectorIndex structs.

- Computes embeddings for content chunks using embeddings.rs.

- Supports smart_insert_content to split text into 300-character chunks.

- Supports cosine similarity retrieval for query relevance.

embeddings.rs – MiniLM embedding pipeline.

- Loads pre-trained all-MiniLM-L6-v2 model.

- Ensures proper F32 tensor usage for Candle-based transformer inference.

- Applies attention-masked mean pooling and L2 normalization for consistent embeddings.

llm.rs – Lightweight LLM wrapper for llama.cpp models.

- Builds RAG prompts including context from retrieved content chunks.

- Spawns subprocess to run llama-simple with specified model.

- Handles prompt stripping from model output to return clean responses.

ingest.rs – Ingests content from CLI, text files, or PDFs.

- For PDFs, uses pdf_extract::extract_text_from_mem.

- Calls smart_insert_content to store content in vector DB.

Key Coding Decisions
1. In-Memory DB vs Persistent Storage
    * Chose lazy_static + Mutex for in-memory storage to simplify concurrency and avoid external DB setup.
    * Ideal for a school project; can scale to persistent DB later.
2. MiniLM Embeddings
    * Selected all-MiniLM-L6-v2 for a compact, fast embedding model.
    * Used Candle-based Rust transformer bindings (candle_transformers) to avoid Python dependencies.
    * Applied attention mask and mean pooling to handle variable-length sequences.
3. Vector Chunks
    * Split content into 300-character chunks to ensure fine-grained retrieval.
    * Avoids prompt length issues when sending context to LLM.
4. LLM Integration
    * Used llama-simple subprocess to call GGUF model (phi-2.Q2_K.gguf) locally.
    * Avoided Rust-native llama bindings to simplify cross-platform execution.
    * Added robust prompt stripping to remove echoed input from responses.
5. Error Handling
    * Extensive use of anyhow::Result and ? operator for concise error propagation.
    * Added guards for missing files, invalid embeddings, and subprocess failures.
6. RAG Prompting
    * build_prompt inserts top-matching content chunks as context before query.
    * Ensures the LLM leverages retrieved knowledge rather than hallucinating.

Usage
Setup
1. Clone the repository:

git clone <repo-url>
cd tapssp-project
1. Install Rust and dependencies:

cargo build
1. Download the GGUF model (phi-2.Q2_K.gguf) and embeddings model (all-MiniLM-L6-v2) into the correct paths.
2. Set environment variables:

export TAPSSP_PHI_PATH="/path/to/phi-2.Q2_K.gguf"
export TAPSSP_LLAMA_BIN="/opt/homebrew/bin/llama-simple"
Commands
* Ask a question:

cargo run -- ask "What is an algorithm?"
* Remember a fact directly:

cargo run -- remember "Algorithms are step-by-step instructions for solving problems."
* Upload a text file:

cargo run -- upload text /path/to/file.txt
* Upload a PDF file:

cargo run -- upload pdf /path/to/file.pdf

Example

tapssp-project remember "Algorithm is a step-by-step procedure."
tapssp-project ask "What is an algorithm?"
# Output: The algorithm is a method of solving a problem that involves following steps.

Future Improvements
* Add persistent vector database (e.g., SQLite, Pinecone).
* Support larger LLM models via optimized Rust bindings.
* Add streaming responses from LLM.
* Handle multi-language content ingestion.
* Add command to delete or update content in vector DB.

This README explains all major design and coding decisions, the architecture, and how to use the project, which should meet school project requirements.

Project Description Page: https://fpl.cs.depaul.edu/cpitcher/courses/csc363/worksheets/project.html#
