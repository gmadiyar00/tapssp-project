use std::fs::{self, DirBuilder};
use std::path::Path;
use anyhow::Result;

/// Creates a directory if it doesn't exist
pub fn ensure_dir(path: impl AsRef<Path>) -> Result<()> {
    DirBuilder::new()
        .recursive(true)
        .create(path)?;
    Ok(())
}

/// Splits text into chunks of approximately max_chars length at sentence boundaries
pub fn split_into_chunks(text: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_length = 0;

    // Simple sentence splitting on .!?
    for sentence in text.split(|c| c == '.' || c == '!' || c == '?') {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            continue;
        }

        let sentence_len = sentence.chars().count();
        if current_length + sentence_len + 2 > max_chars && !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
            current_chunk.clear();
            current_length = 0;
        }

        if !current_chunk.is_empty() {
            current_chunk.push(' ');
            current_length += 1;
        }
        current_chunk.push_str(sentence);
        current_chunk.push('.');
        current_length += sentence_len + 1;
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    chunks
}

/// Loads all text files from a directory recursively
pub fn load_text_files(dir_path: impl AsRef<Path>) -> Result<Vec<String>> {
    let mut texts = Vec::new();
    
    if !dir_path.as_ref().exists() {
        ensure_dir(&dir_path)?;
        return Ok(texts);
    }

    for entry in fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "txt" {
                    let content = fs::read_to_string(path)?;
                    texts.push(content);
                }
            }
        } else if path.is_dir() {
            texts.extend(load_text_files(path)?);
        }
    }

    Ok(texts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_split_into_chunks() {
        let text = "This is a test. It has multiple sentences! How will it be split? Let's see.";
        let chunks = split_into_chunks(text, 20);
        assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 20));
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_load_text_files() -> Result<()> {
        let dir = tempdir()?;
        let file_path = dir.path().join("test.txt");
        let mut file = File::create(file_path)?;
        writeln!(file, "Test content")?;

        let texts = load_text_files(dir.path())?;
        assert_eq!(texts.len(), 1);
        assert_eq!(texts[0].trim(), "Test content");

        Ok(())
    }
}