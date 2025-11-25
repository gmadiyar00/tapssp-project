use crate::ingest::IngestType;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "tapssp-project")]
#[command(about = "tapssp-project is AI assistant which is tailored just for you", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    Ask {
        query: String,
    },

    Remember {
        content: String,
    },

    Upload {
        #[arg(value_name = "Type")]
        content_type: IngestType, 
        path: PathBuf,
    },
}