//! Gemini API Client Library
//!
//! This crate provides a type-safe Rust client for the Google Gemini API.
//!
//! ## Features
//!
//! - Type-safe request and response DTOs
//! - Support for both OpenAPI schema and JSON Schema
//! - Generic text parsing for structured JSON responses
//! - Configurable via environment variables or explicit configuration
//! - Streaming and non-streaming content generation
//!
//! ## Example Usage
//!
//! ### Basic Request
//!
//! ```rust,ignore
//! use gemini::{GeminiConfig, GenerateContentRequest, Content, Part, Role};
//!
//! // Create configuration from environment
//! let config = GeminiConfig::from_env()?;
//!
//! // Build a request
//! let request = GenerateContentRequest {
//!     contents: vec![Content {
//!         role: Some(Role::User),
//!         parts: vec![Part {
//!             text: Some("Tell me a story".to_string()),
//!             inline_data: None,
//!             function_call: None,
//!             function_response: None,
//!             file_data: None,
//!             executable_code: None,
//!             code_execution_result: None,
//!             video_metadata: None,
//!         }],
//!     }],
//!     generation_config: None,
//!     system_instruction: None,
//!     safety_settings: None,
//! };
//! ```
//!
//! ### Parsing Structured JSON Responses
//!
//! When using `response_json_schema`, the response is automatically typed:
//!
//! ```rust,ignore
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Deserialize, Serialize, schemars::JsonSchema)]
//! struct Character {
//!     name: String,
//!     age: u32,
//!     role: String,
//! }
//!
//! // Create typed request with auto-derived schema
//! let request: GenerateContentRequest<Character> = GenerateContentRequest {
//!     contents: vec![/* ... */],
//!     generation_config: Some(GenerationConfig::builder()
//!         .response_json_schema::<Character>()
//!         .build()?),
//!     system_instruction: None,
//!     safety_settings: None,
//! };
//!
//! // Response is automatically typed as Character!
//! let response: GenerateContentResponse<Character> = api.generate_content(request)?;
//!
//! // Access the typed data directly
//! if let Some(character) = response.first_text() {
//!     println!("Character: {} (age {})", character.name, character.age);
//! }
//! ```

// Public module exports
pub mod api;
pub mod client;
pub mod config;
pub mod dto_content;
pub mod dto_request;
pub mod dto_response;

// Re-export commonly used types
pub use api::{GeminiApi, GeminiStreamingApi, StreamingResponseIterator};
pub use client::GeminiV1Beta;
pub use config::GeminiConfig;
pub use dto_content::{
    Blob, CodeExecutionResult, Content, ExecutableCode, FileData, FunctionCall, FunctionResponse,
    JsonString, Part, Role, VideoMetadata,
};
pub use dto_request::{
    GenerateContentRequest, GenerationConfig, GenerationConfigBuilder, MimeType, ResponseMimeType,
    SafetyRating, SafetySetting,
};
pub use dto_response::{Candidate, GenerateContentResponse, PromptFeedback, UsageMetadata};
