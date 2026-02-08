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
//! use gemini::{GeminiConfig, GenerateContentRequest, Content, Part, Role, JsonString};
//!
//! // Create configuration from environment
//! let config = GeminiConfig::from_env()?;
//!
//! // Build a request using the builder pattern
//! let request = GenerateContentRequest::builder()
//!     .add_content(Content {
//!         role: Some(Role::User),
//!         parts: vec![
//!             Part::builder()
//!                 .text(JsonString::new("Tell me a story".to_string()))
//!                 .build(),
//!         ],
//!     })
//!     .build();
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
//! // Create typed request with auto-derived schema using builder pattern
//! let request: GenerateContentRequest<Character> = GenerateContentRequest::builder()
//!     .add_content(Content {
//!         role: Some(Role::User),
//!         parts: vec![
//!             Part::builder()
//!                 .text(JsonString::new("Create a character".to_string()))
//!                 .build(),
//!         ],
//!     })
//!     .generation_config(
//!         GenerationConfig::builder()
//!             .response_json_schema::<Character>()
//!             .build()?
//!     )
//!     .build();
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
pub mod chat;
pub mod client;
pub mod config;
pub mod dto_content;
pub mod dto_request;
pub mod dto_response;

// Re-export commonly used types
pub use api::{GeminiApi, GeminiStreamingApi, StreamingResponseStream};
pub use client::GeminiV1Beta;
pub use config::GeminiConfig;
pub use dto_content::{
    Blob, CodeExecutionResult, Content, ExecutableCode, FileData, FunctionCall, FunctionResponse,
    JsonString, Part, PartBuilder, Role, VideoMetadata,
};
pub use dto_request::{
    GenerateContentRequest, GenerateContentRequestBuilder, GenerationConfig,
    GenerationConfigBuilder, MimeType, ResponseMimeType, SafetyRating, SafetySetting,
};
pub use dto_response::{Candidate, GenerateContentResponse, PromptFeedback, UsageMetadata};
