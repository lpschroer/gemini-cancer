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
//! - Chat wrappers for managing multi-turn conversations with automatic history
//!
//! ## Example Usage
//!
//! ### Basic Request
//!
//! ```rust,ignore
//! use gemini::{GeminiConfig, GenerateContentRequest, Content, Part, JsonString};
//!
//! // Create configuration from environment
//! let config = GeminiConfig::from_env()?;
//!
//! // Build a request using the builder pattern
//! let request = GenerateContentRequest::builder()
//!     .add_content(Content::user(vec![
//!         Part::builder()
//!             .text(JsonString::new("Tell me a story".to_string()))
//!             .build(),
//!     ]))
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
//!     .add_content(Content::user(vec![
//!         Part::builder()
//!             .text(JsonString::new("Create a character".to_string()))
//!             .build(),
//!     ]))
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
//!
//! ### Multi-Turn Conversations (Chat)
//!
//! For managing conversations with automatic history:
//!
//! ```rust,ignore
//! use gemini::{GeminiV1Beta, GeminiConfig, GeminiChat};
//!
//! // Create API client and chat session
//! let config = GeminiConfig::from_env()?;
//! let client = GeminiV1Beta::new(config);
//! let mut chat = GeminiChat::new(client);
//!
//! // Send messages - history is maintained automatically
//! let response = chat
//!     .send_message()
//!     .text("Hello! Can you help me with Rust?")
//!     .send()
//!     .await?;
//!
//! let response2 = chat
//!     .send_message()
//!     .text("What are the main benefits?")
//!     .send()
//!     .await?;
//!
//! // Save conversation history
//! let history = chat.get_history();
//! let json = serde_json::to_string(history)?;
//! ```
//!
//! ### Streaming Chat
//!
//! For real-time streaming responses:
//!
//! ```rust,ignore
//! use gemini::{GeminiV1Beta, GeminiConfig, GeminiStreamChat};
//! use futures::StreamExt;
//!
//! let config = GeminiConfig::from_env()?;
//! let client = GeminiV1Beta::new(config);
//! let mut chat = GeminiStreamChat::new(client);
//!
//! // Stream the response
//! let mut stream = chat
//!     .send_message_stream()
//!     .text("Tell me a story")
//!     .send()
//!     .await?;
//!
//! while let Some(chunk) = stream.next().await {
//!     let response = chunk?;
//!     if let Some(text) = response.first_text() {
//!         print!("{}", text);
//!     }
//! }
//! // History is updated automatically when stream completes
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
pub use chat::{BufferedChatStream, GeminiChat, GeminiStreamChat};
pub use client::GeminiV1Beta;
pub use config::GeminiConfig;
pub use dto_content::{
    Blob, CodeExecutionResult, Content, ExecutableCode, FileData, FunctionCall, FunctionResponse,
    JsonString, Part, PartBuilder, VideoMetadata,
};
pub use dto_request::{
    GenerateContentRequest, GenerateContentRequestBuilder, GenerationConfig,
    GenerationConfigBuilder, MimeType, ResponseMimeType, SafetyRating, SafetySetting,
};
pub use dto_response::{Candidate, GenerateContentResponse, PromptFeedback, UsageMetadata};
