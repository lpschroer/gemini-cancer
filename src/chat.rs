//! Chat wrapper for managing multi-turn conversations with the Gemini API
//!
//! This module provides utilities for managing conversation history automatically
//! across multiple turns, supporting both non-streaming and streaming chat interactions.
//!
//! # Overview
//!
//! The chat module offers two main wrapper types:
//! - [`GeminiChat`] - For non-streaming conversations with blocking responses
//! - [`GeminiStreamChat`] - For streaming conversations with real-time response chunks
//!
//! Both types automatically manage conversation history using `Vec<Content<String>>`,
//! which provides type safety while allowing easy persistence via serde serialization.
//!
//! # Basic Usage (Non-Streaming)
//!
//! ```rust,ignore
//! use gemini::{GeminiV1Beta, GeminiConfig, GeminiChat};
//!
//! // Create API client
//! let config = GeminiConfig::from_env()?;
//! let client = GeminiV1Beta::new(config);
//!
//! // Create chat session
//! let mut chat = GeminiChat::new(client);
//!
//! // Send a simple text message
//! let response = chat
//!     .send_message()
//!     .text("Hello! Can you help me with Rust?")
//!     .send()
//!     .await?;
//!
//! println!("Response: {:?}", response.first_text());
//!
//! // Continue the conversation - history is maintained automatically
//! let response2 = chat
//!     .send_message()
//!     .text("What are the main benefits of Rust?")
//!     .send()
//!     .await?;
//!
//! println!("Response: {:?}", response2.first_text());
//! ```
//!
//! # Streaming Usage
//!
//! ```rust,ignore
//! use gemini::{GeminiV1Beta, GeminiConfig, GeminiStreamChat};
//! use futures::StreamExt;
//!
//! // Create streaming API client
//! let config = GeminiConfig::from_env()?;
//! let client = GeminiV1Beta::new(config);
//!
//! // Create streaming chat session
//! let mut chat = GeminiStreamChat::new(client);
//!
//! // Send a message and stream the response
//! let mut stream = chat
//!     .send_message_stream()
//!     .text("Tell me a story")
//!     .send()
//!     .await?;
//!
//! // Process each chunk as it arrives
//! while let Some(chunk) = stream.next().await {
//!     let response = chunk?;
//!     if let Some(text) = response.first_text() {
//!         print!("{}", text);
//!     }
//! }
//!
//! // History is automatically updated when stream completes
//! ```
//!
//! # Advanced Configuration
//!
//! Both chat types support per-message configuration:
//!
//! ```rust,ignore
//! use gemini::{GeminiChat, GenerationConfig, SafetySetting};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize, schemars::JsonSchema)]
//! struct Character {
//!     name: String,
//!     backstory: String,
//! }
//!
//! let response = chat
//!     .send_message()
//!     .text("Create a fantasy character")
//!     .generation_config(
//!         GenerationConfig::builder()
//!             .temperature(0.9)
//!             .response_json_schema::<Character>()
//!             .build()?
//!     )
//!     .safety_settings(vec![/* custom safety settings */])
//!     .send()
//!     .await?;
//!
//! // Response is typed as Character
//! if let Some(character) = response.first_text() {
//!     println!("Character: {}", character.name);
//! }
//! ```
//!
//! # Persistence
//!
//! Save and restore conversation history:
//!
//! ```rust,ignore
//! use gemini::{GeminiChat, GeminiV1Beta, GeminiConfig};
//!
//! // Get history from active chat
//! let history = chat.get_history();
//! let json = serde_json::to_string(history)?;
//!
//! // Store json to database/file
//! store_to_database(&json)?;
//!
//! // Later: restore the conversation
//! let loaded_json = load_from_database()?;
//! let history = serde_json::from_str(&loaded_json)?;
//!
//! let config = GeminiConfig::from_env()?;
//! let client = GeminiV1Beta::new(config);
//! let mut restored_chat = GeminiChat::from_history(client, history);
//!
//! // Continue the conversation where you left off
//! let response = restored_chat
//!     .send_message()
//!     .text("Let's continue our discussion")
//!     .send()
//!     .await?;
//! ```
//!
//! # Sending Structured Data
//!
//! Send JSON-serialized data as part of messages:
//!
//! ```rust,ignore
//! use serde::Serialize;
//!
//! #[derive(Serialize)]
//! struct UserContext {
//!     user_id: String,
//!     preferences: Vec<String>,
//! }
//!
//! let context = UserContext {
//!     user_id: "user123".to_string(),
//!     preferences: vec!["fantasy".to_string(), "sci-fi".to_string()],
//! };
//!
//! let response = chat
//!     .send_message()
//!     .json(&context)
//!     .send()
//!     .await?;
//! ```
//!
//! # Multi-Part Messages
//!
//! Send messages with multiple parts (text, images, etc.):
//!
//! ```rust,ignore
//! use gemini::{Part, JsonString};
//!
//! let parts = vec![
//!     Part::builder()
//!         .text(JsonString::new("Analyze this image:".to_string()))
//!         .build(),
//!     Part::builder()
//!         .inline_data(blob_data, "image/png".to_string())
//!         .build(),
//! ];
//!
//! let response = chat
//!     .send_message()
//!     .parts(parts)
//!     .send()
//!     .await?;
//! ```

use crate::api::{GeminiApi, GeminiStreamingApi, StreamingResponseStream};
use crate::dto_content::{Content, JsonString, Part};
use crate::dto_request::{GenerateContentRequest, GenerationConfig, SafetySetting};
use crate::dto_response::GenerateContentResponse;
use futures::stream::Stream;
use std::error::Error;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Chat wrapper for managing multi-turn conversations with the Gemini API
///
/// `GeminiChat` maintains conversation history as `Vec<Content<String>>` providing
/// type safety while allowing easy persistence via serde serialization.
///
/// # Type Parameters
/// * `A` - The GeminiApi implementor type
pub struct GeminiChat<A> {
    /// The underlying API client
    api: A,
    /// Conversation history with typed Content messages
    history: Vec<Content<String>>,
}

impl<A> GeminiChat<A>
where
    A: GeminiApi,
{
    /// Creates a new chat session with empty history
    ///
    /// # Arguments
    /// * `api` - A GeminiApi implementor to use for API calls
    ///
    /// # Example
    /// ```ignore
    /// let client = GeminiClient::new(api_key);
    /// let chat = GeminiChat::new(client);
    /// ```
    pub fn new(api: A) -> Self {
        Self {
            api,
            history: Vec::new(),
        }
    }

    /// Creates a chat session from existing history
    ///
    /// Restores a previous conversation by loading serialized history.
    ///
    /// # Arguments
    /// * `api` - A GeminiApi implementor to use for API calls
    /// * `history` - Previously saved conversation history
    ///
    /// # Example
    /// ```ignore
    /// let client = GeminiClient::new(api_key);
    /// let saved_history: Vec<Content<String>> = serde_json::from_str(&stored_json)?;
    /// let chat = GeminiChat::from_history(client, saved_history);
    /// ```
    pub fn from_history(api: A, history: Vec<Content<String>>) -> Self {
        Self { api, history }
    }

    /// Start building a message to send
    ///
    /// Returns a `SendMessageBuilder` that allows configuring the message
    /// with text, JSON, parts, generation config, and safety settings before sending.
    ///
    /// # Example
    /// ```ignore
    /// let response = chat
    ///     .send_message()
    ///     .text("Hello, world!")
    ///     .send()
    ///     .await?;
    /// ```
    pub fn send_message<T>(&mut self) -> SendMessageBuilder<'_, A, T>
    where
        T: serde::de::DeserializeOwned + serde::Serialize + Send + Clone + 'static,
    {
        SendMessageBuilder::new(self)
    }

    /// Returns a reference to the conversation history
    ///
    /// The history can be serialized for persistence using serde.
    ///
    /// # Example
    /// ```ignore
    /// let history = chat.get_history();
    /// let json = serde_json::to_string(history)?;
    /// // Store json for later restoration
    /// ```
    pub fn get_history(&self) -> &[Content<String>] {
        &self.history
    }

    /// Clears the conversation history
    ///
    /// Resets the chat to an empty state while keeping the same API client.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

/// Builder for sending messages in a chat conversation
///
/// Provides a fluent API for configuring messages with various options
/// before sending them to the Gemini API.
///
/// # Type Parameters
/// * `'a` - Lifetime of the chat reference
/// * `A` - The GeminiApi implementor type
/// * `T` - The expected response type
pub struct SendMessageBuilder<'a, A, T> {
    chat: &'a mut GeminiChat<A>,
    message_parts: Option<Vec<Part<String>>>,
    generation_config: Option<GenerationConfig<T>>,
    safety_settings: Option<Vec<SafetySetting>>,
}

impl<'a, A, T> SendMessageBuilder<'a, A, T>
where
    A: GeminiApi,
    T: serde::de::DeserializeOwned + serde::Serialize + Send + Clone + 'static,
{
    /// Creates a new builder for the given chat
    fn new(chat: &'a mut GeminiChat<A>) -> Self {
        Self {
            chat,
            message_parts: None,
            generation_config: None,
            safety_settings: None,
        }
    }

    /// Set the message as plain text
    ///
    /// # Arguments
    /// * `text` - The text message to send
    ///
    /// # Example
    /// ```ignore
    /// chat.send_message()
    ///     .text("Tell me a story")
    ///     .send()
    ///     .await?;
    /// ```
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.message_parts = Some(vec![
            Part::builder().text(JsonString::new(text.into())).build(),
        ]);
        self
    }

    /// Set the message as JSON-serialized text
    ///
    /// Useful for sending structured data that the model should process.
    ///
    /// # Arguments
    /// * `value` - Any serializable value to send as JSON text
    ///
    /// # Example
    /// ```ignore
    /// #[derive(Serialize)]
    /// struct Task {
    ///     title: String,
    ///     priority: u8,
    /// }
    ///
    /// chat.send_message()
    ///     .json(&Task { title: "Review code".into(), priority: 1 })
    ///     .send()
    ///     .await?;
    /// ```
    pub fn json(mut self, value: impl serde::Serialize) -> Result<Self, Box<dyn Error>> {
        let json_str = serde_json::to_string(&value)?;
        self.message_parts = Some(vec![
            Part::builder().text(JsonString::new(json_str)).build(),
        ]);
        Ok(self)
    }

    /// Set the message as multiple parts
    ///
    /// Allows sending complex messages with multiple text, image, or other parts.
    ///
    /// # Arguments
    /// * `parts` - Vector of message parts
    ///
    /// # Example
    /// ```ignore
    /// chat.send_message()
    ///     .parts(vec![
    ///         Part::builder().text(JsonString::new("Describe this:".into())).build(),
    ///         Part::builder().inline_data(blob).build(),
    ///     ])
    ///     .send()
    ///     .await?;
    /// ```
    pub fn parts(mut self, parts: Vec<Part<String>>) -> Self {
        self.message_parts = Some(parts);
        self
    }

    /// Set optional generation configuration for this message
    ///
    /// # Arguments
    /// * `config` - Generation configuration with temperature, tokens, etc.
    ///
    /// # Example
    /// ```ignore
    /// chat.send_message()
    ///     .text("Be creative!")
    ///     .generation_config(GenerationConfig::builder()
    ///         .temperature(1.5)
    ///         .build()?)
    ///     .send()
    ///     .await?;
    /// ```
    pub fn generation_config(mut self, config: GenerationConfig<T>) -> Self {
        self.generation_config = Some(config);
        self
    }

    /// Set optional safety settings for this message
    ///
    /// # Arguments
    /// * `settings` - Vector of safety settings
    ///
    /// # Example
    /// ```ignore
    /// chat.send_message()
    ///     .text("Tell me about...")
    ///     .safety_settings(vec![SafetySetting {
    ///         category: "HARM_CATEGORY_DANGEROUS".into(),
    ///         threshold: "BLOCK_MEDIUM_AND_ABOVE".into(),
    ///     }])
    ///     .send()
    ///     .await?;
    /// ```
    pub fn safety_settings(mut self, settings: Vec<SafetySetting>) -> Self {
        self.safety_settings = Some(settings);
        self
    }

    /// Adds user message to history
    fn add_user_message_to_history(&mut self, parts: Vec<Part<String>>) {
        self.chat.history.push(Content::user(parts));
    }

    /// Converts history to Content objects for the request
    fn convert_history_to_contents(&self) -> Vec<Content> {
        self.chat.history.clone()
    }

    /// Builds the request with contents and optional config
    fn build_request(&self, contents: Vec<Content>) -> GenerateContentRequest<T> {
        let mut request_builder = GenerateContentRequest::<T>::builder().contents(contents);

        if let Some(config) = &self.generation_config {
            request_builder = request_builder.generation_config(config.clone());
        }

        if let Some(settings) = &self.safety_settings {
            request_builder = request_builder.safety_settings(settings.clone());
        }

        request_builder.build()
    }

    /// Converts response parts to string parts for history storage
    fn convert_response_to_string_parts<U>(parts: &[Part<U>]) -> Vec<Part<String>>
    where
        U: serde::Serialize,
    {
        parts
            .iter()
            .map(|part| {
                if let Some(text) = part.text() {
                    let json_str = serde_json::to_string(text).unwrap_or_else(|_| "".to_string());
                    Part::builder().text(JsonString::new(json_str)).build()
                } else {
                    Part::builder()
                        .text(JsonString::new("".to_string()))
                        .build()
                }
            })
            .collect()
    }

    /// Adds model response to history
    fn add_model_response_to_history(&mut self, response: &GenerateContentResponse<T>) {
        if let Some(candidate) = response.candidates.first() {
            let string_parts = Self::convert_response_to_string_parts(candidate.content.parts());
            self.chat.history.push(Content::model(string_parts));
        }
    }

    /// Send the message and update chat history
    ///
    /// This method:
    /// 1. Wraps the message in `Content::User` and adds to history
    /// 2. Builds a request with full conversation history
    /// 3. Calls the API with optional config and safety settings
    /// 4. Extracts the response and adds it to history as `Content::Model`
    /// 5. Returns the typed response
    ///
    /// # Returns
    /// The model's response, or an error if the request fails
    ///
    /// # Errors
    /// Returns an error if:
    /// - No message parts were set
    /// - The API call fails
    pub async fn send(mut self) -> Result<GenerateContentResponse<T>, Box<dyn Error>> {
        let parts = self
            .message_parts
            .take()
            .ok_or("Message parts must be set before sending")?;

        self.add_user_message_to_history(parts);

        let contents = self.convert_history_to_contents();

        let request = self.build_request(contents);

        let response = self.chat.api.generate_content(request).await?;

        self.add_model_response_to_history(&response);

        Ok(response)
    }
}

/// Streaming chat wrapper for managing multi-turn conversations with the Gemini API
///
/// `GeminiStreamChat` maintains conversation history and provides streaming responses,
/// buffering model responses until the stream completes before adding them to history.
///
/// # Type Parameters
/// * `A` - The GeminiStreamingApi implementor type
pub struct GeminiStreamChat<A> {
    /// The underlying streaming API client
    api: A,
    /// Conversation history with typed Content messages
    history: Vec<Content<String>>,
}

impl<A> GeminiStreamChat<A>
where
    A: GeminiStreamingApi,
{
    /// Creates a new chat instance with empty history
    ///
    /// # Arguments
    /// * `api` - The GeminiStreamingApi implementor to use for API calls
    ///
    /// # Example
    /// ```ignore
    /// let chat = GeminiStreamChat::new(api_client);
    /// ```
    pub fn new(api: A) -> Self {
        Self {
            api,
            history: Vec::new(),
        }
    }

    /// Creates a chat instance from existing history
    ///
    /// # Arguments
    /// * `api` - The GeminiStreamingApi implementor to use for API calls
    /// * `history` - Previous conversation history to restore
    pub fn from_history(api: A, history: Vec<Content<String>>) -> Self {
        Self { api, history }
    }

    /// Begin building a streaming message to send
    ///
    /// Returns a builder that allows setting message content and optional
    /// configuration before sending.
    ///
    /// # Example
    /// ```ignore
    /// let mut stream = chat.send_message_stream()
    ///     .text("Hello!")
    ///     .send()
    ///     .await?;
    /// ```
    pub fn send_message_stream<T>(&mut self) -> SendMessageStreamBuilder<'_, A, T>
    where
        T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
    {
        SendMessageStreamBuilder::new(self)
    }

    /// Get a reference to the conversation history
    ///
    /// Returns a slice of `Content<String>` that can be serialized directly
    /// for persistence using `serde_json::to_string()`.
    pub fn get_history(&self) -> &[Content<String>] {
        &self.history
    }

    /// Clear the conversation history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

/// Builder for sending streaming messages with optional configuration
///
/// Created via `GeminiStreamChat::send_message_stream()`
pub struct SendMessageStreamBuilder<'a, A, T> {
    chat: &'a mut GeminiStreamChat<A>,
    message_parts: Option<Vec<Part<String>>>,
    generation_config: Option<GenerationConfig<T>>,
    safety_settings: Option<Vec<SafetySetting>>,
}

impl<'a, A, T> SendMessageStreamBuilder<'a, A, T>
where
    A: GeminiStreamingApi,
    T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
{
    fn new(chat: &'a mut GeminiStreamChat<A>) -> Self {
        Self {
            chat,
            message_parts: None,
            generation_config: None,
            safety_settings: None,
        }
    }

    /// Set the message as plain text
    ///
    /// # Arguments
    /// * `text` - The text content to send
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.message_parts = Some(vec![
            Part::builder().text(JsonString::new(text.into())).build(),
        ]);
        self
    }

    /// Set the message as JSON-serialized text
    ///
    /// This serializes the provided value to JSON and sends it as text content.
    ///
    /// # Arguments
    /// * `value` - A serializable value to send as JSON text
    ///
    /// # Panics
    /// Panics if serialization fails
    pub fn json(mut self, value: impl serde::Serialize) -> Self {
        let json_text = serde_json::to_string(&value).expect("Failed to serialize JSON");
        self.message_parts = Some(vec![
            Part::builder().text(JsonString::new(json_text)).build(),
        ]);
        self
    }

    /// Set the message as multiple parts
    ///
    /// # Arguments
    /// * `parts` - Vector of content parts to send
    pub fn parts(mut self, parts: Vec<Part<String>>) -> Self {
        self.message_parts = Some(parts);
        self
    }

    /// Set optional generation configuration
    ///
    /// # Arguments
    /// * `config` - The generation config to use for this message
    pub fn generation_config(mut self, config: GenerationConfig<T>) -> Self {
        self.generation_config = Some(config);
        self
    }

    /// Set optional safety settings
    ///
    /// # Arguments
    /// * `settings` - Vector of safety settings to apply
    pub fn safety_settings(mut self, settings: Vec<SafetySetting>) -> Self {
        self.safety_settings = Some(settings);
        self
    }

    fn add_user_message_to_history(&mut self, parts: Vec<Part<String>>) {
        self.chat.history.push(Content::User { parts });
    }

    fn convert_history_to_contents(&self) -> Vec<Content> {
        self.chat.history.clone()
    }

    fn build_request(&mut self, contents: Vec<Content>) -> GenerateContentRequest<T> {
        let mut builder = GenerateContentRequest::builder().contents(contents);

        if let Some(config) = self.generation_config.take() {
            builder = builder.generation_config(config);
        }

        if let Some(settings) = self.safety_settings.take() {
            builder = builder.safety_settings(settings);
        }

        builder.build()
    }

    /// Send the message and return a streaming response
    ///
    /// This method:
    /// 1. Wraps the message in `Content::User` and adds to history
    /// 2. Builds a request with full conversation history
    /// 3. Calls the streaming API with optional config and safety settings
    /// 4. Returns a `BufferedChatStream` that buffers the response and updates history on completion
    ///
    /// # Returns
    /// A stream that yields response chunks and updates history when complete
    ///
    /// # Errors
    /// Returns an error if:
    /// - No message parts were set
    /// - The API call fails
    pub async fn send(mut self) -> Result<BufferedChatStream<'a, T>, Box<dyn Error>> {
        let parts = self
            .message_parts
            .take()
            .ok_or("Message parts must be set before sending")?;

        self.add_user_message_to_history(parts);

        let contents = self.convert_history_to_contents();

        let request = self.build_request(contents);

        let stream = self.chat.api.stream_generate_content(request).await?;

        Ok(BufferedChatStream::new(stream, &mut self.chat.history))
    }
}

/// A streaming response wrapper that buffers content and updates chat history on completion
///
/// This stream forwards chunks to the caller while buffering text content internally.
/// When the stream completes, it constructs a `Content::Model` from the buffered content
/// and appends it to the conversation history.
pub struct BufferedChatStream<'a, T> {
    inner: StreamingResponseStream<T>,
    history: &'a mut Vec<Content<String>>,
    buffer: Vec<String>,
    completed: bool,
}

impl<'a, T> BufferedChatStream<'a, T> {
    fn new(stream: StreamingResponseStream<T>, history: &'a mut Vec<Content<String>>) -> Self {
        Self {
            inner: stream,
            history,
            buffer: Vec::new(),
            completed: false,
        }
    }

    fn extract_text_from_response(response: &GenerateContentResponse<T>) -> Vec<String>
    where
        T: Clone + ToString,
    {
        response
            .candidates
            .iter()
            .flat_map(|candidate| {
                candidate
                    .content
                    .parts()
                    .iter()
                    .filter_map(|part| part.text().map(|text| text.to_string()))
            })
            .collect()
    }

    fn finalize_history(&mut self) {
        if !self.completed && !self.buffer.is_empty() {
            let combined_text = self.buffer.join("");
            let parts = vec![Part::builder().text(JsonString::new(combined_text)).build()];
            self.history.push(Content::Model { parts });
            self.completed = true;
        }
    }
}

impl<'a, T> Stream for BufferedChatStream<'a, T>
where
    T: Unpin + Clone + ToString,
{
    type Item = Result<GenerateContentResponse<T>, Box<dyn Error + Send + Sync>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(response))) => {
                // Buffer text content from this chunk
                let texts = Self::extract_text_from_response(&response);
                self.buffer.extend(texts);
                Poll::Ready(Some(Ok(response)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => {
                // Stream complete - finalize history
                self.finalize_history();
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<'a, T> Drop for BufferedChatStream<'a, T> {
    fn drop(&mut self) {
        // Ensure history is updated even if stream is dropped early
        self.finalize_history();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::GeminiApi;
    use futures::stream::StreamExt;

    #[test]
    fn test_gemini_chat_new() {
        struct MockApi;
        #[async_trait::async_trait]
        impl GeminiApi for MockApi {
            async fn generate_content<T>(
                &self,
                _request: GenerateContentRequest<T>,
            ) -> Result<GenerateContentResponse<T>, Box<dyn std::error::Error>>
            where
                T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
            {
                unimplemented!()
            }
        }

        let chat = GeminiChat::new(MockApi);
        assert_eq!(chat.history.len(), 0);
    }

    #[test]
    fn test_gemini_chat_from_history() {
        struct MockApi;
        #[async_trait::async_trait]
        impl GeminiApi for MockApi {
            async fn generate_content<T>(
                &self,
                _request: GenerateContentRequest<T>,
            ) -> Result<GenerateContentResponse<T>, Box<dyn std::error::Error>>
            where
                T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
            {
                unimplemented!()
            }
        }

        let history = vec![
            Content::user(vec![
                Part::builder()
                    .text(JsonString::new("Hello".to_string()))
                    .build(),
            ]),
            Content::model(vec![
                Part::builder()
                    .text(JsonString::new("Hi there!".to_string()))
                    .build(),
            ]),
        ];

        let chat = GeminiChat::from_history(MockApi, history);
        assert_eq!(chat.history.len(), 2);
        assert!(chat.history[0].is_user());
        assert!(chat.history[1].is_model());
    }

    #[test]
    fn test_gemini_chat_get_history() {
        struct MockApi;
        #[async_trait::async_trait]
        impl GeminiApi for MockApi {
            async fn generate_content<T>(
                &self,
                _request: GenerateContentRequest<T>,
            ) -> Result<GenerateContentResponse<T>, Box<dyn std::error::Error>>
            where
                T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
            {
                unimplemented!()
            }
        }

        let history = vec![Content::user(vec![
            Part::builder()
                .text(JsonString::new("Test".to_string()))
                .build(),
        ])];

        let chat = GeminiChat::from_history(MockApi, history);
        assert_eq!(chat.get_history().len(), 1);
    }

    #[test]
    fn test_gemini_chat_clear_history() {
        struct MockApi;
        #[async_trait::async_trait]
        impl GeminiApi for MockApi {
            async fn generate_content<T>(
                &self,
                _request: GenerateContentRequest<T>,
            ) -> Result<GenerateContentResponse<T>, Box<dyn std::error::Error>>
            where
                T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
            {
                unimplemented!()
            }
        }

        let history = vec![Content::user(vec![
            Part::builder()
                .text(JsonString::new("Test".to_string()))
                .build(),
        ])];

        let mut chat = GeminiChat::from_history(MockApi, history);
        assert_eq!(chat.history.len(), 1);

        chat.clear_history();
        assert_eq!(chat.history.len(), 0);
    }

    #[tokio::test]
    async fn test_send_message_builder_text() {
        struct MockApi;
        #[async_trait::async_trait]
        impl GeminiApi for MockApi {
            async fn generate_content<T>(
                &self,
                request: GenerateContentRequest<T>,
            ) -> Result<GenerateContentResponse<T>, Box<dyn std::error::Error>>
            where
                T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
            {
                // Verify request has content
                assert_eq!(request.contents().len(), 1);
                assert!(request.contents()[0].is_user());

                // Return mock response with String type
                let string_response = GenerateContentResponse {
                    candidates: vec![crate::dto_response::Candidate {
                        content: Content::model(vec![
                            Part::builder()
                                .text(JsonString::new("Mock response".to_string()))
                                .build(),
                        ]),
                        finish_reason: Some("STOP".to_string()),
                        safety_ratings: vec![],
                    }],
                    prompt_feedback: None,
                    usage_metadata: None,
                };

                // Serialize and deserialize to convert to type T
                let json = serde_json::to_string(&string_response)?;
                Ok(serde_json::from_str(&json)?)
            }
        }

        let mut chat = GeminiChat::new(MockApi);
        let response: GenerateContentResponse<String> = chat
            .send_message()
            .text("Hello")
            .send()
            .await
            .expect("Failed to send message");

        assert_eq!(chat.history.len(), 2); // User + Model
        assert!(chat.history[0].is_user());
        assert!(chat.history[1].is_model());
        assert_eq!(response.first_text().unwrap(), "Mock response");
    }

    #[tokio::test]
    async fn test_send_message_builder_with_config() {
        struct MockApi;
        #[async_trait::async_trait]
        impl GeminiApi for MockApi {
            async fn generate_content<T>(
                &self,
                request: GenerateContentRequest<T>,
            ) -> Result<GenerateContentResponse<T>, Box<dyn std::error::Error>>
            where
                T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
            {
                // Verify config was passed
                assert!(request.generation_config().is_some());

                let string_response = GenerateContentResponse {
                    candidates: vec![crate::dto_response::Candidate {
                        content: Content::model(vec![
                            Part::builder()
                                .text(JsonString::new("Response".to_string()))
                                .build(),
                        ]),
                        finish_reason: Some("STOP".to_string()),
                        safety_ratings: vec![],
                    }],
                    prompt_feedback: None,
                    usage_metadata: None,
                };

                let json = serde_json::to_string(&string_response)?;
                Ok(serde_json::from_str(&json)?)
            }
        }

        let mut chat = GeminiChat::new(MockApi);
        let config: GenerationConfig<String> = GenerationConfig::builder()
            .temperature(0.8)
            .build()
            .unwrap();

        let response: GenerateContentResponse<String> = chat
            .send_message()
            .text("Test")
            .generation_config(config)
            .send()
            .await
            .expect("Failed to send message");

        assert_eq!(chat.history.len(), 2);
        assert!(response.first_text().is_some());
    }

    #[tokio::test]
    async fn test_send_message_accumulates_history() {
        struct MockApi;
        #[async_trait::async_trait]
        impl GeminiApi for MockApi {
            async fn generate_content<T>(
                &self,
                request: GenerateContentRequest<T>,
            ) -> Result<GenerateContentResponse<T>, Box<dyn std::error::Error>>
            where
                T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
            {
                // Return response with incremented history
                let string_response = GenerateContentResponse {
                    candidates: vec![crate::dto_response::Candidate {
                        content: Content::model(vec![
                            Part::builder()
                                .text(JsonString::new(format!(
                                    "Response {}",
                                    request.contents().len()
                                )))
                                .build(),
                        ]),
                        finish_reason: Some("STOP".to_string()),
                        safety_ratings: vec![],
                    }],
                    prompt_feedback: None,
                    usage_metadata: None,
                };

                let json = serde_json::to_string(&string_response)?;
                Ok(serde_json::from_str(&json)?)
            }
        }

        let mut chat = GeminiChat::new(MockApi);

        // Send first message
        let _: GenerateContentResponse<String> = chat
            .send_message()
            .text("First")
            .send()
            .await
            .expect("Failed to send");
        assert_eq!(chat.history.len(), 2);

        // Send second message - should include previous history
        let _: GenerateContentResponse<String> = chat
            .send_message()
            .text("Second")
            .send()
            .await
            .expect("Failed to send");
        assert_eq!(chat.history.len(), 4);

        // Verify history order
        assert!(chat.history[0].is_user());
        assert!(chat.history[1].is_model());
        assert!(chat.history[2].is_user());
        assert!(chat.history[3].is_model());
    }

    #[test]
    fn test_content_serialization_user() {
        let content = Content::user(vec![
            Part::builder()
                .text(JsonString::new("Hello".to_string()))
                .build(),
        ]);
        let json = serde_json::to_value(&content).expect("Failed to serialize");

        assert_eq!(json["role"], "user");
        assert!(json["parts"].is_array());
        assert_eq!(json["parts"][0]["text"], "Hello");
    }

    #[test]
    fn test_content_serialization_model() {
        let content = Content::model(vec![
            Part::builder()
                .text(JsonString::new("Hi there".to_string()))
                .build(),
        ]);
        let json = serde_json::to_value(&content).expect("Failed to serialize");

        assert_eq!(json["role"], "model");
        assert!(json["parts"].is_array());
        assert_eq!(json["parts"][0]["text"], "Hi there");
    }

    #[test]
    fn test_content_deserialization_user() {
        let json = r#"{
            "role": "user",
            "parts": [{"text": "Hello"}]
        }"#;

        let content: Content<String> = serde_json::from_str(json).expect("Failed to deserialize");
        assert!(content.is_user());
        assert_eq!(content.parts().len(), 1);
        assert_eq!(content.parts()[0].text().unwrap(), "Hello");
    }

    #[test]
    fn test_content_deserialization_model() {
        let json = r#"{
            "role": "model",
            "parts": [{"text": "Response"}]
        }"#;

        let content: Content<String> = serde_json::from_str(json).expect("Failed to deserialize");
        assert!(content.is_model());
        assert_eq!(content.parts().len(), 1);
        assert_eq!(content.parts()[0].text().unwrap(), "Response");
    }

    #[test]
    fn test_content_serialization_roundtrip() {
        let original = Content::user(vec![
            Part::builder()
                .text(JsonString::new("Test".to_string()))
                .build(),
        ]);
        let json = serde_json::to_string(&original).expect("Failed to serialize");
        let deserialized: Content<String> =
            serde_json::from_str(&json).expect("Failed to deserialize");

        assert!(deserialized.is_user());
        assert_eq!(deserialized.parts()[0].text().unwrap(), "Test");
    }

    #[test]
    fn test_history_persistence_roundtrip() {
        let history = vec![
            Content::user(vec![
                Part::builder()
                    .text(JsonString::new("Question".to_string()))
                    .build(),
            ]),
            Content::model(vec![
                Part::builder()
                    .text(JsonString::new("Answer".to_string()))
                    .build(),
            ]),
        ];

        // Serialize history
        let json = serde_json::to_string(&history).expect("Failed to serialize history");

        // Deserialize history
        let restored: Vec<Content<String>> =
            serde_json::from_str(&json).expect("Failed to deserialize history");

        assert_eq!(restored.len(), 2);
        assert!(restored[0].is_user());
        assert!(restored[1].is_model());
        assert_eq!(restored[0].parts()[0].text().unwrap(), "Question");
        assert_eq!(restored[1].parts()[0].text().unwrap(), "Answer");
    }

    #[tokio::test]
    async fn test_gemini_stream_chat_new() {
        struct MockStreamApi;
        #[async_trait::async_trait]
        impl GeminiStreamingApi for MockStreamApi {
            async fn stream_generate_content<T>(
                &self,
                _request: GenerateContentRequest<T>,
            ) -> Result<StreamingResponseStream<T>, Box<dyn Error>>
            where
                T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
            {
                unimplemented!()
            }
        }

        let chat = GeminiStreamChat::new(MockStreamApi);
        assert_eq!(chat.get_history().len(), 0);
    }

    #[tokio::test]
    async fn test_gemini_stream_chat_from_history() {
        struct MockStreamApi;
        #[async_trait::async_trait]
        impl GeminiStreamingApi for MockStreamApi {
            async fn stream_generate_content<T>(
                &self,
                _request: GenerateContentRequest<T>,
            ) -> Result<StreamingResponseStream<T>, Box<dyn Error>>
            where
                T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
            {
                unimplemented!()
            }
        }

        let history = vec![
            Content::User {
                parts: vec![
                    Part::builder()
                        .text(JsonString::new("Hello".to_string()))
                        .build(),
                ],
            },
            Content::Model {
                parts: vec![
                    Part::builder()
                        .text(JsonString::new("Hi there!".to_string()))
                        .build(),
                ],
            },
        ];

        let chat = GeminiStreamChat::from_history(MockStreamApi, history);
        assert_eq!(chat.get_history().len(), 2);
    }

    #[tokio::test]
    async fn test_gemini_stream_chat_clear_history() {
        struct MockStreamApi;
        #[async_trait::async_trait]
        impl GeminiStreamingApi for MockStreamApi {
            async fn stream_generate_content<T>(
                &self,
                _request: GenerateContentRequest<T>,
            ) -> Result<StreamingResponseStream<T>, Box<dyn Error>>
            where
                T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
            {
                unimplemented!()
            }
        }

        let history = vec![Content::User {
            parts: vec![
                Part::builder()
                    .text(JsonString::new("Hello".to_string()))
                    .build(),
            ],
        }];

        let mut chat = GeminiStreamChat::from_history(MockStreamApi, history);
        assert_eq!(chat.get_history().len(), 1);

        chat.clear_history();
        assert_eq!(chat.get_history().len(), 0);
    }

    #[tokio::test]
    async fn test_send_message_stream_accumulates_history() {
        use futures::stream;

        struct MockStreamApi;
        #[async_trait::async_trait]
        impl GeminiStreamingApi for MockStreamApi {
            async fn stream_generate_content<T>(
                &self,
                request: GenerateContentRequest<T>,
            ) -> Result<StreamingResponseStream<T>, Box<dyn Error>>
            where
                T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
            {
                // Verify request contains history
                assert!(!request.contents().is_empty());

                // Helper to create properly typed response
                fn create_response<T>(text: &str) -> GenerateContentResponse<T>
                where
                    T: serde::de::DeserializeOwned + serde::Serialize + 'static,
                {
                    // Create a response with String type first
                    let string_response = GenerateContentResponse::<String> {
                        candidates: vec![crate::dto_response::Candidate {
                            content: Content::Model {
                                parts: vec![
                                    Part::builder()
                                        .text(JsonString::new(text.to_string()))
                                        .build(),
                                ],
                            },
                            finish_reason: None,
                            safety_ratings: vec![],
                        }],
                        prompt_feedback: None,
                        usage_metadata: None,
                    };

                    // Serialize and deserialize to convert types
                    let json = serde_json::to_string(&string_response).unwrap();
                    serde_json::from_str(&json).unwrap()
                }

                let response1 = create_response::<T>("Hello ");
                let response2 = create_response::<T>("world!");

                let stream = stream::iter(vec![Ok(response1), Ok(response2)]);
                Ok(Box::pin(stream))
            }
        }

        let mut chat = GeminiStreamChat::new(MockStreamApi);

        // Send first message
        let stream = chat
            .send_message_stream::<String>()
            .text("First question")
            .send()
            .await
            .expect("Failed to send message");

        // Consume the stream
        let mut chunks = Vec::new();
        {
            let mut pinned_stream = Box::pin(stream);
            while let Some(result) = pinned_stream.next().await {
                let response = result.expect("Stream error");
                chunks.push(response);
            }
        }

        assert_eq!(chunks.len(), 2);

        // After stream completes, history should have user message and buffered model response
        let history = chat.get_history();
        assert_eq!(history.len(), 2);
        assert!(history[0].is_user());
        assert!(history[1].is_model());
        assert_eq!(history[0].parts()[0].text().unwrap(), "First question");
        assert_eq!(history[1].parts()[0].text().unwrap(), "Hello world!");
    }

    #[tokio::test]
    async fn test_buffered_stream_updates_history_on_drop() {
        use futures::stream;

        struct MockStreamApi;
        #[async_trait::async_trait]
        impl GeminiStreamingApi for MockStreamApi {
            async fn stream_generate_content<T>(
                &self,
                _request: GenerateContentRequest<T>,
            ) -> Result<StreamingResponseStream<T>, Box<dyn Error>>
            where
                T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
            {
                fn create_response<T>(text: &str) -> GenerateContentResponse<T>
                where
                    T: serde::de::DeserializeOwned + serde::Serialize + 'static,
                {
                    let string_response = GenerateContentResponse::<String> {
                        candidates: vec![crate::dto_response::Candidate {
                            content: Content::Model {
                                parts: vec![
                                    Part::builder()
                                        .text(JsonString::new(text.to_string()))
                                        .build(),
                                ],
                            },
                            finish_reason: None,
                            safety_ratings: vec![],
                        }],
                        prompt_feedback: None,
                        usage_metadata: None,
                    };
                    let json = serde_json::to_string(&string_response).unwrap();
                    serde_json::from_str(&json).unwrap()
                }

                let response = create_response::<T>("Partial response");

                let stream = stream::iter(vec![Ok(response)]);
                Ok(Box::pin(stream))
            }
        }

        let mut chat = GeminiStreamChat::new(MockStreamApi);

        {
            let stream = chat
                .send_message_stream::<String>()
                .text("Question")
                .send()
                .await
                .expect("Failed to send message");

            // Poll stream once to get the partial response
            let mut pinned = Box::pin(stream);
            let _ = pinned.next().await;

            // Stream is dropped here without being fully consumed
        }

        // History should still be updated with the partial response
        let history = chat.get_history();
        assert_eq!(history.len(), 2);
        assert!(history[0].is_user());
        assert!(history[1].is_model());
    }

    #[tokio::test]
    async fn test_send_message_stream_with_config() {
        use futures::stream;

        struct MockStreamApi;
        #[async_trait::async_trait]
        impl GeminiStreamingApi for MockStreamApi {
            async fn stream_generate_content<T>(
                &self,
                request: GenerateContentRequest<T>,
            ) -> Result<StreamingResponseStream<T>, Box<dyn Error>>
            where
                T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
            {
                // Verify config was passed through
                assert!(request.generation_config().is_some());

                fn create_response<T>(text: &str) -> GenerateContentResponse<T>
                where
                    T: serde::de::DeserializeOwned + serde::Serialize + 'static,
                {
                    let string_response = GenerateContentResponse::<String> {
                        candidates: vec![crate::dto_response::Candidate {
                            content: Content::Model {
                                parts: vec![
                                    Part::builder()
                                        .text(JsonString::new(text.to_string()))
                                        .build(),
                                ],
                            },
                            finish_reason: None,
                            safety_ratings: vec![],
                        }],
                        prompt_feedback: None,
                        usage_metadata: None,
                    };
                    let json = serde_json::to_string(&string_response).unwrap();
                    serde_json::from_str(&json).unwrap()
                }

                let response = create_response::<T>("Response");

                let stream = stream::iter(vec![Ok(response)]);
                Ok(Box::pin(stream))
            }
        }

        let mut chat = GeminiStreamChat::new(MockStreamApi);

        let config = GenerationConfig::<String>::builder()
            .temperature(0.7)
            .build()
            .expect("Failed to build config");

        let stream = chat
            .send_message_stream::<String>()
            .text("Question")
            .generation_config(config)
            .send()
            .await
            .expect("Failed to send message");

        // Consume stream
        {
            let mut pinned_stream = Box::pin(stream);
            while (pinned_stream.next().await).is_some() {}
        }

        assert_eq!(chat.get_history().len(), 2);
    }

    #[tokio::test]
    async fn test_send_message_stream_multiple_turns() {
        use futures::stream;

        struct MockStreamApi {
            call_count: std::sync::Arc<std::sync::Mutex<usize>>,
        }

        #[async_trait::async_trait]
        impl GeminiStreamingApi for MockStreamApi {
            async fn stream_generate_content<T>(
                &self,
                request: GenerateContentRequest<T>,
            ) -> Result<StreamingResponseStream<T>, Box<dyn Error>>
            where
                T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
            {
                let mut count = self.call_count.lock().unwrap();
                *count += 1;
                let current_count = *count;

                // Verify history grows with each call
                assert_eq!(request.contents().len(), (current_count - 1) * 2 + 1);

                fn create_response<T>(text: &str) -> GenerateContentResponse<T>
                where
                    T: serde::de::DeserializeOwned + serde::Serialize + 'static,
                {
                    let string_response = GenerateContentResponse::<String> {
                        candidates: vec![crate::dto_response::Candidate {
                            content: Content::Model {
                                parts: vec![
                                    Part::builder()
                                        .text(JsonString::new(text.to_string()))
                                        .build(),
                                ],
                            },
                            finish_reason: None,
                            safety_ratings: vec![],
                        }],
                        prompt_feedback: None,
                        usage_metadata: None,
                    };
                    let json = serde_json::to_string(&string_response).unwrap();
                    serde_json::from_str(&json).unwrap()
                }

                let response_text = format!("Response {}", current_count);
                let response = create_response::<T>(&response_text);

                let stream = stream::iter(vec![Ok(response)]);
                Ok(Box::pin(stream))
            }
        }

        let call_count = std::sync::Arc::new(std::sync::Mutex::new(0));
        let mut chat = GeminiStreamChat::new(MockStreamApi {
            call_count: call_count.clone(),
        });

        // First message
        {
            let stream1 = chat
                .send_message_stream::<String>()
                .text("First")
                .send()
                .await
                .expect("Failed to send first message");
            let mut pinned = Box::pin(stream1);
            while (pinned.next().await).is_some() {}
        }

        assert_eq!(chat.get_history().len(), 2);

        // Second message
        {
            let stream2 = chat
                .send_message_stream::<String>()
                .text("Second")
                .send()
                .await
                .expect("Failed to send second message");
            let mut pinned = Box::pin(stream2);
            while (pinned.next().await).is_some() {}
        }

        assert_eq!(chat.get_history().len(), 4);

        // Verify content
        assert_eq!(chat.get_history()[0].parts()[0].text().unwrap(), "First");
        assert_eq!(
            chat.get_history()[1].parts()[0].text().unwrap(),
            "Response 1"
        );
        assert_eq!(chat.get_history()[2].parts()[0].text().unwrap(), "Second");
        assert_eq!(
            chat.get_history()[3].parts()[0].text().unwrap(),
            "Response 2"
        );
    }
}
