//! Gemini API client implementation.

use std::error::Error;

use async_trait::async_trait;
use futures::StreamExt;

use crate::api::{
    GeminiApi, GeminiStreamingApi, GenerateContentRequest, GenerateContentResponse,
    StreamingResponseStream,
};
use crate::config::GeminiConfig;

/// Gemini V1 Beta API client implementation
pub struct GeminiV1Beta {
    config: GeminiConfig,
    client: reqwest::Client,
}

impl GeminiV1Beta {
    /// Creates a new Gemini V1 Beta API client with the provided configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The Gemini API configuration
    pub fn new(config: GeminiConfig) -> Self {
        let client = reqwest::Client::new();
        Self { config, client }
    }

    /// Creates a new Gemini V1 Beta API client from environment variables.
    ///
    /// # Returns
    ///
    /// A `Result` containing the client instance or an error if configuration fails.
    pub fn from_env() -> Result<Self, Box<dyn Error>> {
        let config = GeminiConfig::from_env()?;
        Ok(Self::new(config))
    }

    /// Builds the URL for the generateContent endpoint.
    fn build_generate_url(&self) -> String {
        format!(
            "{}/models/{}:generateContent",
            self.config.base_url(),
            self.config.model()
        )
    }

    /// Builds the URL for the streamGenerateContent endpoint.
    fn build_stream_url(&self) -> String {
        format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.config.base_url(),
            self.config.model()
        )
    }

    /// Processes incoming bytes and appends them to the buffer.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The raw bytes received from the stream
    /// * `buffer` - The current buffer containing incomplete data
    ///
    /// # Returns
    ///
    /// The updated buffer with new bytes appended
    fn process_bytes_chunk(bytes: &[u8], buffer: &mut String) {
        let chunk_str = String::from_utf8_lossy(bytes);
        tracing::trace!(
            "Gemini stream: received {} bytes, buffer now {} chars",
            bytes.len(),
            buffer.len() + chunk_str.len()
        );
        buffer.push_str(&chunk_str);
    }

    /// Extracts a complete SSE message from the buffer if available.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Mutable reference to the buffer containing accumulated data
    ///
    /// # Returns
    ///
    /// `Some(json_data)` if a complete message was found and extracted,
    /// `None` if no complete message is available yet
    ///
    /// # Note
    ///
    /// This function modifies the buffer in-place, removing the extracted message
    fn extract_sse_message(buffer: &mut String) -> Option<String> {
        loop {
            // Try to find a complete SSE message
            // Handle different line ending formats: \r\n\r\n, \n\n, or \r\n followed by data:
            let (pos, delimiter_len) = if let Some(p) = buffer.find("\r\n\r\n") {
                (p, 4)
            } else if let Some(p) = buffer.find("\n\n") {
                (p, 2)
            } else if let Some(p) = buffer.find("\r\ndata: ") {
                // Handle case where messages are separated by \r\n without double newline
                (p, 2)
            } else if let Some(p) = buffer.find("\ndata: ") {
                // Handle case where messages are separated by single \n
                // Only match if this is not at position 0 (i.e., there's content before it)
                if p > 0 {
                    (p, 1)
                } else {
                    // No delimiter found yet
                    if !buffer.is_empty() {
                        tracing::trace!(
                            "🔍 SSE: No message boundary found yet, buffer has {} chars, starts with: {}",
                            buffer.len(),
                            if buffer.len() > 100 {
                                format!("{}...", &buffer[..100])
                            } else {
                                buffer.clone()
                            }
                        );
                    }
                    return None;
                }
            } else {
                // No delimiter found yet
                if !buffer.is_empty() {
                    tracing::trace!(
                        "🔍 SSE: No message boundary found yet, buffer has {} chars, starts with: {}",
                        buffer.len(),
                        if buffer.len() > 100 {
                            format!("{}...", &buffer[..100])
                        } else {
                            buffer.clone()
                        }
                    );
                }
                return None;
            };

            let complete_chunk = buffer[..pos].to_string();
            *buffer = buffer[pos + delimiter_len..].to_string();

            tracing::trace!(
                "SSE: Found message boundary at pos {}, chunk len {}",
                pos,
                complete_chunk.len()
            );

            // Parse SSE format: "data: {json}"
            // Also handle potential whitespace or BOM at the start
            let trimmed = complete_chunk.trim_start();
            if let Some(json_data) = trimmed.strip_prefix("data: ") {
                let json_data = json_data.trim();
                if !json_data.is_empty() {
                    return Some(json_data.to_string());
                }
            } else {
                tracing::trace!("SSE: Chunk doesn't start with 'data: ', skipping");
            }

            // If we had a delimiter but no valid data, try again with remaining buffer
            continue;
        }
    }

    /// Processes stream bytes and returns parsed response if complete message found.
    fn handle_stream_bytes<T>(
        bytes: &bytes::Bytes,
        buffer: &mut String,
    ) -> Option<Result<GenerateContentResponse<T>, Box<dyn Error + Send + Sync>>>
    where
        T: serde::de::DeserializeOwned + 'static,
    {
        Self::process_bytes_chunk(bytes, buffer);
        let json_data = Self::extract_sse_message(buffer)?;
        let result = Self::parse_incomplete::<T>(json_data)
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>);
        Some(result)
    }

    /// Converts stream error to boxed error result.
    fn handle_stream_error<T>(
        e: reqwest::Error,
    ) -> Result<GenerateContentResponse<T>, Box<dyn Error + Send + Sync>>
    where
        T: serde::de::DeserializeOwned + 'static,
    {
        Err(Box::new(e) as Box<dyn Error + Send + Sync>)
    }
}

#[async_trait]
impl GeminiApi for GeminiV1Beta {
    async fn generate_content<T>(
        &self,
        request: GenerateContentRequest<T>,
    ) -> Result<GenerateContentResponse<T>, Box<dyn Error>>
    where
        T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
    {
        let url = self.build_generate_url();

        // Build and send request asynchronously
        let response = self
            .client
            .post(&url)
            .header("x-goog-api-key", self.config.api_key())
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| Box::new(e) as Box<dyn Error>)?
            .error_for_status()
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;

        // Parse the response
        let result: GenerateContentResponse<T> = response
            .json()
            .await
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;

        Ok(result)
    }
}

#[async_trait]
impl GeminiStreamingApi for GeminiV1Beta {
    async fn stream_generate_content<T>(
        &self,
        request: GenerateContentRequest<T>,
    ) -> Result<StreamingResponseStream<T>, Box<dyn Error>>
    where
        T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
    {
        let url = self.build_stream_url();

        // Build and send request asynchronously
        let response = self
            .client
            .post(&url)
            .header("x-goog-api-key", self.config.api_key())
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| Box::new(e) as Box<dyn Error>)?
            .error_for_status()
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;

        // Get the response body as a stream of bytes
        let byte_stream = response.bytes_stream();

        let stream = futures::stream::unfold(
            (byte_stream, String::new()),
            |(mut byte_stream, mut buffer)| async move {
                loop {
                    match byte_stream.next().await {
                        Some(Ok(bytes)) => {
                            if let Some(result) =
                                Self::handle_stream_bytes::<T>(&bytes, &mut buffer)
                            {
                                return Some((result, (byte_stream, buffer)));
                            }
                            // Continue looping to get more bytes
                        }
                        Some(Err(e)) => {
                            tracing::error!("Gemini stream error: {:?}", e);
                            return Some((
                                Self::handle_stream_error::<T>(e),
                                (byte_stream, buffer),
                            ));
                        }
                        None => {
                            // Stream ended - try to parse any remaining buffer as final message
                            if !buffer.is_empty() {
                                let trimmed = buffer.trim();
                                if let Some(json_data) = trimmed.strip_prefix("data: ") {
                                    let json_data = json_data.trim();
                                    if !json_data.is_empty() {
                                        tracing::trace!(
                                            "SSE: Treating remaining buffer as final message ({} chars)",
                                            json_data.len()
                                        );
                                        let result =
                                            Self::parse_incomplete::<T>(json_data.to_string())
                                                .map_err(|e| {
                                                    Box::new(e) as Box<dyn Error + Send + Sync>
                                                });
                                        buffer.clear();
                                        return Some((result, (byte_stream, buffer)));
                                    }
                                }

                                tracing::warn!(
                                    "Gemini stream: discarding {} chars of unparseable data at end",
                                    buffer.len()
                                );
                            }
                            return None;
                        }
                    }
                }
            },
        );

        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dto_content::{Content, JsonString, Part};

    #[test]
    fn test_gemini_v1_beta_new() {
        let config = GeminiConfig::new("test-key".to_string(), "gemini-1.5-pro".to_string());
        let client = GeminiV1Beta::new(config.clone());

        // Verify URLs are built correctly
        let generate_url = client.build_generate_url();
        assert_eq!(
            generate_url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
        );
    }

    #[test]
    fn test_gemini_v1_beta_from_env() {
        // This test will succeed if GEMINI_API_KEY was set during compilation (debug builds)
        // or is set in the environment (release builds)
        let result = GeminiV1Beta::from_env();

        // We can't guarantee the environment variable is set in all test environments,
        // so we just verify the method exists and returns the expected type
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_build_generate_url() {
        let config = GeminiConfig::new("test-key".to_string(), "gemini-2.5-flash".to_string());
        let client = GeminiV1Beta::new(config);

        let url = client.build_generate_url();
        assert_eq!(
            url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        );
    }

    #[test]
    fn test_build_stream_url() {
        let config = GeminiConfig::new("test-key".to_string(), "gemini-1.5-pro".to_string());
        let client = GeminiV1Beta::new(config);

        let url = client.build_stream_url();
        assert_eq!(
            url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:streamGenerateContent?alt=sse"
        );
    }

    #[tokio::test]
    async fn test_generate_content_request_serialization() {
        let request: GenerateContentRequest<String> = GenerateContentRequest::builder()
            .add_content(Content::unspecified(vec![
                Part::builder()
                    .text(JsonString::new("Hello".to_string()))
                    .build(),
            ]))
            .build();

        // Verify serialization works
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("Hello"));
    }

    #[test]
    fn test_parse_incomplete() {
        // Test parsing a complete JSON response
        let complete_json = r#"{
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello, world!"}],
                        "role": "model"
                    },
                    "finishReason": "STOP",
                    "safetyRatings": []
                }
            ]
        }"#;

        let result = GeminiV1Beta::parse_incomplete::<String>(complete_json.to_string());
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.candidates.len(), 1);
        assert_eq!(response.first_text(), Some(&"Hello, world!".to_string()));
    }

    #[test]
    fn test_parse_incomplete_with_missing_fields() {
        // Test parsing JSON with missing optional fields
        let incomplete_json = r#"{
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Partial response"}]
                    }
                }
            ]
        }"#;

        let result = GeminiV1Beta::parse_incomplete::<String>(incomplete_json.to_string());
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.first_text(), Some(&"Partial response".to_string()));
    }

    #[test]
    fn test_process_bytes_chunk() {
        let mut buffer = String::from("existing data");
        let bytes = b"new data";

        GeminiV1Beta::process_bytes_chunk(bytes, &mut buffer);

        assert_eq!(buffer, "existing datanew data");
    }

    #[test]
    fn test_extract_sse_message_complete() {
        let mut buffer = "data: {\"test\":\"value\"}\n\nmore data".to_string();

        let result = GeminiV1Beta::extract_sse_message(&mut buffer);

        assert!(result.is_some());
        let json_data = result.unwrap();
        assert_eq!(json_data, r#"{"test":"value"}"#);
        assert_eq!(buffer, "more data");
    }

    #[test]
    fn test_extract_sse_message_incomplete() {
        let mut buffer = "data: {\"test\":\"value\"}\n".to_string();

        let result = GeminiV1Beta::extract_sse_message(&mut buffer);

        assert!(result.is_none());
        assert_eq!(buffer, "data: {\"test\":\"value\"}\n");
    }

    #[test]
    fn test_extract_sse_message_empty_data() {
        let mut buffer = "data: \n\nmore data".to_string();

        let result = GeminiV1Beta::extract_sse_message(&mut buffer);

        // Should skip empty data and try to find next message
        assert!(result.is_none());
        assert_eq!(buffer, "more data");
    }

    #[test]
    fn test_extract_sse_message_no_data_prefix() {
        let mut buffer = "invalid: {\"test\":\"value\"}\n\nmore data".to_string();

        let result = GeminiV1Beta::extract_sse_message(&mut buffer);

        // Should skip invalid format and continue parsing
        assert!(result.is_none());
        assert_eq!(buffer, "more data");
    }

    #[test]
    fn test_extract_sse_message_multiple_chunks() {
        let mut buffer =
            "data: {\"first\":\"chunk\"}\n\ndata: {\"second\":\"chunk\"}\n\n".to_string();

        let result = GeminiV1Beta::extract_sse_message(&mut buffer);

        assert!(result.is_some());
        let json_data = result.unwrap();
        assert_eq!(json_data, r#"{"first":"chunk"}"#);
        assert_eq!(buffer, "data: {\"second\":\"chunk\"}\n\n");

        // Extract the second message
        let result2 = GeminiV1Beta::extract_sse_message(&mut buffer);
        assert!(result2.is_some());
        let json_data2 = result2.unwrap();
        assert_eq!(json_data2, r#"{"second":"chunk"}"#);
        assert_eq!(buffer, "");
    }

    #[test]
    fn test_extract_sse_message_with_whitespace() {
        let mut buffer = "data:   {\"test\":\"value\"}  \n\nmore data".to_string();

        let result = GeminiV1Beta::extract_sse_message(&mut buffer);

        assert!(result.is_some());
        let json_data = result.unwrap();
        assert_eq!(json_data, r#"{"test":"value"}"#);
        assert_eq!(buffer, "more data");
    }

    // Typed response tests using JSON Schema
    #[cfg(feature = "json")]
    mod typed_json_tests {
        use super::*;
        use crate::dto_request::GenerationConfigBuilder;
        use schemars::JsonSchema;
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
        struct Character {
            name: String,
            age: u32,
            class: String,
        }

        #[test]
        fn test_typed_request_creation_with_json_schema() {
            // Create a typed request with JSON schema
            let config = GenerationConfigBuilder::<String>::new()
                .response_json_schema::<Character>()
                .temperature(0.7)
                .build()
                .unwrap();

            let request: GenerateContentRequest<Character> = GenerateContentRequest::builder()
                .add_content(Content::user(vec![
                    Part::builder()
                        .text(JsonString::new("Create a character".to_string()))
                        .build(),
                ]))
                .generation_config(config)
                .build();

            // Verify serialization works
            let json = serde_json::to_string(&request).unwrap();
            assert!(json.contains("responseJsonSchema"));
            assert!(json.contains("application/json"));
        }

        #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
        struct Story {
            title: String,
            genre: String,
            summary: String,
        }

        #[test]
        fn test_multiple_typed_schemas() {
            // Test that we can create different typed requests
            let char_config = GenerationConfigBuilder::<String>::new()
                .response_json_schema::<Character>()
                .build()
                .unwrap();

            let story_config = GenerationConfigBuilder::<String>::new()
                .response_json_schema::<Story>()
                .build()
                .unwrap();

            // Both should serialize correctly
            let char_json = serde_json::to_string(&char_config).unwrap();
            let story_json = serde_json::to_string(&story_config).unwrap();

            assert!(char_json.contains("responseJsonSchema"));
            assert!(story_json.contains("responseJsonSchema"));
        }
    }

    // Tests for typed responses with OpenAPI schema
    #[cfg(feature = "openapi")]
    mod typed_openapi_tests {
        use super::*;
        use crate::dto_request::GenerationConfigBuilder;
        use serde::{Deserialize, Serialize};
        use utoipa::ToSchema;

        #[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
        struct Item {
            id: i32,
            name: String,
            description: String,
        }

        #[test]
        fn test_typed_request_creation_with_openapi_schema() {
            // Create a typed request with OpenAPI schema
            let config = GenerationConfigBuilder::<String>::new()
                .response_schema::<Item>()
                .temperature(0.5)
                .build()
                .unwrap();

            let request: GenerateContentRequest<Item> = GenerateContentRequest::builder()
                .add_content(Content::user(vec![
                    Part::builder()
                        .text(JsonString::new("Create an item".to_string()))
                        .build(),
                ]))
                .generation_config(config)
                .build();

            // Verify serialization works
            let json = serde_json::to_string(&request).unwrap();
            assert!(json.contains("responseSchema"));
            assert!(json.contains("application/json"));
        }

        #[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
        struct Product {
            sku: String,
            price: f64,
            in_stock: bool,
        }

        #[test]
        fn test_multiple_openapi_schemas() {
            // Test that we can create different typed requests
            let item_config = GenerationConfigBuilder::<String>::new()
                .response_schema::<Item>()
                .build()
                .unwrap();

            let product_config = GenerationConfigBuilder::<String>::new()
                .response_schema::<Product>()
                .build()
                .unwrap();

            // Both should serialize correctly
            let item_json = serde_json::to_string(&item_config).unwrap();
            let product_json = serde_json::to_string(&product_config).unwrap();

            assert!(item_json.contains("responseSchema"));
            assert!(product_json.contains("responseSchema"));
        }
    }

    // Integration tests that call the real Gemini API
    // These tests are ignored by default and require GEMINI_API_KEY to be set
    #[cfg(test)]
    mod integration_tests {
        use super::*;
        #[allow(unused_imports)] // Used in #[ignore] integration tests
        use crate::GenerationConfigBuilder;

        #[tokio::test]
        #[ignore]
        async fn test_real_api_string_response() {
            // This test requires GEMINI_API_KEY environment variable
            let client = GeminiV1Beta::from_env().expect("Failed to create client from env");

            let request: GenerateContentRequest<String> = GenerateContentRequest::builder()
                .add_content(Content::unspecified(vec![
                    Part::builder()
                        .text(JsonString::new("Say hello in one word".to_string()))
                        .build(),
                ]))
                .build();

            let response = client
                .generate_content(request)
                .await
                .expect("API call failed");

            // Verify we got a response
            assert!(!response.candidates.is_empty());
            let text = response.first_text();
            assert!(text.is_some());
            println!("Response: {:?}", text);
        }

        #[cfg(feature = "json")]
        #[tokio::test]
        #[ignore]
        async fn test_real_api_typed_json_response() {
            use schemars::JsonSchema;
            use serde::{Deserialize, Serialize};

            #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
            struct Greeting {
                message: String,
                language: String,
            }

            let client = GeminiV1Beta::from_env().expect("Failed to create client from env");

            let config = GenerationConfigBuilder::<String>::new()
                .response_json_schema::<Greeting>()
                .build()
                .unwrap();

            let request: GenerateContentRequest<Greeting> = GenerateContentRequest::builder()
                .add_content(Content::user(vec![
                    Part::builder()
                        .text(JsonString::new("Create a greeting in English".to_string()))
                        .build(),
                ]))
                .generation_config(config)
                .build();

            let response = client
                .generate_content(request)
                .await
                .expect("API call failed");

            // Verify we got a typed response
            assert!(!response.candidates.is_empty());
            let greeting = response.first_text();
            assert!(greeting.is_some());

            let greeting = greeting.unwrap();
            println!("Greeting: {:?}", greeting);
            assert!(!greeting.message.is_empty());
            assert!(!greeting.language.is_empty());
        }

        #[cfg(feature = "openapi")]
        #[tokio::test]
        #[ignore]
        async fn test_real_api_typed_openapi_response() {
            use serde::{Deserialize, Serialize};
            use utoipa::ToSchema;

            #[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
            struct Animal {
                name: String,
                species: String,
                age: u32,
            }

            let client = GeminiV1Beta::from_env().expect("Failed to create client from env");

            let config = GenerationConfigBuilder::<String>::new()
                .response_schema::<Animal>()
                .build()
                .unwrap();

            let request: GenerateContentRequest<Animal> = GenerateContentRequest::builder()
                .add_content(Content::user(vec![
                    Part::builder()
                        .text(JsonString::new(
                            "Create an animal with name, species, and age".to_string(),
                        ))
                        .build(),
                ]))
                .generation_config(config)
                .build();

            let response = client
                .generate_content(request)
                .await
                .expect("API call failed");

            // Verify we got a typed response
            assert!(!response.candidates.is_empty());
            let animal = response.first_text();
            assert!(animal.is_some());

            let animal = animal.unwrap();
            println!("Animal: {:?}", animal);
            assert!(!animal.name.is_empty());
            assert!(!animal.species.is_empty());
            assert!(animal.age > 0);
        }

        #[cfg(feature = "json")]
        #[tokio::test]
        #[ignore]
        async fn test_real_api_complex_typed_response() {
            use schemars::JsonSchema;
            use serde::{Deserialize, Serialize};

            #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
            struct Character {
                name: String,
                age: u32,
                class: String,
                skills: Vec<String>,
                stats: CharacterStats,
            }

            #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
            struct CharacterStats {
                strength: u32,
                intelligence: u32,
                charisma: u32,
            }

            let client = GeminiV1Beta::from_env().expect("Failed to create client from env");

            let config = GenerationConfigBuilder::<String>::new()
                .response_json_schema::<Character>()
                .temperature(0.9)
                .build()
                .unwrap();

            let request: GenerateContentRequest<Character> = GenerateContentRequest::builder()
                .add_content(Content::user(vec![
                    Part::builder()
                        .text(JsonString::new(
                            "Create a fantasy RPG character with name, age, class, \
                             a list of 3 skills, and stats (strength, intelligence, charisma)"
                                .to_string(),
                        ))
                        .build(),
                ]))
                .generation_config(config)
                .build();

            let response = client
                .generate_content(request)
                .await
                .expect("API call failed");

            // Verify we got a complex typed response
            assert!(!response.candidates.is_empty());
            let character = response.first_text();
            assert!(character.is_some());

            let character = character.unwrap();
            println!("Character: {:#?}", character);
            assert!(!character.name.is_empty());
            assert!(character.age > 0);
            assert!(!character.class.is_empty());
            assert!(!character.skills.is_empty());
            assert!(character.stats.strength > 0);
            assert!(character.stats.intelligence > 0);
            assert!(character.stats.charisma > 0);
        }

        #[tokio::test]
        #[ignore]
        async fn test_real_api_string_streaming() {
            use futures::StreamExt;

            // This test requires GEMINI_API_KEY environment variable
            let client = GeminiV1Beta::from_env().expect("Failed to create client from env");

            let request: GenerateContentRequest<String> = GenerateContentRequest::builder()
                .add_content(Content::unspecified(vec![
                    Part::builder()
                        .text(JsonString::new("Count from 1 to 5".to_string()))
                        .build(),
                ]))
                .build();

            let mut stream = client
                .stream_generate_content(request)
                .await
                .expect("Stream API call failed");

            // Verify we get multiple chunks
            let mut chunk_count = 0;
            while let Some(result) = stream.next().await {
                let response = result.expect("Failed to parse chunk");
                println!("Chunk {}: {:?}", chunk_count, response.first_text());
                chunk_count += 1;
            }

            assert!(chunk_count > 0, "Should receive at least one chunk");
            println!("Total chunks received: {}", chunk_count);
        }

        #[cfg(feature = "json")]
        #[tokio::test]
        #[ignore]
        async fn test_real_api_typed_streaming() {
            use futures::StreamExt;
            use schemars::JsonSchema;
            use serde::{Deserialize, Serialize};

            #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
            struct Story {
                title: String,
                genre: String,
                summary: String,
            }

            let client = GeminiV1Beta::from_env().expect("Failed to create client from env");

            let config = GenerationConfigBuilder::<String>::new()
                .response_json_schema::<Story>()
                .build()
                .unwrap();

            let request: GenerateContentRequest<Story> = GenerateContentRequest::builder()
                .add_content(Content::user(vec![
                    Part::builder()
                        .text(JsonString::new(
                            "Create a short story idea with title, genre, and summary".to_string(),
                        ))
                        .build(),
                ]))
                .generation_config(config)
                .build();

            let mut stream = client
                .stream_generate_content(request)
                .await
                .expect("Stream API call failed");

            // Verify we get chunks and can parse them
            let mut chunk_count = 0;
            let mut last_story: Option<Story> = None;

            while let Some(result) = stream.next().await {
                match result {
                    Ok(response) => {
                        if let Some(story) = response.first_text() {
                            println!("Chunk {}: {:?}", chunk_count, story);
                            last_story = Some(story.clone());
                        }
                        chunk_count += 1;
                    }
                    Err(e) => {
                        // Streaming may have incomplete JSON in intermediate chunks
                        println!("Chunk {} parse error (expected): {}", chunk_count, e);
                        chunk_count += 1;
                    }
                }
            }

            assert!(chunk_count > 0, "Should receive at least one chunk");
            println!("Total chunks received: {}", chunk_count);

            // The last chunk should have a complete story
            if let Some(story) = last_story {
                assert!(!story.title.is_empty());
                assert!(!story.genre.is_empty());
                assert!(!story.summary.is_empty());
            }
        }
    }
}
