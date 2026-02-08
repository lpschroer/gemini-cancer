//! Chat wrapper for managing multi-turn conversations with the Gemini API
//!
//! This module provides utilities for managing conversation history
//! automatically across multiple turns.

use crate::api::GeminiApi;
use crate::dto_content::{Content, JsonString, Part};
use crate::dto_request::{GenerateContentRequest, GenerationConfig, SafetySetting};
use crate::dto_response::GenerateContentResponse;
use std::error::Error;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::GeminiApi;

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
}
