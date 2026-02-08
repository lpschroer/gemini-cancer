//! Chat wrapper for managing multi-turn conversations with the Gemini API
//!
//! This module provides types for wrapping messages with role information and
//! automatically managing conversation history across multiple turns.

use crate::api::GeminiApi;
use crate::dto_content::{Content, JsonString, Part, Role};
use serde::{Deserialize, Serialize};

/// User message wrapper with automatic role tagging
///
/// Wraps `Content<T>` and automatically serializes with `role: "user"`.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "T: Serialize + 'static",
    deserialize = "T: serde::de::DeserializeOwned + 'static"
))]
pub struct UserMessage<T = String> {
    /// Parts of the message content
    pub parts: Vec<Part<T>>,
}

impl<T> UserMessage<T> {
    /// Creates a new user message from parts
    pub fn new(parts: Vec<Part<T>>) -> Self {
        Self { parts }
    }
}

impl<T> From<Vec<Part<T>>> for UserMessage<T> {
    fn from(parts: Vec<Part<T>>) -> Self {
        Self::new(parts)
    }
}

impl From<String> for UserMessage<String> {
    fn from(text: String) -> Self {
        Self::new(vec![Part::builder().text(JsonString::new(text)).build()])
    }
}

impl From<&str> for UserMessage<String> {
    fn from(text: &str) -> Self {
        Self::from(text.to_string())
    }
}

/// Model message wrapper with automatic role tagging
///
/// Wraps `Content<T>` and automatically serializes with `role: "model"`.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "T: Serialize + 'static",
    deserialize = "T: serde::de::DeserializeOwned + 'static"
))]
pub struct ModelMessage<T = String> {
    /// Parts of the message content
    pub parts: Vec<Part<T>>,
}

impl<T> ModelMessage<T> {
    /// Creates a new model message from parts
    pub fn new(parts: Vec<Part<T>>) -> Self {
        Self { parts }
    }
}

impl<T> From<Vec<Part<T>>> for ModelMessage<T> {
    fn from(parts: Vec<Part<T>>) -> Self {
        Self::new(parts)
    }
}

/// Chat message enum that holds either a user or model message
///
/// Serializes with the correct role tag based on the variant.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "role")]
#[serde(bound(
    serialize = "T: Serialize + 'static",
    deserialize = "T: serde::de::DeserializeOwned + 'static"
))]
pub enum ChatMessage<T = String> {
    /// Message from the user
    #[serde(rename = "user")]
    User(UserMessage<T>),
    /// Message from the model
    #[serde(rename = "model")]
    Model(ModelMessage<T>),
}

impl<T> ChatMessage<T> {
    /// Converts the chat message to a `Content<T>` with the appropriate role
    pub fn into_content(self) -> Content<T> {
        match self {
            ChatMessage::User(msg) => Content {
                role: Some(Role::User),
                parts: msg.parts,
            },
            ChatMessage::Model(msg) => Content {
                role: Some(Role::Model),
                parts: msg.parts,
            },
        }
    }
}

/// Chat wrapper for managing multi-turn conversations with the Gemini API
///
/// `GeminiChat` maintains conversation history as `Vec<serde_json::Value>` to allow
/// mixed message types (e.g., user sends `String`, model responds with structured schema)
/// and simplifies persistence.
///
/// # Type Parameters
/// * `A` - The GeminiApi implementor type
#[allow(dead_code)]
pub struct GeminiChat<A> {
    /// The underlying API client
    api: A,
    /// Conversation history stored as JSON values for flexibility
    history: Vec<serde_json::Value>,
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
    /// * `history` - Previously saved conversation history as JSON values
    ///
    /// # Example
    /// ```ignore
    /// let client = GeminiClient::new(api_key);
    /// let saved_history: Vec<serde_json::Value> = load_from_storage();
    /// let chat = GeminiChat::from_history(client, saved_history);
    /// ```
    pub fn from_history(api: A, history: Vec<serde_json::Value>) -> Self {
        Self { api, history }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_message_from_parts() {
        let parts = vec![
            Part::builder()
                .text(JsonString::new("Hello".to_string()))
                .build(),
        ];
        let msg = UserMessage::from(parts);
        assert_eq!(msg.parts.len(), 1);
        assert_eq!(msg.parts[0].text().unwrap(), "Hello");
    }

    #[test]
    fn test_user_message_from_string() {
        let msg = UserMessage::from("Hello, world!");
        assert_eq!(msg.parts.len(), 1);
        assert_eq!(msg.parts[0].text().unwrap(), "Hello, world!");
    }

    #[test]
    fn test_user_message_from_str() {
        let msg = UserMessage::from("Test message");
        assert_eq!(msg.parts.len(), 1);
        assert_eq!(msg.parts[0].text().unwrap(), "Test message");
    }

    #[test]
    fn test_model_message_from_parts() {
        let parts = vec![
            Part::builder()
                .text(JsonString::new("Response".to_string()))
                .build(),
        ];
        let msg = ModelMessage::from(parts);
        assert_eq!(msg.parts.len(), 1);
        assert_eq!(msg.parts[0].text().unwrap(), "Response");
    }

    #[test]
    fn test_user_message_serialization() {
        let msg = ChatMessage::User(UserMessage::from("Hello"));
        let json = serde_json::to_value(&msg).expect("Failed to serialize");

        assert_eq!(json["role"], "user");
        assert!(json["parts"].is_array());
        assert_eq!(json["parts"][0]["text"], "Hello");
    }

    #[test]
    fn test_model_message_serialization() {
        let msg = ChatMessage::Model(ModelMessage::from(vec![
            Part::builder()
                .text(JsonString::new("Hi there".to_string()))
                .build(),
        ]));
        let json = serde_json::to_value(&msg).expect("Failed to serialize");

        assert_eq!(json["role"], "model");
        assert!(json["parts"].is_array());
        assert_eq!(json["parts"][0]["text"], "Hi there");
    }

    #[test]
    fn test_chat_message_user_serialization() {
        let msg = ChatMessage::User(UserMessage::from("User input"));
        let json = serde_json::to_value(&msg).expect("Failed to serialize");

        assert_eq!(json["role"], "user");
        assert_eq!(json["parts"][0]["text"], "User input");
    }

    #[test]
    fn test_chat_message_model_serialization() {
        let msg = ChatMessage::Model(ModelMessage::from(vec![
            Part::builder()
                .text(JsonString::new("Model output".to_string()))
                .build(),
        ]));
        let json = serde_json::to_value(&msg).expect("Failed to serialize");

        assert_eq!(json["role"], "model");
        assert_eq!(json["parts"][0]["text"], "Model output");
    }

    #[test]
    fn test_user_message_deserialization() {
        let json = r#"{
            "role": "user",
            "parts": [{"text": "Hello"}]
        }"#;

        let msg: ChatMessage<String> = serde_json::from_str(json).expect("Failed to deserialize");
        match msg {
            ChatMessage::User(user_msg) => {
                assert_eq!(user_msg.parts.len(), 1);
                assert_eq!(user_msg.parts[0].text().unwrap(), "Hello");
            }
            _ => panic!("Expected User variant"),
        }
    }

    #[test]
    fn test_model_message_deserialization() {
        let json = r#"{
            "role": "model",
            "parts": [{"text": "Response"}]
        }"#;

        let msg: ChatMessage<String> = serde_json::from_str(json).expect("Failed to deserialize");
        match msg {
            ChatMessage::Model(model_msg) => {
                assert_eq!(model_msg.parts.len(), 1);
                assert_eq!(model_msg.parts[0].text().unwrap(), "Response");
            }
            _ => panic!("Expected Model variant"),
        }
    }

    #[test]
    fn test_chat_message_deserialization() {
        let user_json = r#"{
            "role": "user",
            "parts": [{"text": "Question"}]
        }"#;

        let user_msg: ChatMessage<String> =
            serde_json::from_str(user_json).expect("Failed to deserialize user message");
        match user_msg {
            ChatMessage::User(msg) => {
                assert_eq!(msg.parts[0].text().unwrap(), "Question");
            }
            _ => panic!("Expected User variant"),
        }

        let model_json = r#"{
            "role": "model",
            "parts": [{"text": "Answer"}]
        }"#;

        let model_msg: ChatMessage<String> =
            serde_json::from_str(model_json).expect("Failed to deserialize model message");
        match model_msg {
            ChatMessage::Model(msg) => {
                assert_eq!(msg.parts[0].text().unwrap(), "Answer");
            }
            _ => panic!("Expected Model variant"),
        }
    }

    #[test]
    fn test_chat_message_into_content() {
        let user_msg = ChatMessage::User(UserMessage::from("Hello"));
        let content = user_msg.into_content();
        assert_eq!(content.role, Some(Role::User));
        assert_eq!(content.parts.len(), 1);

        let model_msg = ChatMessage::Model(ModelMessage::from(vec![
            Part::builder()
                .text(JsonString::new("Hi".to_string()))
                .build(),
        ]));
        let content = model_msg.into_content();
        assert_eq!(content.role, Some(Role::Model));
        assert_eq!(content.parts.len(), 1);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let original = ChatMessage::User(UserMessage::from("Test"));
        let json = serde_json::to_string(&original).expect("Failed to serialize");
        let deserialized: ChatMessage<String> =
            serde_json::from_str(&json).expect("Failed to deserialize");

        match deserialized {
            ChatMessage::User(msg) => {
                assert_eq!(msg.parts[0].text().unwrap(), "Test");
            }
            _ => panic!("Expected User variant"),
        }
    }

    #[test]
    fn test_gemini_chat_new() {
        struct MockApi;
        #[async_trait::async_trait]
        impl GeminiApi for MockApi {
            async fn generate_content<T>(
                &self,
                _request: crate::dto_request::GenerateContentRequest<T>,
            ) -> Result<crate::dto_response::GenerateContentResponse<T>, Box<dyn std::error::Error>>
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
                _request: crate::dto_request::GenerateContentRequest<T>,
            ) -> Result<crate::dto_response::GenerateContentResponse<T>, Box<dyn std::error::Error>>
            where
                T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
            {
                unimplemented!()
            }
        }

        let history = vec![
            serde_json::json!({
                "role": "user",
                "parts": [{"text": "Hello"}]
            }),
            serde_json::json!({
                "role": "model",
                "parts": [{"text": "Hi there!"}]
            }),
        ];

        let chat = GeminiChat::from_history(MockApi, history.clone());
        assert_eq!(chat.history.len(), 2);
        assert_eq!(chat.history[0]["role"], "user");
        assert_eq!(chat.history[1]["role"], "model");
    }
}
