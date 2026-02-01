use serde::{Deserialize, Serialize};

use super::dto_content::Content;
use super::dto_request::SafetyRating;

/// Response from generateContent API call
///
/// Generic over the text content type `T`, which defaults to `String`.
/// When using `response_json_schema`, specify the type parameter and JSON deserialization
/// happens automatically during response parsing.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "T: Serialize + 'static",
    deserialize = "T: serde::de::DeserializeOwned + 'static"
))]
pub struct GenerateContentResponse<T = String> {
    /// Candidate responses from the model
    #[serde(default)]
    pub candidates: Vec<Candidate<T>>,

    /// Prompt feedback related to content filters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_feedback: Option<PromptFeedback>,

    /// Token usage metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<UsageMetadata>,
}

/// A candidate response
///
/// Generic over the text content type `T`, which defaults to `String`.
/// When using `response_json_schema`, the text content is automatically deserialized into `T`.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "T: Serialize + 'static",
    deserialize = "T: serde::de::DeserializeOwned + 'static"
))]
pub struct Candidate<T = String> {
    /// Generated content
    pub content: Content<T>,

    /// Reason why generation stopped
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,

    /// Safety ratings for the candidate
    #[serde(default)]
    pub safety_ratings: Vec<SafetyRating>,
}

/// Feedback about the prompt
#[derive(Debug, Serialize, Deserialize)]
pub struct PromptFeedback {
    /// Reason the prompt was blocked
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_reason: Option<String>,

    /// Safety ratings for the prompt
    #[serde(default)]
    pub safety_ratings: Vec<SafetyRating>,
}

/// Token usage metadata
#[derive(Debug, Serialize, Deserialize)]
pub struct UsageMetadata {
    /// Number of tokens in the prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_token_count: Option<i32>,

    /// Number of tokens in the response candidates
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates_token_count: Option<i32>,

    /// Total token count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_token_count: Option<i32>,
}

impl<T> GenerateContentResponse<T> {
    /// Returns the first candidate, if any
    pub fn first_candidate(&self) -> Option<&Candidate<T>> {
        self.candidates.first()
    }

    /// Returns the first candidate's content, if any
    pub fn first_content(&self) -> Option<&Content<T>> {
        self.first_candidate().map(|c| &c.content)
    }

    /// Returns the first candidate's first text, if any
    pub fn first_text(&self) -> Option<&T> {
        self.first_content().and_then(|c| c.first_text())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dto_content::{JsonString, Part, Role};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestSchema {
        name: String,
        age: u32,
    }

    #[test]
    fn test_response_automatic_deserialization() {
        // Simulate JSON response from API with double-escaped JSON
        let json = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "{\"name\":\"Alice\",\"age\":30}"}]
                },
                "finish_reason": "STOP",
                "safety_ratings": []
            }]
        }"#;

        let response: GenerateContentResponse<TestSchema> = serde_json::from_str(json).unwrap();
        assert_eq!(response.candidates.len(), 1);
        assert_eq!(response.first_text().unwrap().name, "Alice");
        assert_eq!(response.first_text().unwrap().age, 30);
    }

    #[test]
    fn test_response_helpers() {
        let response = GenerateContentResponse {
            candidates: vec![Candidate {
                content: Content {
                    role: Some(Role::Model),
                    parts: vec![Part {
                        text: Some(JsonString::new("Hello".to_string())),
                        inline_data: None,
                        function_call: None,
                        function_response: None,
                        file_data: None,
                        executable_code: None,
                        code_execution_result: None,
                        video_metadata: None,
                    }],
                },
                finish_reason: None,
                safety_ratings: vec![],
            }],
            prompt_feedback: None,
            usage_metadata: None,
        };

        assert!(response.first_candidate().is_some());
        assert!(response.first_content().is_some());
        assert_eq!(response.first_text(), Some(&"Hello".to_string()));
    }

    #[test]
    fn test_end_to_end_automatic_deserialization() {
        // Simulate a raw API response with JSON content
        let json = r#"{
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "{\"name\":\"Alice\",\"age\":30}"}]
                    },
                    "finish_reason": "STOP",
                    "safety_ratings": []
                },
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "{\"name\":\"Bob\",\"age\":25}"}]
                    },
                    "finish_reason": "STOP",
                    "safety_ratings": []
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 10,
                "candidates_token_count": 20,
                "total_token_count": 30
            }
        }"#;

        // Deserialize directly into typed response
        let typed_response: GenerateContentResponse<TestSchema> =
            serde_json::from_str(json).unwrap();

        // Verify structure is preserved
        assert_eq!(typed_response.candidates.len(), 2);
        assert!(typed_response.usage_metadata.is_some());

        // Verify first candidate
        let first = &typed_response.candidates[0];
        assert_eq!(first.finish_reason, Some("STOP".to_string()));
        let first_schema = first.content.parts[0].text().unwrap();
        assert_eq!(first_schema.name, "Alice");
        assert_eq!(first_schema.age, 30);

        // Verify second candidate
        let second = &typed_response.candidates[1];
        let second_schema = second.content.parts[0].text().unwrap();
        assert_eq!(second_schema.name, "Bob");
        assert_eq!(second_schema.age, 25);

        // Verify helper methods work with typed response
        assert_eq!(typed_response.first_text().unwrap().name, "Alice");
    }

    #[test]
    fn test_deserialization_preserves_metadata() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "{\"name\":\"Test\",\"age\":20}"}]
                },
                "finish_reason": "MAX_TOKENS",
                "safety_ratings": []
            }],
            "prompt_feedback": {
                "safety_ratings": []
            },
            "usage_metadata": {
                "prompt_token_count": 5,
                "candidates_token_count": 15,
                "total_token_count": 20
            }
        }"#;

        let typed_response: GenerateContentResponse<TestSchema> =
            serde_json::from_str(json).unwrap();

        // All metadata should be preserved
        assert!(typed_response.prompt_feedback.is_some());
        assert_eq!(
            typed_response
                .usage_metadata
                .as_ref()
                .unwrap()
                .total_token_count,
            Some(20)
        );
        assert_eq!(
            typed_response.candidates[0].finish_reason,
            Some("MAX_TOKENS".to_string())
        );
    }

    #[test]
    fn test_string_type_passthrough() {
        // For String type, content should pass through without double-escaping
        let json = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Plain text response"}]
                },
                "finish_reason": "STOP",
                "safety_ratings": []
            }]
        }"#;

        let response: GenerateContentResponse<String> = serde_json::from_str(json).unwrap();
        assert_eq!(
            response.first_text(),
            Some(&"Plain text response".to_string())
        );
    }
}
