//! Content DTOs for the Gemini API
//!
//! This module contains the core content structures used in Gemini API requests and responses,
//! including the `Content` struct and all its component parts.
//!
//! ## Generic Text Content
//!
//! `Content` and `Part` are generic over the text content type, defaulting to `String`.
//! When using `response_json_schema`, specify the desired type parameter and deserialization
//! happens automatically:
//!
//! ```rust,ignore
//! let response: GenerateContentResponse<MySchema> = api.generate_content(request).await?;
//! let my_data = response.first_text(); // Returns Option<&MySchema>
//! ```

use serde::{Deserialize, Serialize};

use crate::dto_request::MimeType;

/// Role of the content creator
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Content from the user
    User,
    /// Content from the model
    Model,
}

/// Content object containing parts
///
/// Generic over the text content type `T`, which defaults to `String`.
/// When using `response_json_schema`, specify the type parameter and JSON deserialization
/// happens automatically during response parsing.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "T: Serialize + 'static",
    deserialize = "T: serde::de::DeserializeOwned + 'static"
))]
pub struct Content<T = String> {
    /// Optional role of the content creator (user or model)
    /// Useful to set for multi-turn conversations, otherwise can be left blank or unset
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,

    /// Parts of the content
    pub parts: Vec<Part<T>>,
}

/// A wrapper type for JSON values stored as strings
///
/// This type handles fields that need to be JSON-encoded and then serialized as strings.
/// For example: `{"key": "{\"inner_key\": \"value\"}"}`
///
/// Special behavior for `JsonString<String>`:
/// - Strings are serialized directly without additional JSON encoding
/// - This prevents double-escaping of string values
#[derive(Debug, Clone, PartialEq)]
pub struct JsonString<T> {
    inner: T,
}

impl<T> JsonString<T> {
    /// Create a new JSON string wrapper
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    /// Get a reference to the inner value
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Unwrap the inner value
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T: Serialize + 'static> Serialize for JsonString<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Special case for String to avoid double-escaping
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<String>() {
            // SAFETY: We just checked that T is String
            let s = unsafe { &*(&self.inner as *const T as *const String) };
            return serializer.serialize_str(s);
        }

        // For all other types, JSON-encode then serialize as string
        let json_string = serde_json::to_string(&self.inner).map_err(serde::ser::Error::custom)?;
        serializer.serialize_str(&json_string)
    }
}

impl<'de, T: serde::de::DeserializeOwned + 'static> Deserialize<'de> for JsonString<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;

        // Special case for String to avoid double-escaping
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<String>() {
            // SAFETY: We just checked that T is String
            let inner = unsafe { std::ptr::read(&s as *const String as *const T) };
            std::mem::forget(s); // Prevent double-free
            return Ok(JsonString { inner });
        }

        // For all other types, parse the JSON string
        let inner = serde_json::from_str(&s).map_err(serde::de::Error::custom)?;
        Ok(JsonString { inner })
    }
}

/// A part of the content
///
/// Generic over the text content type `T`, which defaults to `String`.
/// When using `response_json_schema`, specify the type parameter and JSON deserialization
/// happens automatically during response parsing.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "T: Serialize + 'static",
    deserialize = "T: serde::de::DeserializeOwned + 'static"
))]
pub struct Part<T = String> {
    /// Inline text content
    ///
    /// When `T` is not `String`, this field automatically deserializes JSON strings into the target type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<JsonString<T>>,

    /// Inline media bytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<Blob>,

    /// A predicted function call returned from the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,

    /// The result output of a function call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_response: Option<FunctionResponse>,

    /// URI based data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_data: Option<FileData>,

    /// Code generated by the model that is meant to be executed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub executable_code: Option<ExecutableCode>,

    /// Result of executing the ExecutableCode
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_execution_result: Option<CodeExecutionResult>,

    /// Video metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_metadata: Option<VideoMetadata>,
}

/// Raw media bytes
#[derive(Debug, Serialize, Deserialize)]
pub struct Blob {
    /// The IANA standard MIME type of the source data
    pub mime_type: MimeType,

    /// Raw bytes for media formats (base64-encoded)
    pub data: String,
}

/// A predicted function call returned from the model
#[derive(Debug, Serialize, Deserialize)]
pub struct FunctionCall {
    /// The name of the function to call
    pub name: String,

    /// The function parameters and values in JSON object format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,
}

/// The result output from a function call
#[derive(Debug, Serialize, Deserialize)]
pub struct FunctionResponse {
    /// The name of the function
    pub name: String,

    /// The function response in JSON object format
    pub response: serde_json::Value,
}

/// URI based data
#[derive(Debug, Serialize, Deserialize)]
pub struct FileData {
    /// The IANA standard MIME type of the source data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<MimeType>,

    /// URI
    pub file_uri: String,
}

/// Code generated by the model that is meant to be executed
#[derive(Debug, Serialize, Deserialize)]
pub struct ExecutableCode {
    /// Programming language of the code
    pub language: String,

    /// The code to be executed
    pub code: String,
}

/// Result of executing the ExecutableCode
#[derive(Debug, Serialize, Deserialize)]
pub struct CodeExecutionResult {
    /// Outcome of the code execution
    pub outcome: String,

    /// Contains stdout when code execution is successful, stderr or other description otherwise
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
}

/// Metadata describing the input video content
#[derive(Debug, Serialize, Deserialize)]
pub struct VideoMetadata {
    /// The start offset of the video
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_offset: Option<String>,

    /// The end offset of the video
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_offset: Option<String>,

    /// The frame rate of the video (0.0, 24.0]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<f32>,
}

impl<T> Part<T> {
    /// Returns true if this part has text content
    pub fn has_text(&self) -> bool {
        self.text.is_some()
    }

    /// Get a reference to the text content
    pub fn text(&self) -> Option<&T> {
        self.text.as_ref().map(|field| field.inner())
    }

    /// Get a mutable reference to the text content
    pub fn text_mut(&mut self) -> Option<&mut T> {
        self.text.as_mut().map(|field| &mut field.inner)
    }

    /// Consume self and return the text content
    pub fn into_text(self) -> Option<T> {
        self.text.map(|field| field.into_inner())
    }
}

impl<T> Content<T> {
    /// Returns the first part, if any
    pub fn first_part(&self) -> Option<&Part<T>> {
        self.parts.first()
    }

    /// Returns the first part's text, if any
    pub fn first_text(&self) -> Option<&T> {
        self.first_part().and_then(|p| p.text())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestSchema {
        name: String,
        age: u32,
    }

    #[test]
    fn test_part_text_helpers() {
        let part = Part {
            text: Some(JsonString::new("Hello".to_string())),
            inline_data: None,
            function_call: None,
            function_response: None,
            file_data: None,
            executable_code: None,
            code_execution_result: None,
            video_metadata: None,
        };

        assert!(part.has_text());
        assert_eq!(part.text(), Some(&"Hello".to_string()));
        assert_eq!(part.into_text(), Some("Hello".to_string()));
    }

    #[test]
    fn test_part_text_mut() {
        let mut part = Part {
            text: Some(JsonString::new(TestSchema {
                name: "Alice".to_string(),
                age: 30,
            })),
            inline_data: None,
            function_call: None,
            function_response: None,
            file_data: None,
            executable_code: None,
            code_execution_result: None,
            video_metadata: None,
        };

        if let Some(schema) = part.text_mut() {
            schema.age = 31;
        }

        assert_eq!(part.text().unwrap().age, 31);
    }

    #[test]
    fn test_part_has_text() {
        let part_with_text: Part<String> = Part {
            text: Some(JsonString::new("Hello".to_string())),
            inline_data: None,
            function_call: None,
            function_response: None,
            file_data: None,
            executable_code: None,
            code_execution_result: None,
            video_metadata: None,
        };

        let part_without_text: Part<String> = Part {
            text: None,
            inline_data: None,
            function_call: None,
            function_response: None,
            file_data: None,
            executable_code: None,
            code_execution_result: None,
            video_metadata: None,
        };

        assert!(part_with_text.has_text());
        assert!(!part_without_text.has_text());
    }

    #[test]
    fn test_content_automatic_deserialization() {
        // Simulate JSON response that would be automatically deserialized
        let json = r#"{
            "role": "model",
            "parts": [
                {"text": "{\"name\":\"Alice\",\"age\":30}"},
                {"text": "{\"name\":\"Bob\",\"age\":25}"}
            ]
        }"#;

        let content: Content<TestSchema> = serde_json::from_str(json).unwrap();
        assert_eq!(content.parts.len(), 2);
        assert_eq!(content.parts[0].text().unwrap().name, "Alice");
        assert_eq!(content.parts[0].text().unwrap().age, 30);
        assert_eq!(content.parts[1].text().unwrap().name, "Bob");
        assert_eq!(content.parts[1].text().unwrap().age, 25);
    }

    #[test]
    fn test_content_helpers() {
        let content = Content {
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
        };

        assert!(content.first_part().is_some());
        assert_eq!(content.first_text(), Some(&"Hello".to_string()));
    }

    #[test]
    fn test_content_string_passthrough() {
        // For String type, content should pass through without double-escaping
        let json = r#"{
            "role": "model",
            "parts": [
                {"text": "Plain text response"}
            ]
        }"#;

        let content: Content<String> = serde_json::from_str(json).unwrap();
        assert_eq!(
            content.first_text(),
            Some(&"Plain text response".to_string())
        );
    }

    #[test]
    fn test_json_string_serialize() {
        #[derive(Debug, Serialize)]
        struct Inner {
            key: String,
            value: i32,
        }

        let inner = Inner {
            key: "test".to_string(),
            value: 42,
        };

        let wrapped = JsonString::new(inner);
        let serialized = serde_json::to_string(&wrapped).unwrap();

        // Should be JSON-encoded as a string: the outer quotes are from to_string,
        // the inner escaped quotes are from JsonString
        assert_eq!(serialized, r#""{\"key\":\"test\",\"value\":42}""#);
    }

    #[test]
    fn test_json_string_deserialize() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct Inner {
            key: String,
            value: i32,
        }

        let json = r#""{\"key\":\"test\",\"value\":42}""#;
        let wrapped: JsonString<Inner> = serde_json::from_str(json).unwrap();

        assert_eq!(wrapped.inner().key, "test");
        assert_eq!(wrapped.inner().value, 42);
    }

    #[test]
    fn test_json_string_roundtrip() {
        let original = TestSchema {
            name: "Alice".to_string(),
            age: 30,
        };

        let wrapped = JsonString::new(original.clone());
        let serialized = serde_json::to_string(&wrapped).unwrap();
        let deserialized: JsonString<TestSchema> = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.inner(), &original);
    }

    #[test]
    fn test_json_string_in_struct() {
        #[derive(Debug, Serialize, Deserialize)]
        struct Outer {
            field: JsonString<TestSchema>,
        }

        let outer = Outer {
            field: JsonString::new(TestSchema {
                name: "Bob".to_string(),
                age: 25,
            }),
        };

        let serialized = serde_json::to_string(&outer).unwrap();
        assert!(serialized.contains(r#"\"name\":\"Bob\""#));

        let deserialized: Outer = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.field.inner().name, "Bob");
        assert_eq!(deserialized.field.inner().age, 25);
    }

    #[test]
    fn test_json_string_with_value() {
        use serde_json::json;

        let value = json!({
            "nested": {
                "key": "value",
                "array": [1, 2, 3]
            }
        });

        let wrapped = JsonString::new(value.clone());
        let serialized = serde_json::to_string(&wrapped).unwrap();
        let deserialized: JsonString<serde_json::Value> =
            serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.inner(), &value);
    }

    #[test]
    fn test_json_string_new_and_into_inner() {
        let original = "test data".to_string();
        let wrapped = JsonString::new(original.clone());
        let extracted = wrapped.into_inner();

        assert_eq!(extracted, original);
    }

    #[test]
    fn test_json_string_with_plain_string() {
        let original = "simple string".to_string();
        let wrapped = JsonString::new(original.clone());
        let serialized = serde_json::to_string(&wrapped).unwrap();

        // The serialized form should still be a JSON string
        assert_eq!(serialized, r#""simple string""#);

        let deserialized: JsonString<String> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.into_inner(), original);
    }
}
