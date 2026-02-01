//! Request DTOs for the Gemini API
//!
//! ## Schema Support
//!
//! The Gemini API supports two types of schemas for structured output:
//!
//! ### `response_schema` (OpenAPI Schema)
//! - Uses a subset of the OpenAPI schema format
//! - Supports: objects, primitives, arrays
//! - More limited feature set
//!
//! ### `response_json_schema` (JSON Schema)
//! - Uses the full JSON Schema specification
//! - Automatically derived from Rust types that implement `JsonSchema`
//! - Supports advanced features like: `$ref`, `$defs`, `anyOf`, `oneOf`, etc.
//! - More flexible and powerful
//!
//! **Important**: These two fields are **mutually exclusive**. Only one should be set.
//! The builder methods automatically enforce this by clearing one when the other is set.
//!
//! ## Type Safety and Validation
//!
//! The `build()` method validates that typed responses (non-`String` types) have a schema configured:
//! - **String responses**: No schema required (plain text output)
//! - **Typed responses**: Must provide either `response_schema` or `response_json_schema`
//! - Returns `Result<GenerationConfig<T>, BuildError>` for safety
//!
//! ### Example Usage
//!
//! ```rust,ignore
//! // Using OpenAPI schema (auto-derived from type)
//! #[derive(serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
//! struct Person {
//!     name: String,
//! }
//!
//! let config = GenerationConfig::<Person>::builder()
//!     .response_schema::<Person>()
//!     .build()
//!     .unwrap();
//!
//! // Using JSON Schema (auto-derived from type)
//! #[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
//! struct Character {
//!     name: String,
//!     level: u32,
//! }
//!
//! let config = GenerationConfig::<Character>::builder()
//!     .response_json_schema::<Character>()
//!     .build()
//!     .unwrap();
//! ```

use serde::{Deserialize, Serialize};
use std::any::TypeId;
use std::marker::PhantomData;

pub use crate::dto_content::{
    Blob, CodeExecutionResult, Content, ExecutableCode, FileData, FunctionCall, FunctionResponse,
    Part, Role, VideoMetadata,
};

/// Error type for GenerationConfig builder validation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// A schema must be provided when using typed responses (non-String types)
    SchemaRequiredForTypedResponse,
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::SchemaRequiredForTypedResponse => {
                write!(
                    f,
                    "A response schema must be provided when using typed responses. \
                     Use .response_schema() or .response_json_schema() to specify the expected structure."
                )
            }
        }
    }
}

impl std::error::Error for BuildError {}

/// Supported MIME types for Gemini API responses
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ResponseMimeType {
    /// Plain text output (default)
    #[default]
    #[serde(rename = "text/plain")]
    TextPlain,
    /// JSON response
    #[serde(rename = "application/json")]
    ApplicationJson,
    /// ENUM as a string response
    #[serde(rename = "text/x.enum")]
    TextEnum,
}

/// MIME type for media content supported by Gemini API
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MimeType {
    // Image formats
    #[serde(rename = "image/png")]
    ImagePng,
    #[serde(rename = "image/jpeg")]
    ImageJpeg,
    #[serde(rename = "image/webp")]
    ImageWebp,
    #[serde(rename = "image/heic")]
    ImageHeic,
    #[serde(rename = "image/heif")]
    ImageHeif,

    // Audio formats
    #[serde(rename = "audio/wav")]
    AudioWav,
    #[serde(rename = "audio/mp3")]
    AudioMp3,
    #[serde(rename = "audio/mpeg")]
    AudioMpeg,
    #[serde(rename = "audio/aiff")]
    AudioAiff,
    #[serde(rename = "audio/aac")]
    AudioAac,
    #[serde(rename = "audio/ogg")]
    AudioOgg,
    #[serde(rename = "audio/flac")]
    AudioFlac,

    // Video formats
    #[serde(rename = "video/mp4")]
    VideoMp4,
    #[serde(rename = "video/mpeg")]
    VideoMpeg,
    #[serde(rename = "video/mov")]
    VideoMov,
    #[serde(rename = "video/avi")]
    VideoAvi,
    #[serde(rename = "video/x-flv")]
    VideoFlv,
    #[serde(rename = "video/mpg")]
    VideoMpg,
    #[serde(rename = "video/webm")]
    VideoWebm,
    #[serde(rename = "video/wmv")]
    VideoWmv,
    #[serde(rename = "video/3gpp")]
    Video3gpp,

    // Document formats
    #[serde(rename = "application/pdf")]
    ApplicationPdf,
    #[serde(rename = "text/plain")]
    TextPlain,
    #[serde(rename = "text/html")]
    TextHtml,
    #[serde(rename = "text/css")]
    TextCss,
    #[serde(rename = "text/javascript")]
    TextJavascript,
    #[serde(rename = "application/x-javascript")]
    ApplicationJavascript,
    #[serde(rename = "text/x-typescript")]
    TextTypescript,
    #[serde(rename = "application/x-typescript")]
    ApplicationTypescript,
    #[serde(rename = "text/csv")]
    TextCsv,
    #[serde(rename = "text/markdown")]
    TextMarkdown,
    #[serde(rename = "text/x-python")]
    TextPython,
    #[serde(rename = "application/x-python-code")]
    ApplicationPythonCode,
    #[serde(rename = "application/json")]
    ApplicationJson,
    #[serde(rename = "text/xml")]
    TextXml,
    #[serde(rename = "application/rtf")]
    ApplicationRtf,
    #[serde(rename = "text/rtf")]
    TextRtf,
}

/// Request body for generateContent API call
///
/// Generic over the expected response type `T`, which defaults to `String`.
#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateContentRequest<T = String> {
    /// The content of the current conversation with the model
    pub contents: Vec<Content>,

    /// Optional configuration options for model generation and outputs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig<T>>,

    /// Optional system instruction for the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,

    /// Optional safety settings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_settings: Option<Vec<SafetySetting>>,
}

/// Configuration options for model generation
///
/// Generic over the expected response type `T`, which defaults to `String`.
/// Use turbofish syntax like `.response_schema::<MyType>(schema)` for type safety.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig<T = String> {
    /// Phantom data to track the expected response type at compile time
    #[serde(skip)]
    _response_type: PhantomData<T>,

    /// The set of character sequences (up to 5) that will stop output generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    /// MIME type of the generated candidate text.
    /// Defaults to TextPlain when not set.
    /// Must be set to ApplicationJson when response_schema or response_json_schema is provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<ResponseMimeType>,

    /// Output schema of the generated candidate text (OpenAPI schema subset)
    /// Note: Mutually exclusive with response_json_schema. Only one should be set.
    /// Use the builder methods to ensure proper mutual exclusivity.
    #[cfg(feature = "openapi")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<serde_json::Value>,

    /// Output schema of the generated response (full JSON Schema support)
    /// This is an alternative to response_schema that accepts JSON Schema format.
    /// Note: Mutually exclusive with response_schema. Only one should be set.
    /// Use the builder methods to ensure proper mutual exclusivity.
    #[cfg(feature = "json")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_json_schema: Option<serde_json::Value>,

    /// The requested modalities of the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_modalities: Option<Vec<String>>,

    /// Number of generated responses to return
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<i32>,

    /// Maximum number of tokens to include in a response candidate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,

    /// Controls the randomness of the output (0.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Maximum cumulative probability of tokens to consider when sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Maximum number of tokens to consider when sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,

    /// Seed used in decoding
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,

    /// Presence penalty applied to the next token's logprobs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Frequency penalty applied to the next token's logprobs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// If true, export the logprobs results in response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_logprobs: Option<bool>,

    /// Number of top logprobs to return at each decoding step (0-20)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<i32>,

    /// Enables enhanced civic answers
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_enhanced_civic_answers: Option<bool>,

    /// Speech generation config
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speech_config: Option<serde_json::Value>,

    /// Config for thinking features
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<serde_json::Value>,

    /// Config for image generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_config: Option<serde_json::Value>,

    /// Media resolution for input media
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_resolution: Option<String>,
}

impl<T> GenerationConfig<T> {
    /// Creates a builder for GenerationConfig
    pub fn builder() -> GenerationConfigBuilder<T> {
        GenerationConfigBuilder::default()
    }
}

impl<T> Default for GenerationConfig<T> {
    fn default() -> Self {
        Self {
            _response_type: PhantomData,
            stop_sequences: None,
            response_mime_type: None,
            #[cfg(feature = "openapi")]
            response_schema: None,
            #[cfg(feature = "json")]
            response_json_schema: None,
            response_modalities: None,
            candidate_count: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
            presence_penalty: None,
            frequency_penalty: None,
            response_logprobs: None,
            logprobs: None,
            enable_enhanced_civic_answers: None,
            speech_config: None,
            thinking_config: None,
            image_config: None,
            media_resolution: None,
        }
    }
}

/// Builder for GenerationConfig
///
/// Generic over the expected response type `T`, which defaults to `String`.
#[derive(Debug)]
pub struct GenerationConfigBuilder<T = String> {
    _response_type: PhantomData<T>,
    stop_sequences: Option<Vec<String>>,
    response_mime_type: Option<ResponseMimeType>,
    #[cfg(feature = "openapi")]
    response_schema: Option<serde_json::Value>,
    #[cfg(feature = "json")]
    response_json_schema: Option<serde_json::Value>,
    response_modalities: Option<Vec<String>>,
    candidate_count: Option<i32>,
    max_output_tokens: Option<i32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    seed: Option<i32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    response_logprobs: Option<bool>,
    logprobs: Option<i32>,
    enable_enhanced_civic_answers: Option<bool>,
    speech_config: Option<serde_json::Value>,
    thinking_config: Option<serde_json::Value>,
    image_config: Option<serde_json::Value>,
    media_resolution: Option<String>,
}

impl<T> Default for GenerationConfigBuilder<T> {
    fn default() -> Self {
        Self {
            _response_type: PhantomData,
            stop_sequences: None,
            response_mime_type: None,
            #[cfg(feature = "openapi")]
            response_schema: None,
            #[cfg(feature = "json")]
            response_json_schema: None,
            response_modalities: None,
            candidate_count: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
            presence_penalty: None,
            frequency_penalty: None,
            response_logprobs: None,
            logprobs: None,
            enable_enhanced_civic_answers: None,
            speech_config: None,
            thinking_config: None,
            image_config: None,
            media_resolution: None,
        }
    }
}

impl<T> GenerationConfigBuilder<T> {
    /// Creates a new builder with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the stop sequences (up to 5)
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }

    /// Adds a single stop sequence
    pub fn add_stop_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.stop_sequences
            .get_or_insert_with(Vec::new)
            .push(sequence.into());
        self
    }

    /// Sets the response schema (OpenAPI schema subset)
    /// Automatically derives the schema from the type parameter.
    /// Automatically sets response_mime_type to ApplicationJson
    /// Note: This is mutually exclusive with response_json_schema.
    ///
    /// Use turbofish to specify response type: `.response_schema::<MyType>()`
    #[cfg(feature = "openapi")]
    pub fn response_schema<R>(self) -> GenerationConfigBuilder<R>
    where
        R: serde::de::DeserializeOwned + serde::Serialize + 'static + utoipa::ToSchema,
    {
        let schema = R::schema();
        let schema_value =
            serde_json::to_value(schema).expect("Failed to serialize OpenAPI schema");

        GenerationConfigBuilder {
            _response_type: PhantomData,
            #[cfg(feature = "openapi")]
            response_schema: Some(schema_value),
            #[cfg(feature = "json")]
            response_json_schema: None,
            response_mime_type: Some(ResponseMimeType::ApplicationJson),
            stop_sequences: self.stop_sequences,
            response_modalities: self.response_modalities,
            candidate_count: self.candidate_count,
            max_output_tokens: self.max_output_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            seed: self.seed,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            response_logprobs: self.response_logprobs,
            logprobs: self.logprobs,
            enable_enhanced_civic_answers: self.enable_enhanced_civic_answers,
            speech_config: self.speech_config,
            thinking_config: self.thinking_config,
            image_config: self.image_config,
            media_resolution: self.media_resolution,
        }
    }

    /// Sets the response JSON schema (full JSON Schema support)
    /// Automatically derives the schema from the type parameter.
    /// Automatically sets response_mime_type to ApplicationJson
    /// Note: This is mutually exclusive with response_schema.
    ///
    /// Use turbofish to specify response type: `.response_json_schema::<MyType>()`
    #[cfg(feature = "json")]
    pub fn response_json_schema<R>(self) -> GenerationConfigBuilder<R>
    where
        R: serde::de::DeserializeOwned + serde::Serialize + 'static + schemars::JsonSchema,
    {
        let schema = schemars::schema_for!(R);
        let schema_value = serde_json::to_value(schema).expect("Failed to serialize JSON schema");

        GenerationConfigBuilder {
            _response_type: PhantomData,
            #[cfg(feature = "json")]
            response_json_schema: Some(schema_value),
            #[cfg(feature = "openapi")]
            response_schema: None,
            response_mime_type: Some(ResponseMimeType::ApplicationJson),
            stop_sequences: self.stop_sequences,
            response_modalities: self.response_modalities,
            candidate_count: self.candidate_count,
            max_output_tokens: self.max_output_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            seed: self.seed,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            response_logprobs: self.response_logprobs,
            logprobs: self.logprobs,
            enable_enhanced_civic_answers: self.enable_enhanced_civic_answers,
            speech_config: self.speech_config,
            thinking_config: self.thinking_config,
            image_config: self.image_config,
            media_resolution: self.media_resolution,
        }
    }

    /// Sets the response modalities
    pub fn response_modalities(mut self, modalities: Vec<String>) -> Self {
        self.response_modalities = Some(modalities);
        self
    }

    /// Adds a single response modality
    pub fn add_response_modality(mut self, modality: impl Into<String>) -> Self {
        self.response_modalities
            .get_or_insert_with(Vec::new)
            .push(modality.into());
        self
    }

    /// Sets the number of generated responses to return
    pub fn candidate_count(mut self, count: i32) -> Self {
        self.candidate_count = Some(count);
        self
    }

    /// Sets the maximum number of output tokens
    pub fn max_output_tokens(mut self, tokens: i32) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    /// Sets the temperature (0.0 to 2.0)
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Sets the top-p sampling parameter
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Sets the top-k sampling parameter
    pub fn top_k(mut self, k: i32) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Sets the random seed for decoding
    pub fn seed(mut self, seed: i32) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the presence penalty
    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    /// Sets the frequency penalty
    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    /// Enables response logprobs export
    pub fn response_logprobs(mut self, enabled: bool) -> Self {
        self.response_logprobs = Some(enabled);
        self
    }

    /// Sets the number of top logprobs to return (0-20)
    pub fn logprobs(mut self, count: i32) -> Self {
        self.logprobs = Some(count);
        self
    }

    /// Enables enhanced civic answers
    pub fn enable_enhanced_civic_answers(mut self, enabled: bool) -> Self {
        self.enable_enhanced_civic_answers = Some(enabled);
        self
    }

    /// Sets the speech generation config
    pub fn speech_config(mut self, config: serde_json::Value) -> Self {
        self.speech_config = Some(config);
        self
    }

    /// Sets the thinking config
    pub fn thinking_config(mut self, config: serde_json::Value) -> Self {
        self.thinking_config = Some(config);
        self
    }

    /// Sets the image generation config
    pub fn image_config(mut self, config: serde_json::Value) -> Self {
        self.image_config = Some(config);
        self
    }

    /// Sets the media resolution
    pub fn media_resolution(mut self, resolution: impl Into<String>) -> Self {
        self.media_resolution = Some(resolution.into());
        self
    }

    /// Configures for plain text response
    pub fn text_response(mut self) -> Self {
        self.response_mime_type = Some(ResponseMimeType::TextPlain);
        self
    }

    /// Configures for enum response
    pub fn enum_response(mut self) -> Self {
        self.response_mime_type = Some(ResponseMimeType::TextEnum);
        self
    }

    /// Builds the GenerationConfig
    ///
    /// # Type Safety
    ///
    /// This method validates that typed responses have appropriate schemas configured:
    /// - For `String` type: No schema required (returns plain text)
    /// - For custom types: Requires either `response_schema` or `response_json_schema`
    ///
    /// # Errors
    ///
    /// Returns `BuildError::SchemaRequiredForTypedResponse` if:
    /// - `T` is not `String` (i.e., using typed responses)
    /// - AND no schema is provided (neither `response_schema` nor `response_json_schema`)
    ///
    /// # Examples
    ///
    /// ```
    /// # use gemini::dto_request::{GenerationConfig, BuildError};
    /// // String type - no schema required
    /// let config = GenerationConfig::<String>::builder()
    ///     .temperature(0.7)
    ///     .build();
    /// assert!(config.is_ok());
    /// ```
    ///
    /// ```should_panic
    /// # use gemini::dto_request::GenerationConfig;
    /// # use serde::{Deserialize, Serialize};
    /// # #[derive(Deserialize, Serialize)]
    /// # struct MyType { value: String }
    /// // Typed response without schema - will fail
    /// let config = GenerationConfig::<MyType>::builder()
    ///     .temperature(0.7)
    ///     .build()
    ///     .unwrap(); // This will panic!
    /// ```
    pub fn build(self) -> Result<GenerationConfig<T>, BuildError>
    where
        T: 'static,
    {
        // Validate that typed responses have a schema configured
        let is_string_type = TypeId::of::<T>() == TypeId::of::<String>();

        if !is_string_type {
            // Check if at least one schema is set (considering feature flags)
            let has_schema = {
                #[cfg(all(feature = "openapi", feature = "json"))]
                {
                    self.response_schema.is_some() || self.response_json_schema.is_some()
                }

                #[cfg(all(feature = "openapi", not(feature = "json")))]
                {
                    self.response_schema.is_some()
                }

                #[cfg(all(not(feature = "openapi"), feature = "json"))]
                {
                    self.response_json_schema.is_some()
                }

                #[cfg(not(any(feature = "openapi", feature = "json")))]
                {
                    false
                }
            };

            if !has_schema {
                return Err(BuildError::SchemaRequiredForTypedResponse);
            }
        }

        Ok(GenerationConfig {
            _response_type: PhantomData,
            stop_sequences: self.stop_sequences,
            response_mime_type: self.response_mime_type,
            #[cfg(feature = "openapi")]
            response_schema: self.response_schema,
            #[cfg(feature = "json")]
            response_json_schema: self.response_json_schema,
            response_modalities: self.response_modalities,
            candidate_count: self.candidate_count,
            max_output_tokens: self.max_output_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            seed: self.seed,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            response_logprobs: self.response_logprobs,
            logprobs: self.logprobs,
            enable_enhanced_civic_answers: self.enable_enhanced_civic_answers,
            speech_config: self.speech_config,
            thinking_config: self.thinking_config,
            image_config: self.image_config,
            media_resolution: self.media_resolution,
        })
    }
}

/// Safety setting for content filtering
#[derive(Debug, Serialize, Deserialize)]
pub struct SafetySetting {
    /// Safety category
    pub category: String,

    /// Harm block threshold
    pub threshold: String,
}

/// Safety rating for content
#[derive(Debug, Serialize, Deserialize)]
pub struct SafetyRating {
    /// Harm category
    pub category: String,

    /// Probability of harm
    pub probability: String,

    /// Whether content was blocked
    #[serde(default)]
    pub blocked: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_builder_basic() {
        let config: GenerationConfig = GenerationConfig::builder()
            .temperature(0.7)
            .max_output_tokens(1024)
            .build()
            .unwrap();

        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_output_tokens, Some(1024));
        assert_eq!(config.response_mime_type, None);
    }

    #[cfg(feature = "openapi")]
    #[test]
    fn test_generation_config_builder_json_response() {
        #[derive(serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
        struct TestType {
            name: String,
        }

        let config: GenerationConfig<TestType> = GenerationConfig::<TestType>::builder()
            .response_schema::<TestType>()
            .temperature(0.5)
            .build()
            .unwrap();

        assert_eq!(
            config.response_mime_type,
            Some(ResponseMimeType::ApplicationJson)
        );
        assert!(config.response_schema.is_some());
        assert_eq!(config.temperature, Some(0.5));
    }

    #[test]
    fn test_generation_config_builder_stop_sequences() {
        let config: GenerationConfig = GenerationConfig::builder()
            .add_stop_sequence("END")
            .add_stop_sequence("STOP")
            .build()
            .unwrap();

        assert_eq!(
            config.stop_sequences,
            Some(vec!["END".to_string(), "STOP".to_string()])
        );
    }

    #[test]
    fn test_generation_config_builder_full() {
        let config: GenerationConfig = GenerationConfig::builder()
            .temperature(0.9)
            .max_output_tokens(2048)
            .top_p(0.95)
            .top_k(40)
            .seed(12345)
            .presence_penalty(0.5)
            .frequency_penalty(0.3)
            .candidate_count(3)
            .response_logprobs(true)
            .logprobs(5)
            .add_stop_sequence("END")
            .build()
            .unwrap();

        assert_eq!(config.temperature, Some(0.9));
        assert_eq!(config.max_output_tokens, Some(2048));
        assert_eq!(config.top_p, Some(0.95));
        assert_eq!(config.top_k, Some(40));
        assert_eq!(config.seed, Some(12345));
        assert_eq!(config.presence_penalty, Some(0.5));
        assert_eq!(config.frequency_penalty, Some(0.3));
        assert_eq!(config.candidate_count, Some(3));
        assert_eq!(config.response_logprobs, Some(true));
        assert_eq!(config.logprobs, Some(5));
        assert_eq!(config.stop_sequences, Some(vec!["END".to_string()]));
        assert_eq!(config.response_mime_type, None);
    }

    #[test]
    fn test_generation_config_builder_modalities() {
        let config: GenerationConfig = GenerationConfig::builder()
            .add_response_modality("TEXT")
            .add_response_modality("IMAGE")
            .build()
            .unwrap();

        assert_eq!(
            config.response_modalities,
            Some(vec!["TEXT".to_string(), "IMAGE".to_string()])
        );
    }

    #[test]
    fn test_generation_config_default() {
        let config: GenerationConfig = GenerationConfig::default();

        assert_eq!(config.temperature, None);
        assert_eq!(config.max_output_tokens, None);
        assert_eq!(config.response_mime_type, None);
    }

    #[cfg(feature = "openapi")]
    #[test]
    fn test_generation_config_serialization() {
        #[derive(serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
        struct TestResult {
            result: String,
        }

        let config: GenerationConfig<TestResult> = GenerationConfig::<TestResult>::builder()
            .temperature(0.7)
            .max_output_tokens(1024)
            .response_schema::<TestResult>()
            .build()
            .unwrap();

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("temperature"));
        assert!(json.contains("maxOutputTokens")); // camelCase check
        assert!(json.contains("responseMimeType")); // camelCase check
    }

    #[test]
    fn test_response_mime_type_enum_serialization() {
        // Test TextPlain serialization
        let config: GenerationConfig = GenerationConfig::<String> {
            response_mime_type: Some(ResponseMimeType::TextPlain),
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"text/plain\""));

        // Test ApplicationJson serialization
        let config: GenerationConfig = GenerationConfig::<String> {
            response_mime_type: Some(ResponseMimeType::ApplicationJson),
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"application/json\""));

        // Test TextEnum serialization
        let config: GenerationConfig = GenerationConfig::builder().enum_response().build().unwrap();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"text/x.enum\""));
    }

    #[cfg(all(feature = "openapi", feature = "json"))]
    #[test]
    fn test_schema_auto_sets_mime_type() {
        // Test response_schema auto-sets mime type
        #[derive(serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
        struct TestTypeOpenApi {
            result: String,
        }

        let config: GenerationConfig<TestTypeOpenApi> =
            GenerationConfig::<TestTypeOpenApi>::builder()
                .response_schema::<TestTypeOpenApi>()
                .build()
                .unwrap();
        assert_eq!(
            config.response_mime_type,
            Some(ResponseMimeType::ApplicationJson)
        );

        // Test response_json_schema auto-sets mime type
        #[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
        struct TestTypeJson {
            result: String,
        }

        let config2: GenerationConfig<TestTypeJson> = GenerationConfig::<TestTypeJson>::builder()
            .response_json_schema::<TestTypeJson>()
            .build()
            .unwrap();

        assert_eq!(
            config2.response_mime_type,
            Some(ResponseMimeType::ApplicationJson)
        );
    }

    #[cfg(all(feature = "openapi", feature = "json"))]
    #[test]
    fn test_schema_mutual_exclusivity() {
        // Test that setting response_schema first, then response_json_schema clears response_schema
        #[derive(serde::Serialize, serde::Deserialize, utoipa::ToSchema, schemars::JsonSchema)]
        struct TestType1 {
            name: String,
        }

        #[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
        struct TestType2 {
            age: u32,
        }

        let config: GenerationConfig<TestType2> = GenerationConfig::<TestType1>::builder()
            .response_schema::<TestType1>()
            .response_json_schema::<TestType2>()
            .build()
            .unwrap();

        assert_eq!(config.response_schema, None);
        assert!(config.response_json_schema.is_some());
        assert_eq!(
            config.response_mime_type,
            Some(ResponseMimeType::ApplicationJson)
        );

        // Test that setting response_json_schema first, then response_schema clears response_json_schema
        #[derive(serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
        struct TestType3 {
            value: String,
        }

        let config2: GenerationConfig<TestType3> = GenerationConfig::<TestType2>::builder()
            .response_json_schema::<TestType2>()
            .response_schema::<TestType3>()
            .build()
            .unwrap();

        assert_eq!(config2.response_json_schema, None);
        assert!(config2.response_schema.is_some());
        assert_eq!(
            config2.response_mime_type,
            Some(ResponseMimeType::ApplicationJson)
        );
    }

    #[test]
    fn test_response_mime_type_default() {
        let default_mime = ResponseMimeType::default();
        assert_eq!(default_mime, ResponseMimeType::TextPlain);
    }

    #[cfg(feature = "json")]
    #[test]
    fn test_typed_response_config() {
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
        struct Character {
            name: String,
            level: u32,
        }

        // Use turbofish to specify response type - schema is auto-derived
        let config: GenerationConfig<Character> = GenerationConfig::<Character>::builder()
            .response_json_schema::<Character>()
            .temperature(0.7)
            .build()
            .unwrap();

        assert!(config.response_json_schema.is_some());
        assert_eq!(config.temperature, Some(0.7));

        // Use typed config in a request
        let _request: GenerateContentRequest<Character> = GenerateContentRequest {
            contents: vec![],
            generation_config: Some(config),
            system_instruction: None,
            safety_settings: None,
        };
    }

    #[test]
    fn test_build_validates_string_type_allows_no_schema() {
        // String type should not require a schema
        let result = GenerationConfig::<String>::builder()
            .temperature(0.7)
            .build();

        assert!(result.is_ok());
    }

    #[cfg(feature = "openapi")]
    #[test]
    fn test_build_validates_typed_response_requires_schema() {
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Deserialize, Serialize, utoipa::ToSchema)]
        struct TestStruct {
            value: String,
        }

        // Typed response without schema should fail
        let result = GenerationConfig::<TestStruct>::builder()
            .temperature(0.7)
            .build();

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            BuildError::SchemaRequiredForTypedResponse
        );

        // Typed response WITH schema should succeed (auto-derived)
        let result = GenerationConfig::<TestStruct>::builder()
            .response_schema::<TestStruct>()
            .temperature(0.7)
            .build();

        assert!(result.is_ok());
    }

    #[cfg(feature = "json")]
    #[test]
    fn test_build_validates_typed_response_requires_json_schema() {
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
        struct TestStruct {
            value: String,
        }

        // Typed response without schema should fail
        let result = GenerationConfig::<TestStruct>::builder()
            .temperature(0.7)
            .build();

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            BuildError::SchemaRequiredForTypedResponse
        );

        // Typed response WITH json_schema should succeed (auto-derived)
        let result = GenerationConfig::<TestStruct>::builder()
            .response_json_schema::<TestStruct>()
            .temperature(0.7)
            .build();

        assert!(result.is_ok());
    }

    #[cfg(not(any(feature = "openapi", feature = "json")))]
    #[test]
    fn test_build_validates_typed_response_fails_without_features() {
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Deserialize, Serialize)]
        struct TestStruct {
            value: String,
        }

        // Without schema features, typed responses should always fail
        let result = GenerationConfig::<TestStruct>::builder()
            .temperature(0.7)
            .build();

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            BuildError::SchemaRequiredForTypedResponse
        );
    }

    #[cfg(feature = "json")]
    #[test]
    fn test_auto_derived_json_schema_content() {
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
        struct TestPerson {
            name: String,
            age: u32,
        }

        let config: GenerationConfig<TestPerson> = GenerationConfig::<TestPerson>::builder()
            .response_json_schema::<TestPerson>()
            .build()
            .unwrap();

        // Verify schema was generated
        assert!(config.response_json_schema.is_some());

        let schema = config.response_json_schema.unwrap();

        // Verify schema structure
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["name"].is_object());
        assert!(schema["properties"]["age"].is_object());
        assert_eq!(schema["properties"]["name"]["type"], "string");
        assert_eq!(schema["properties"]["age"]["type"], "integer");
    }

    #[cfg(feature = "openapi")]
    #[test]
    fn test_auto_derived_openapi_schema_content() {
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Deserialize, Serialize, utoipa::ToSchema)]
        struct TestPerson {
            name: String,
            age: u32,
        }

        let config: GenerationConfig<TestPerson> = GenerationConfig::<TestPerson>::builder()
            .response_schema::<TestPerson>()
            .build()
            .unwrap();

        // Verify schema was generated
        assert!(config.response_schema.is_some());

        let schema = config.response_schema.unwrap();

        // Verify schema structure (OpenAPI format)
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["name"].is_object());
        assert!(schema["properties"]["age"].is_object());
        assert_eq!(schema["properties"]["name"]["type"], "string");
        assert_eq!(schema["properties"]["age"]["type"], "integer");
    }

    #[test]
    fn test_build_error_display() {
        let error = BuildError::SchemaRequiredForTypedResponse;
        let error_msg = error.to_string();
        assert!(error_msg.contains("response schema must be provided"));
        assert!(error_msg.contains("typed responses"));
    }
}
