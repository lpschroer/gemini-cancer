use std::error::Error;

use deser_incomplete::from_json_str;

pub use super::dto_content::{
    Blob, CodeExecutionResult, Content, ExecutableCode, FileData, FunctionCall, FunctionResponse,
    JsonString, Part, VideoMetadata,
};
pub use super::dto_request::{
    GenerateContentRequest, GenerationConfig, GenerationConfigBuilder, MimeType, ResponseMimeType,
    SafetyRating, SafetySetting,
};
pub use super::dto_response::{Candidate, GenerateContentResponse, PromptFeedback, UsageMetadata};
pub use super::stream_ext::BoxResponseStream;

/// Trait for Gemini content generation API
#[async_trait::async_trait]
pub trait GeminiApi {
    /// Generates content from the model given an input request
    ///
    /// # Arguments
    /// * `request` - The content generation request with type parameter `T`
    ///
    /// # Returns
    /// The model's response containing candidate text completions typed as `T`
    async fn generate_content<T>(
        &self,
        request: GenerateContentRequest<T>,
    ) -> Result<GenerateContentResponse<T>, Box<dyn Error>>
    where
        T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static;
}

#[async_trait::async_trait]
pub trait GeminiStreamingApi {
    /// Generates a streamed response from the model given an input request
    ///
    /// # Arguments
    /// * `request` - The content generation request with type parameter `T`
    ///
    /// # Returns
    /// An iterator of response chunks as they are generated, typed as `T`
    async fn stream_generate_content<T>(
        &self,
        request: GenerateContentRequest<T>,
    ) -> Result<BoxResponseStream<T>, Box<dyn Error>>
    where
        T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static;

    fn parse_incomplete<T>(
        raw: String,
    ) -> Result<GenerateContentResponse<T>, deser_incomplete::Error<serde_json::Error>>
    where
        T: serde::de::DeserializeOwned + 'static,
    {
        from_json_str::<GenerateContentResponse<T>>(&raw)
    }
}
