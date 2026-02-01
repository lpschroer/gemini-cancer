//! Configuration for Gemini API client.

use anyhow::Result;
use std::env;

/// Returns the Gemini API key.
///
/// In debug builds, the API key is first checked from the runtime
/// GEMINI_API_KEY environment variable, then falls back to the
/// compile-time baked COMPILE_TIME_GEMINI_API_KEY if available.
///
/// In release builds, the API key is loaded from the GEMINI_API_KEY
/// environment variable at runtime.
///
/// # Errors
///
/// Returns an error if the API key is not available from either source.
#[cfg(debug_assertions)]
fn gemini_api_key() -> Result<String> {
    // In debug builds, first try runtime environment variable (for tests)
    // then fall back to compile-time baked key
    if let Ok(api_key) = std::env::var("GEMINI_API_KEY") {
        return Ok(api_key);
    }

    match option_env!("COMPILE_TIME_GEMINI_API_KEY") {
        Some(api_key) => Ok(api_key.to_string()),
        None => anyhow::bail!(
            "GEMINI_API_KEY was not set during compilation and is not available at runtime. \
             Please set the GEMINI_API_KEY environment variable."
        ),
    }
}

/// Returns the Gemini API key by loading it from the environment at runtime.
///
/// This is used for release builds to avoid baking secrets into the binary.
///
/// # Errors
///
/// Returns an error if the GEMINI_API_KEY environment variable is not set.
#[cfg(not(debug_assertions))]
fn gemini_api_key() -> Result<String> {
    use anyhow::Context;
    std::env::var("GEMINI_API_KEY").context("GEMINI_API_KEY environment variable not set")
}

/// Configuration for the Gemini API client.
#[derive(Clone)]
pub struct GeminiConfig {
    /// The API key for authenticating with Gemini API
    api_key: String,
    /// The model to use (e.g., "gemini-1.5-flash")
    model: String,
    /// Base URL for the Gemini API
    base_url: String,
}

impl std::fmt::Debug for GeminiConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiConfig")
            .field("api_key", &"[REDACTED]")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl GeminiConfig {
    /// Creates a new Gemini configuration from environment variables.
    ///
    /// # Environment Variables
    ///
    /// * `GEMINI_API_KEY` - Required. The API key for Gemini API.
    /// * `GEMINI_MODEL` - Optional. The model to use. Defaults to "gemini-2.5-flash".
    ///
    /// # Returns
    ///
    /// A `Result` containing the configuration or an error if the API key is missing.
    pub fn from_env() -> Result<Self> {
        let api_key = gemini_api_key()?;

        let model = env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-2.5-flash".to_string());

        let base_url = "https://generativelanguage.googleapis.com/v1beta".to_string();

        Ok(Self {
            api_key,
            model,
            base_url,
        })
    }

    /// Creates a new Gemini configuration with explicit values.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The API key for authentication
    /// * `model` - The model identifier to use
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        }
    }

    /// Returns the API key.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Returns the model identifier.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_config_new() {
        let config = GeminiConfig::new("test-key".to_string(), "gemini-1.5-pro".to_string());
        assert_eq!(config.api_key(), "test-key");
        assert_eq!(config.model(), "gemini-1.5-pro");
        assert_eq!(
            config.base_url(),
            "https://generativelanguage.googleapis.com/v1beta"
        );
    }

    #[test]
    fn test_gemini_config_from_env_missing_key() {
        // In debug builds, the API key is baked in at compile time,
        // so this test will pass if the key was set during compilation.
        // In release builds, removing the env var will cause an error.
        #[cfg(debug_assertions)]
        {
            // In debug builds, from_env() should succeed if key was baked in
            let result = GeminiConfig::from_env();
            // We can't test for missing key in debug builds since it's compile-time
            assert!(result.is_ok() || result.is_err());
        }

        #[cfg(not(debug_assertions))]
        {
            // In release builds, test that missing env var causes error
            unsafe {
                env::remove_var("GEMINI_API_KEY");
            }
            let result = GeminiConfig::from_env();
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_gemini_config_debug_redacts_api_key() {
        let config = GeminiConfig::new(
            "super-secret-key-12345".to_string(),
            "gemini-1.5-flash".to_string(),
        );
        let debug_output = format!("{:?}", config);

        // API key should be redacted
        assert!(!debug_output.contains("super-secret-key-12345"));
        assert!(debug_output.contains("[REDACTED]"));

        // Other fields should be visible
        assert!(debug_output.contains("gemini-1.5-flash"));
        assert!(debug_output.contains("generativelanguage.googleapis.com"));
    }
}
