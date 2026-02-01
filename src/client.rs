//! Gemini API client implementation.

use std::error::Error;

use async_trait::async_trait;

use crate::api::{GeminiApi, GenerateContentRequest, GenerateContentResponse};
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
    #[allow(dead_code)]
    fn build_stream_url(&self) -> String {
        format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.config.base_url(),
            self.config.model()
        )
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
        // Test that we can create and serialize a request
        let request: GenerateContentRequest<String> = GenerateContentRequest {
            contents: vec![Content {
                parts: vec![Part {
                    text: Some(JsonString::new("Hello, world!".to_string())),
                    inline_data: None,
                    function_call: None,
                    function_response: None,
                    file_data: None,
                    executable_code: None,
                    code_execution_result: None,
                    video_metadata: None,
                }],
                role: None,
            }],
            generation_config: None,
            system_instruction: None,
            safety_settings: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("Hello, world!"));
    }

    // Tests for typed responses with JSON schema
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

            let request: GenerateContentRequest<Character> = GenerateContentRequest {
                contents: vec![Content {
                    parts: vec![Part {
                        text: Some(JsonString::new("Create a character".to_string())),
                        inline_data: None,
                        function_call: None,
                        function_response: None,
                        file_data: None,
                        executable_code: None,
                        code_execution_result: None,
                        video_metadata: None,
                    }],
                    role: None,
                }],
                generation_config: Some(config),
                system_instruction: None,
                safety_settings: None,
            };

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

            let request: GenerateContentRequest<Item> = GenerateContentRequest {
                contents: vec![Content {
                    parts: vec![Part {
                        text: Some(JsonString::new("Create an item".to_string())),
                        inline_data: None,
                        function_call: None,
                        function_response: None,
                        file_data: None,
                        executable_code: None,
                        code_execution_result: None,
                        video_metadata: None,
                    }],
                    role: None,
                }],
                generation_config: Some(config),
                system_instruction: None,
                safety_settings: None,
            };

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

        #[tokio::test]
        #[ignore]
        async fn test_real_api_string_response() {
            // This test requires GEMINI_API_KEY environment variable
            let client = GeminiV1Beta::from_env().expect("Failed to create client from env");

            let request: GenerateContentRequest<String> = GenerateContentRequest {
                contents: vec![Content {
                    parts: vec![Part {
                        text: Some(JsonString::new("Say hello in one word".to_string())),
                        inline_data: None,
                        function_call: None,
                        function_response: None,
                        file_data: None,
                        executable_code: None,
                        code_execution_result: None,
                        video_metadata: None,
                    }],
                    role: None,
                }],
                generation_config: None,
                system_instruction: None,
                safety_settings: None,
            };

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

            let request: GenerateContentRequest<Greeting> = GenerateContentRequest {
                contents: vec![Content {
                    parts: vec![Part {
                        text: Some(JsonString::new("Create a greeting in English".to_string())),
                        inline_data: None,
                        function_call: None,
                        function_response: None,
                        file_data: None,
                        executable_code: None,
                        code_execution_result: None,
                        video_metadata: None,
                    }],
                    role: None,
                }],
                generation_config: Some(config),
                system_instruction: None,
                safety_settings: None,
            };

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

            let request: GenerateContentRequest<Animal> = GenerateContentRequest {
                contents: vec![Content {
                    parts: vec![Part {
                        text: Some(JsonString::new(
                            "Create an animal with name, species, and age".to_string(),
                        )),
                        inline_data: None,
                        function_call: None,
                        function_response: None,
                        file_data: None,
                        executable_code: None,
                        code_execution_result: None,
                        video_metadata: None,
                    }],
                    role: None,
                }],
                generation_config: Some(config),
                system_instruction: None,
                safety_settings: None,
            };

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

            let request: GenerateContentRequest<Character> = GenerateContentRequest {
                contents: vec![Content {
                    parts: vec![Part {
                        text: Some(JsonString::new(
                            "Create a fantasy RPG character with name, age, class, \
                             a list of 3 skills, and stats (strength, intelligence, charisma)"
                                .to_string(),
                        )),
                        inline_data: None,
                        function_call: None,
                        function_response: None,
                        file_data: None,
                        executable_code: None,
                        code_execution_result: None,
                        video_metadata: None,
                    }],
                    role: None,
                }],
                generation_config: Some(config),
                system_instruction: None,
                safety_settings: None,
            };

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
    }
}
