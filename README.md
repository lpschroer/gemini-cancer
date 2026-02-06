> [!CAUTION]
> This project is the result of late night tinkering. 
> I hope it fuels your own nocturnal coding, but consume with caution!

# Gemini API Client

A type-safe Rust client library for the Google Gemini API.

## Features

- **Type-safe DTOs**: Fully typed request and response data structures
- **Automatic JSON Deserialization**: Type-safe parsing of structured responses
- **Dual Schema Support**: Both OpenAPI schema subset and full JSON Schema
- **Flexible Configuration**: Environment variables or explicit configuration
- **Streaming Support**: Both streaming and non-streaming content generation
- **Safety Settings**: Configurable content filtering and safety thresholds
- **Builder Pattern**: Ergonomic API for constructing requests

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
gemini = { path = "../gemini" }
```

## Quick Start

### Basic Configuration

```rust
use gemini::GeminiConfig;

// From environment variables (GEMINI_API_KEY, GEMINI_MODEL)
let config = GeminiConfig::from_env()?;

// Or with explicit values
let config = GeminiConfig::new(
    "your-api-key".to_string(),
    "gemini-2.5-flash".to_string()
);
```

### Simple Text Generation

```rust
use gemini::{GenerateContentRequest, Content, Part, Role};

let request = GenerateContentRequest {
    contents: vec![Content {
        role: Role::User,
        parts: vec![Part {
            text: Some("Write a haiku about Rust programming".to_string()),
            ..Default::default()
        }],
    }],
    generation_config: None,
    system_instruction: None,
    safety_settings: None,
};
```

### Structured Output with Type-Safe Schema Configuration

The Gemini API client provides **type-safe schema configuration** that enforces compile-time type safety between your schema definition and response parsing:

```rust
use gemini::{GenerateContentRequest, GenerateContentResponse, GenerationConfig, Content, Part, Role};
use serde::{Deserialize, Serialize};

// Define your response structure with JsonSchema derive
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct Character {
    character_name: String,
    character_class: String,
    level: u32,
}

// JSON schema is automatically derived from the type!
// Use turbofish syntax to specify the response type
let config: GenerationConfig<Character> = GenerationConfig::builder()
    .response_json_schema::<Character>()
    .temperature(0.7)
    .build()
    .unwrap();

// Request automatically inherits the type from config
let request: GenerateContentRequest<Character> = GenerateContentRequest {
    contents: vec![Content {
        role: Some(Role::User),
        parts: vec![Part {
            text: Some(JsonString::new("Create a fantasy RPG character".to_string())),
            inline_data: None,
            function_call: None,
            function_response: None,
            file_data: None,
            executable_code: None,
            code_execution_result: None,
            video_metadata: None,
        }],
    }],
    generation_config: Some(config),
    system_instruction: None,
    safety_settings: None,
};

// Response type is automatically inferred from the request!
let response = api.generate_content(request).await?;
// response is GenerateContentResponse<Character>

// Access the typed data directly - no manual parsing needed
if let Some(character) = response.first_text() {
    println!("Character: {} (Level {} {})", 
             character.character_name, 
             character.level, 
             character.character_class);
}
```

**Type Safety Benefits:**
- **Compile-time checking**: Type mismatch between schema and response is caught at compile time
- **Type inference**: Response type automatically flows from config to request to response
- **No manual parsing**: JSON deserialization happens automatically
- **Zero runtime cost**: Uses `PhantomData` - compiles to nothing

**Key Points:**
- Use turbofish syntax to specify response type: `.response_json_schema::<YourType>(schema)`
- The type flows: `GenerationConfig<T>` → `GenerateContentRequest<T>` → `GenerateContentResponse<T>`
- Use helper methods like `first_text()` to access the parsed data
- For plain text responses, use the default: `GenerationConfig` (same as `GenerationConfig<String>`)

### Advanced Configuration

```rust
use gemini::{GenerationConfig, ResponseMimeType};

let config = GenerationConfig::builder()
    .temperature(0.9)
    .max_output_tokens(2048)
    .top_p(0.95)
    .top_k(40)
    .add_stop_sequence("END")
    .build()
    .unwrap();
```

## Schema Support

The Gemini API supports two types of schemas for structured output:

### OpenAPI Schema (Subset)

- **Automatically derived from Rust types** that implement `ToSchema`
- Supports: objects, primitives, arrays
- Uses OpenAPI 3.0 schema format

### JSON Schema (Full)

- Full JSON Schema specification
- **Automatically derived from Rust types** that implement `JsonSchema`
- Supports advanced features: `$ref`, `$defs`, `anyOf`, `oneOf`, etc.
- More powerful and flexible

**Important**: These are mutually exclusive. Only one can be set per request.

```rust
// Using OpenAPI schema (auto-derived from type)
#[derive(serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
struct Person {
    name: String,
}

let config = GenerationConfig::<Person>::builder()
    .response_schema::<Person>()
    .build()
    .unwrap();

// Using JSON Schema (auto-derived from type)
#[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct Character {
    name: String,
    level: u32,
}

let config = GenerationConfig::<Character>::builder()
    .response_json_schema::<Character>()
    .build()
    .unwrap();
```

### Type Safety and Validation

The `build()` method validates that typed responses have appropriate schemas:
- **String responses**: No schema required (plain text output)
- **Typed responses**: Must provide either `response_schema` or `response_json_schema`

```rust
// String type - no schema required ✓
let config = GenerationConfig::<String>::builder()
    .temperature(0.7)
    .build()
    .unwrap();

// Typed response without schema - will fail ✗
#[derive(Deserialize, Serialize)]
struct MyType { value: String }

let result = GenerationConfig::<MyType>::builder()
    .temperature(0.7)
    .build();
assert!(result.is_err()); // BuildError::SchemaRequiredForTypedResponse

// Typed response with schema - succeeds ✓
// Option 1: Auto-derived OpenAPI schema
#[derive(Deserialize, Serialize, utoipa::ToSchema)]
struct MyType { value: String }

let config = GenerationConfig::<MyType>::builder()
    .response_schema::<MyType>()
    .build()
    .unwrap();

// Option 2: Auto-derived JSON schema
#[derive(Deserialize, Serialize, schemars::JsonSchema)]
struct MyType2 { value: String }

let config = GenerationConfig::<MyType2>::builder()
    .response_json_schema::<MyType2>()
    .build()
    .unwrap();
```

## Streaming API

The client supports streaming responses for real-time content generation:

```rust
use gemini::{GeminiStreamingApi, GenerateContentRequest, Content, Part, JsonString};
use futures::StreamExt;

let client = GeminiV1Beta::from_env()?;

let request: GenerateContentRequest<String> = GenerateContentRequest::builder()
    .add_content(Content {
        parts: vec![
            Part::builder()
                .text(JsonString::new("Write a short story".to_string()))
                .build(),
        ],
        role: None,
    })
    .build();

// Get a streaming async stream
let mut stream = client.stream_generate_content(request).await?;

// Process chunks as they arrive in real-time
while let Some(result) = stream.next().await {
    match result {
        Ok(response) => {
            if let Some(text) = response.first_text() {
                print!("{}", text);
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

### Streaming with Typed Responses

Streaming also works with typed responses, though intermediate chunks may contain incomplete JSON:

```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use futures::StreamExt;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Story {
    title: String,
    genre: String,
    summary: String,
}

let config = GenerationConfigBuilder::<String>::new()
    .response_json_schema::<Story>()
    .build()
    .unwrap();

let request: GenerateContentRequest<Story> = GenerateContentRequest::builder()
    .add_content(Content {
        parts: vec![
            Part::builder()
                .text(JsonString::new("Create a story idea".to_string()))
                .build(),
        ],
        role: None,
    })
    .generation_config(config)
    .build();

let mut stream = client.stream_generate_content(request).await?;

let mut last_complete_story: Option<Story> = None;
while let Some(result) = stream.next().await {
    match result {
        Ok(response) => {
            if let Some(story) = response.first_text() {
                last_complete_story = Some(story.clone());
                println!("Received: {:?}", story);
            }
        }
        Err(e) => {
            // Intermediate chunks may have incomplete JSON
            println!("Partial chunk (expected): {}", e);
        }
    }
}

// The final chunk should have the complete typed object
if let Some(story) = last_complete_story {
    println!("Final story: {}", story.title);
}
```

### Parse Incomplete JSON

The streaming API includes a `parse_incomplete` method that uses the `deser-incomplete` crate to handle partial JSON:

```rust
use gemini::GeminiStreamingApi;

// This is handled automatically in the stream, but you can use it directly:
let raw_json = r#"{"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}"#;
let response = GeminiV1Beta::parse_incomplete::<String>(raw_json.to_string())?;
```

## API Traits

The crate provides two traits for implementing custom clients:

### GeminiApi

For non-streaming content generation:

```rust
use gemini::{GeminiApi, GenerateContentRequest, GenerateContentResponse};
use std::error::Error;

#[async_trait::async_trait]
pub trait GeminiApi {
    async fn generate_content<T>(
        &self,
        request: GenerateContentRequest<T>,
    ) -> Result<GenerateContentResponse<T>, Box<dyn Error>>
    where
        T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static;
}
```

### GeminiStreamingApi

For streaming content generation (returns async Stream):

```rust
use gemini::{GeminiStreamingApi, GenerateContentRequest, StreamingResponseStream};
use std::error::Error;

#[async_trait::async_trait]
pub trait GeminiStreamingApi {
    async fn stream_generate_content<T>(
        &self,
        request: GenerateContentRequest<T>,
    ) -> Result<StreamingResponseStream<T>, Box<dyn Error>>
    where
        T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static;

    fn parse_incomplete<T>(
        raw: String,
    ) -> Result<GenerateContentResponse<T>, deser_incomplete::Error<serde_json::Error>>
    where
        T: serde::de::DeserializeOwned + 'static;
}
```

**Note**: `StreamingResponseStream<T>` is a pinned, boxed async Stream that yields results as they arrive from the API in real-time. Use with `futures::StreamExt` to consume chunks with `.next().await`.

## Environment Variables

- `GEMINI_API_KEY` (required): Your Gemini API key
- `GEMINI_MODEL` (optional): Model to use (defaults to "gemini-2.5-flash")

### Debug vs Release Builds

- **Debug builds**: API key is baked in at compile time from `COMPILE_TIME_GEMINI_API_KEY`
- **Release builds**: API key is loaded from environment at runtime

## Safety and Content Filtering

Configure safety settings to control content filtering:

```rust
use gemini::{SafetySetting, GenerateContentRequest};

let request = GenerateContentRequest {
    contents: vec![/* ... */],
    generation_config: None,
    system_instruction: None,
    safety_settings: Some(vec![
        SafetySetting {
            category: "HARM_CATEGORY_HATE_SPEECH".to_string(),
            threshold: "BLOCK_MEDIUM_AND_ABOVE".to_string(),
        }
    ]),
};
```

## Testing

### Unit Tests

Run the standard test suite:

```bash
cargo test --package gemini
```

Run tests with all features enabled:

```bash
cargo test --package gemini --all-features
```

### Integration Tests

The crate includes integration tests that make real API calls to the Gemini API. These tests are **ignored by default** and require a valid `GEMINI_API_KEY` environment variable.

#### Running Integration Tests

```bash
# Set your API key
export GEMINI_API_KEY="your-api-key-here"

# Run all tests including ignored integration tests
cargo test --package gemini --all-features -- --ignored

# Run only integration tests
cargo test --package gemini --all-features integration_tests -- --ignored

# Run a specific integration test
cargo test --package gemini --all-features test_real_api_string_response -- --ignored
```

#### Available Integration Tests

**Non-Streaming Tests:**
- **`test_real_api_string_response`**: Tests basic string response from the API
- **`test_real_api_typed_json_response`** (requires `json` feature): Tests typed response with JSON schema
- **`test_real_api_typed_openapi_response`** (requires `openapi` feature): Tests typed response with OpenAPI schema
- **`test_real_api_complex_typed_response`** (requires `json` feature): Tests complex nested typed response

**Streaming Tests:**
- **`test_real_api_string_streaming`**: Tests streaming string responses from the API
- **`test_real_api_typed_streaming`** (requires `json` feature): Tests streaming with typed JSON responses

#### Feature-Specific Tests

To run tests for specific features:

```bash
# Test JSON schema support
cargo test --package gemini --features json typed_json_tests

# Test OpenAPI schema support
cargo test --package gemini --features openapi typed_openapi_tests

# Test both
cargo test --package gemini --all-features
```

### Test Categories

The test suite includes:

1. **Unit tests**: Test individual components without API calls (run by default)
2. **Typed schema tests**: Test schema generation and serialization for both JSON and OpenAPI schemas
3. **Integration tests**: Make real API calls to verify end-to-end functionality (ignored by default, require `GEMINI_API_KEY`)

## License

MIT