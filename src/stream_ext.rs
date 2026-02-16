//! Stream extensions and utilities for Gemini API responses.
//!
//! This module provides extension traits and utilities for working with
//! streams returned by the Gemini API, particularly for transforming
//! response streams into simpler typed streams.

use std::error::Error;
use std::pin::Pin;

pub use futures::stream::BoxStream;
use futures::{Stream, StreamExt, stream};

use super::dto_response::GenerateContentResponse;

/// Type alias for boxed error used in stream results
pub type BoxError = Box<dyn Error + Send + Sync>;

/// Boxed stream of `GenerateContentResponse<T>` results.
///
/// Following the `Box*Stream` naming convention from the futures crate.
/// Contains full response DTOs from the Gemini API.
pub type BoxResponseStream<T = String> =
    Pin<Box<dyn Stream<Item = Result<GenerateContentResponse<T>, BoxError>> + Send>>;

/// Extension trait for extracting inner typed data from Gemini streaming responses.
///
/// Provides idiomatic stream transformation methods for unwrapping
/// `GenerateContentResponse<T>` to get the inner type `T`.
pub trait IntoInnerStreamExt<T>: Sized {
    /// Extracts the inner typed data from a streaming response.
    ///
    /// Transforms a `BoxResponseStream<T>` (stream of `GenerateContentResponse<T>`)
    /// into a simpler stream that yields the inner typed data `T` directly,
    /// filtering out responses that don't contain data.
    ///
    /// # Returns
    /// A `BoxStream<'static, Result<T, BoxError>>` containing only the inner typed data
    ///
    /// # Example
    /// ```rust,ignore
    /// use gemini::{IntoInnerStreamExt, BoxResponseStream};
    ///
    /// let inner_stream = response_stream.into_inner();
    /// ```
    fn into_inner(self) -> BoxStream<'static, Result<T, BoxError>>
    where
        T: Clone + Send + 'static;
}

impl<T> IntoInnerStreamExt<T> for BoxResponseStream<T> {
    fn into_inner(self) -> BoxStream<'static, Result<T, BoxError>>
    where
        T: Clone + Send + 'static,
    {
        Box::pin(stream::unfold(self, |mut stream| async move {
            loop {
                match stream.next().await {
                    Some(Ok(response)) => {
                        // Extract typed data from the response
                        if let Some(candidate) = response.first_candidate()
                            && let Some(first_part) = candidate.content.parts().first()
                            && let Some(data) = first_part.text()
                        {
                            return Some((Ok(data.clone()), stream));
                        }
                        // Continue to next item if no data in this response
                    }
                    Some(Err(e)) => {
                        return Some((Err(e), stream));
                    }
                    None => {
                        return None;
                    }
                }
            }
        }))
    }
}
