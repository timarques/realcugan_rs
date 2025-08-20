use thiserror::Error;

#[derive(Debug, Error, Clone)]
pub enum Error {
    #[error("GPU {requested} not found. Available GPUs: {available}")]
    GpuNotFound { requested: i32, available: i32 },

    #[error("Invalid model provided. Both parameter and binary data must be non-empty.")]
    InvalidModel,

    #[error("Failed to create in-memory file pointers for model data.")]
    FilePointerCreationFailed,

    #[error("Failed to load model files. Error code: {code}")]
    ModelLoadFailed { code: i32 },

    #[error("The instance pointer is null. It may have been dropped or failed to initialize.")]
    InvalidPointer,

    #[error("Failed to initialize instance")]
    InitializationFailed,

    #[error("Invalid input dimensions: expected byte length to be a multiple of {expected_length}, but got {actual_length}.")]
    InvalidInput { expected_length: usize, actual_length: usize },

    #[error("Failed to process the image. Error code: {code}")]
    ProcessingFailed { code: i32 },

    #[cfg(feature = "image")]
    #[error("Failed to open image file: {0}")]
    ImageOpenFailed(String),

    #[cfg(feature = "image")]
    #[error("Failed to convert the processed image buffer to the target color type.")]
    ColorConversionFailed,
}