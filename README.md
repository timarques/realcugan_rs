# RealCugan-rs

**RealCugan-rs** is a Rust wrapper for the realcugan-ncnn-vulkan. It provides a convenient interface for using realcugan-ncnn-vulkan.

## Installation

Install dependencies
```sh
dnf install vulkan-headers vulkan-loader-devel ncnn-devel
```
```sh
apt-get install libvulkan-dev
build ncnn from source
```
```sh
pacman -S vulkan-headers vulkan-icd-loader ncnn
```

Add this to your Cargo.toml:

```toml
[dependencies]
realcugan-rs = { git = "https://github.com/timarques/realcugan_rs.git" }
```

```rs
use realcugan_rs::RealCugan;
use image;

let param_path = "path/to/param/file";
let bin_path = "path/to/bin/file";

let realcugan = RealCugan::build()
    .gpu(0)         // Use GPU 0, or .cpu() for CPU processing
    .scale(2)       // 2x upscaling
    .noise(0)       // No denoise
    .model_files(param_path, bin_path)
    .build()?;

let input_image = image::open("input.png").unwrap();
let output_image = realcugan.process_image(input_image)?;
output_image.save("output.png").unwrap();
```
## Advanced Configuration

The Builder pattern allows for detailed configuration:

```rs
use realcugan_rs::{RealCugan, SyncGap};

let realcugan = RealCugan::build()
    .gpu(0)
    .scale(3)
    .noise(1)
    .tta()
    .tile_size(400)
    .sync_gap(SyncGap::Moderate)
    .threads(4)
    .model_files(param_path, bin_path)
    .build()?;
```

## Built-in Models

RealCugan-rs supports built-in models when compiled with appropriate features. To use built-in models, add one of the following feature flags to your Cargo.toml:

- models: Enables all models
- models-nose: Enables support for nose models
- models-pro: Enables support for pro models
- models-se: Enables support for SE models

```toml
[dependencies]
realcugan-rs = { git = "https://github.com/timarques/realcugan_rs.git", features = ["models"] }
```

```rs
use realcugan_rs::{RealCugan, Model};
use image;

let realcugan = RealCugan::from_model(Model::Se2xHighDenoise);
let input_image = image::open("input.png").unwrap();
let output_image = realcugan.process_image(input_image)?;
output_image.save("output.png").unwrap();
```

## API Overview

- RealCugan::new(): Creates a new RealCugan instance with specified parameters.
- RealCugan::build(): Starts the builder pattern for custom configuration.
- RealCugan::from_model(): Creates an instance with a built-in model (requires feature flags).
- process_image(): Processes a DynamicImage.
- process_raw_image(): Processes a raw image buffer.
- process_image_from_path(): Processes an image file from a given path.

The new() method is a more direct way to create a RealCugan instance if you don't need the flexibility of the builder pattern. It's useful when you know all the parameters you need upfront.

```rs
let realcugan = RealCugan::new(
    gpu,
    threads,
    tta,
    sync_gap,
    tile_size,
    scale,
    noise,
    param,
    bin
)?;
```
