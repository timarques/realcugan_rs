# RealCugan-rs

**RealCugan-rs** is a Rust wrapper for the [Real-CUGAN ncnn Vulkan](https://github.com/nihui/realcugan-ncnn-vulkan). It provides a convenient interface for using realcugan-ncnn-vulkan.

## Installation

Install dependencies
```sh
dnf install vulkan-headers vulkan-loader-devel
```
```sh
apt-get install libvulkan-dev
```
```sh
pacman -S vulkan-headers vulkan-icd-loader
```

Add this to your Cargo.toml:

```toml
[dependencies]
realcugan-rs = { git = "https://github.com/timarques/realcugan_rs.git" }
```

```rs
use realcugan_rs::{RealCugan, Options, OptionsModel};
use image;

let realcugan = RealCugan::new(Options::default().model(OptionsModel::Pro2xNoDenoise)).unwrap();
let input_image = image::open("input.png").unwrap();
let output_image = realcugan.process_image(input_image)?;
output_image.save("output.png").unwrap();
```
## Advanced Configuration

The Builder pattern allows for detailed configuration:

```rs
use realcugan_rs::{RealCugan, Options, OptionsNoiseLevel, OptionsScaleFactor, OptionsSyncGap};

let realcugan = RealCugan::new(
    Options::default()
        .noise_level(OptionsNoiseLevel::Conservative)
        .scale_factor(OptionsScaleFactor::Double)
        .sync_gap(OptionsSyncGap::Strict)
        .gpuid(0)
        .tta_mode(false)
        .tile_size(0)
        .model_files(&format!("{}.param", MODEL), &format!("{}.bin", MODEL)).unwrap()
).unwrap();
```

## Features

This project uses feature flags to control optional dependencies and functionalities. Below is an explanation of the available features:

- **default = ["image", "models"]**
  The default feature set includes support for image processing using the Rust image crate, along with access to a variety of embedded AI-based upscaling models.

- **image**
  The image feature enables the use of the Rust image crate for tasks like decoding, encoding, and manipulating image data.

- **system-ncnn**
  The system-ncnn feature allows the project to link against an externally installed ncnn library on your system. This can be useful if you have pre-installed ncnn and want to avoid rebuilding it.

- **models = ["models-se", "models-pro", "models-nose"]**
  The models feature enables support for several embedded AI-based upscaling models. These models specialize in enhancing image and video quality based on different optimization strategies:

- **models-se:** Adds support for a "standard edition" model, focusing on general-purpose upscaling with balanced performance and quality.

- **models-pro:** Includes a "professional" edition model, designed for more advanced upscaling tasks that require finer detail and improved resolution.

- **models-nose:** Provides support for a noise reduction-focused model that is optimized for reducing artifacts and noise in images and videos while maintaining clarity.
