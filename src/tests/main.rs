use std::path::Path;
use realcugan_rs::{RealCugan, Options, OptionsScaleFactor, OptionsNoiseLevel};

const IMAGE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/tests/image.jpg");
const MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/models-se/up2x-conservative");

#[test]
#[cfg(feature = "image")]
fn with_image() {
    // Assert that the test image exists
    assert!(Path::new(IMAGE).exists(), "Test image does not exist");

    let result = RealCugan::new(
        Options::default()
        .noise_level(OptionsNoiseLevel::Conservative)
        .scale_factor(OptionsScaleFactor::Double)
        .model_files(&format!("{MODEL}.param"), &format!("{MODEL}.bin")).unwrap()
    );

    // Assert that RealCugan instance was created successfully
    assert!(result.is_ok(), "{}", result.err().unwrap().to_string());
    let realcugan = result.unwrap();

    // Open the original image
    let d_image = image::open(IMAGE).expect("Failed to open test image");
    let original_with = d_image.width();
    let original_height = d_image.height();

    // Upscale the image
    let upscaled_image = realcugan.process_image(d_image).expect("Failed to upscale image");

    // Save the upscaled image and assert it was saved
    let upscaled_save_path = "/tmp/upscaled.png";
    upscaled_image.save_with_format(upscaled_save_path, image::ImageFormat::Png).unwrap();
    assert!(Path::new(upscaled_save_path).exists(), "Failed to save upscaled image");

    // Compare dimensions of original and upscaled images
    let upscaled_dimensions = image::open(upscaled_save_path).unwrap();
    assert!(
        upscaled_dimensions.width() > original_with && upscaled_dimensions.height() > original_height,
        "Upscaled image is not larger than the original"
    );

    // Optionally, check file size to ensure upscaled image is larger
    let original_metadata = std::fs::metadata(IMAGE).unwrap();
    let upscaled_metadata = std::fs::metadata(upscaled_save_path).unwrap();
    assert!(
        upscaled_metadata.len() > original_metadata.len(),
        "Upscaled image file is not larger than the original"
    );
    let _ = std::fs::remove_file(upscaled_save_path);

}

#[cfg(feature = "image")]
#[test]
fn with_loop() {
    let result = RealCugan::new(
        Options::default()
        .noise_level(OptionsNoiseLevel::Conservative)
        .scale_factor(OptionsScaleFactor::Double)
        .model_files(&format!("{MODEL}.param"), &format!("{MODEL}.bin")).unwrap()
    );

    assert!(result.is_ok(), "{}", result.err().unwrap().to_string());
    let realcugan = result.unwrap();

    for i in 0..10 {

        let result = realcugan.process_file(std::path::PathBuf::from(IMAGE));
        assert!(result.is_ok());
        let upscaled_image = result.unwrap();
        let path = format!("/tmp/upscaled{i}.png");
        upscaled_image.save_with_format(&path, image::ImageFormat::Png).unwrap();
        assert!(Path::new(&path).exists(), "Failed to save upscaled image");
        assert!(std::fs::remove_file(&path).is_ok());
    }
}

#[cfg(feature = "models-pro")]
#[test]
fn with_model_pro() {
    let result = RealCugan::new(
        Options::default().model(realcugan_rs::OptionsModel::Pro2xNoDenoise)
    );
    assert!(result.is_ok(), "{}", result.err().unwrap().to_string());
    let r = result.unwrap();
    let result = r.process_file(std::path::PathBuf::from(IMAGE));
    assert!(result.is_ok());
    let upscaled_image = result.unwrap();
    let path = "/tmp/upscaled_models_pro.png";
    upscaled_image.save_with_format(path, image::ImageFormat::Png).unwrap();
    assert!(Path::new(&path).exists(), "Failed to save upscaled image");
    assert!(std::fs::remove_file(path).is_ok());
}

#[cfg(feature = "models-se")]
#[test]
fn with_model_se() {
    let result = RealCugan::new(
        Options::default().model(realcugan_rs::OptionsModel::Se2xConservative)
    );
    assert!(result.is_ok(), "{}", result.err().unwrap().to_string());
    let r = result.unwrap();
    let result = r.process_file(std::path::PathBuf::from(IMAGE));
    assert!(result.is_ok());
    let upscaled_image = result.unwrap();
    let path = "/tmp/upscaled_models_se.png";
    upscaled_image.save_with_format(path, image::ImageFormat::Png).unwrap();
    assert!(Path::new(&path).exists(), "Failed to save upscaled image");
    let _ = std::fs::remove_file(path);
}