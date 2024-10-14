use std::path::Path;
use realcugan_rs::{RealCugan, Options, OptionsScaleFactor, OptionsNoiseLevel};

const IMAGE: &str = "./tests/image.jpg";
const MODEL: &str = "./models/models-se/up2x-conservative";

#[test]
#[cfg(feature = "image")]
fn with_image() {
    // Assert that the test image exists
    assert!(Path::new(IMAGE).exists(), "Test image does not exist");

    let result = RealCugan::new(
        Options::default()
        .noise_level(OptionsNoiseLevel::Conservative)
        .scale_factor(OptionsScaleFactor::Double)
        .model_files(&format!("{}.param", MODEL), &format!("{}.bin", MODEL)).unwrap()
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
    let _ = std::fs::remove_file(&upscaled_save_path);

}

#[cfg(feature = "image")]
#[test]
fn with_threads() {
    let result = RealCugan::new(
        Options::default()
        .noise_level(OptionsNoiseLevel::Conservative)
        .scale_factor(OptionsScaleFactor::Double)
        .model_files(&format!("{}.param", MODEL), &format!("{}.bin", MODEL)).unwrap()
    );

    assert!(result.is_ok(), "{}", result.err().unwrap().to_string());
    let realcugan = result.unwrap();

    let mut threads = Vec::new();

    for i in 0..10 {

        let realcugan_clone = realcugan.clone();

        let handle = std::thread::spawn(move || {
            let result = realcugan_clone.process_file(&std::path::PathBuf::from(IMAGE));
            assert!(result.is_ok());
            let upscaled_image = result.unwrap();
            let path = format!("/tmp/upscaled{}.png", i);
            upscaled_image.save_with_format(&path, image::ImageFormat::Png).unwrap();
            assert!(Path::new(&path).exists(), "Failed to save upscaled image");
            let _ = std::fs::remove_file(&path);
        });

        threads.push(handle);

    }

    for thread in threads {
        assert!(thread.join().is_ok());
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
    let result = r.process_file(&std::path::PathBuf::from(IMAGE));
    assert!(result.is_ok());
    let upscaled_image = result.unwrap();
    let path = "/tmp/upscaled_embeded_models.png";
    upscaled_image.save_with_format(path, image::ImageFormat::Png).unwrap();
    assert!(Path::new(&path).exists(), "Failed to save upscaled image");
    let _ = std::fs::remove_file(&path);
}

#[cfg(feature = "models-se")]
#[test]
fn with_model_se() {
    let result = RealCugan::new(
        Options::default().model(realcugan_rs::OptionsModel::Se2xConservative)
    );
    assert!(result.is_ok(), "{}", result.err().unwrap().to_string());
    let r = result.unwrap();
    let result = r.process_file(&std::path::PathBuf::from(IMAGE));
    assert!(result.is_ok());
    let upscaled_image = result.unwrap();
    let path = "/tmp/upscaled_embeded_models.png";
    upscaled_image.save_with_format(path, image::ImageFormat::Png).unwrap();
    assert!(Path::new(&path).exists(), "Failed to save upscaled image");
    let _ = std::fs::remove_file(&path);
}