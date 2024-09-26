use std::path::Path;

#[test]
fn base() {
    // Use a test image path instead of command line arguments
    let image = "./tests/image.jpg";
    let model = "./tests/up2x-conservative";

    // Assert that the test image exists
    assert!(Path::new(image).exists(), "Test image does not exist");

    // Create RealCugan instance
    let result = realcugan_rs::RealCuganBuilder::new_from_files(
        format!("{}.bin", model), 
        format!("{}.param", model)
    ).build();

    // Assert that RealCugan instance was created successfully
    assert!(result.is_ok(), "{}", result.err().unwrap().to_string());
    let realcugan = result.unwrap();

    // Open the original image
    let d_image = image::open(image).expect("Failed to open test image");
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
    let original_metadata = std::fs::metadata(image).unwrap();
    let upscaled_metadata = std::fs::metadata(upscaled_save_path).unwrap();
    assert!(
        upscaled_metadata.len() > original_metadata.len(),
        "Upscaled image file is not larger than the original"
    );
}