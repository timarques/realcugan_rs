use std::path::Path;

const IMAGE: &str = "./tests/image.jpg";
const MODEL: &str = "./tests/up2x-conservative";

#[test]
fn base() {
    // Assert that the test image exists
    assert!(Path::new(IMAGE).exists(), "Test image does not exist");

    // Create RealCugan instance
    let result = realcugan_rs::RealCugan::from_files(
        &format!("{}.param", MODEL),
        &format!("{}.bin", MODEL)
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

}

#[test]
fn threads() {
    let realcugan = realcugan_rs::RealCugan::from_files(
        &format!("{}.param", MODEL),
        &format!("{}.bin", MODEL)
    ).unwrap();

    let mut threads = Vec::new();

    for i in 0..10 {

        let realcugan_clone = realcugan.clone();

        let handle = std::thread::spawn(move || {
            let result = realcugan_clone.process_image_from_path(&std::path::PathBuf::from(IMAGE));
            assert!(result.is_ok());
            let upscaled_image = result.unwrap();
            let path = format!("/tmp/upscaled{}.png", i);
            upscaled_image.save_with_format(&path, image::ImageFormat::Png).unwrap();
            assert!(Path::new(&path).exists(), "Failed to save upscaled image");
        });

        threads.push(handle);

    }

    for thread in threads {
        assert!(thread.join().is_ok());
    }
}