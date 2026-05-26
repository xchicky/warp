use super::*;

#[test]
fn test_is_supported_image_mime_type() {
    assert!(is_supported_image_mime_type("image/png"));
    assert!(is_supported_image_mime_type("image/jpeg"));
    assert!(is_supported_image_mime_type("image/jpg"));
    assert!(is_supported_image_mime_type("image/gif"));
    assert!(is_supported_image_mime_type("image/webp"));
    assert!(!is_supported_image_mime_type("image/bmp"));
    assert!(!is_supported_image_mime_type("application/pdf"));
    assert!(!is_supported_image_mime_type("text/plain"));
}

/// Creates a small test PNG image and returns its bytes.
fn create_small_test_png() -> Vec<u8> {
    use image::{ImageBuffer, Rgba};
    // Create a small 10x10 red image
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_fn(10, 10, |_x, _y| Rgba([255u8, 0u8, 0u8, 255u8]));

    let mut bytes: Vec<u8> = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut bytes);
    img.write_to(&mut cursor, image::ImageFormat::Png).unwrap();
    bytes
}

fn create_small_test_jpeg() -> Vec<u8> {
    use image::{ImageBuffer, Rgb};
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_fn(10, 10, |_x, _y| Rgb([255u8, 0u8, 0u8]));

    let mut bytes: Vec<u8> = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut bytes);
    img.write_to(&mut cursor, image::ImageFormat::Jpeg).unwrap();
    bytes
}

fn insert_jpeg_app1_segment(jpeg: &[u8], payload: &[u8]) -> Vec<u8> {
    assert!(jpeg.starts_with(&[0xff, 0xd8]));
    let segment_len = u16::try_from(payload.len() + 2).unwrap();
    let mut with_app1 = Vec::with_capacity(jpeg.len() + payload.len() + 4);
    with_app1.extend_from_slice(&jpeg[..2]);
    with_app1.extend_from_slice(&[0xff, 0xe1]);
    with_app1.extend_from_slice(&segment_len.to_be_bytes());
    with_app1.extend_from_slice(payload);
    with_app1.extend_from_slice(&jpeg[2..]);
    with_app1
}

fn bytes_contain(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

#[test]
fn test_process_image_for_agent_small_image() {
    let small_png = create_small_test_png();
    let result = process_image_for_agent(&small_png);

    match result {
        ProcessImageResult::Success { data } => {
            // The processed image should be non-empty
            assert!(!data.is_empty());
            // For a small image, it should not be resized, so data should be similar
            assert!(data.len() <= MAX_IMAGE_SIZE_BYTES);
        }
        other => panic!("Expected Success, got {:?}", other),
    }
}

#[test]
fn test_resize_image_small_image_unchanged() {
    let small_png = create_small_test_png();
    let original_len = small_png.len();

    let result = resize_image(&small_png).unwrap();
    // Small images should be returned as-is
    assert_eq!(result.len(), original_len);
}

#[test]
fn test_process_image_for_agent_invalid_data() {
    let invalid_data = vec![0u8; 100];
    let result = process_image_for_agent(&invalid_data);

    match result {
        ProcessImageResult::Error(_) => {
            // Expected - invalid data should produce an error
        }
        other => panic!("Expected Error, got {:?}", other),
    }
}

#[test]
fn test_normalize_image_for_agent_strips_jpeg_exif_location_metadata() {
    let payload = b"Exif\0\0GPSLatitude=37.7749;GPSLongitude=-122.4194;CameraSerial=SECRET";
    let jpeg = create_small_test_jpeg();
    let jpeg_with_exif = insert_jpeg_app1_segment(&jpeg, payload);

    assert!(bytes_contain(&jpeg_with_exif, payload));

    let result = normalize_image_for_agent(&jpeg_with_exif, "image/jpeg");
    let ProcessImageResult::Success { data } = result else {
        panic!("Expected Success, got {:?}", result);
    };

    assert!(!bytes_contain(&data, b"GPSLatitude=37.7749"));
    assert!(!bytes_contain(&data, b"GPSLongitude=-122.4194"));
    assert!(!bytes_contain(&data, b"CameraSerial=SECRET"));
}

/// Creates a large test PNG image that exceeds MAX_IMAGE_PIXELS.
fn create_large_test_png() -> Vec<u8> {
    use image::{ImageBuffer, Rgba};
    // Create a 2000x1000 image (2M pixels, exceeds MAX_IMAGE_PIXELS of 1.15M)
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_fn(2000, 1000, |_x, _y| Rgba([255u8, 0u8, 0u8, 255u8]));

    let mut bytes: Vec<u8> = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut bytes);
    img.write_to(&mut cursor, image::ImageFormat::Png).unwrap();
    bytes
}

#[test]
fn test_resize_image_large_image_gets_resized() {
    let large_png = create_large_test_png();

    // Verify the image exceeds the pixel limit
    let img = image::load_from_memory(&large_png).unwrap();
    let (width, height) = img.dimensions();
    let original_pixels = (width * height) as f64;
    assert!(original_pixels > MAX_IMAGE_PIXELS);

    let result = resize_image(&large_png).unwrap();

    // The resized image should be smaller
    let resized_img = image::load_from_memory(&result).unwrap();
    let (new_width, new_height) = resized_img.dimensions();
    let new_pixels = (new_width * new_height) as f64;

    assert!(new_pixels <= MAX_IMAGE_PIXELS);
    assert!(new_width < width || new_height < height);
}

/// Creates a very tall/narrow image to test dimension clamping.
fn create_tall_test_png() -> Vec<u8> {
    use image::{ImageBuffer, Rgba};
    // Create a 100x20000 image (2M pixels, but very tall)
    // After pixel-based scaling, height would still exceed MAX_IMAGE_DIMENSION
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_fn(100, 20000, |_x, _y| Rgba([0u8, 255u8, 0u8, 255u8]));

    let mut bytes: Vec<u8> = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut bytes);
    img.write_to(&mut cursor, image::ImageFormat::Png).unwrap();
    bytes
}

#[test]
fn test_resize_image_respects_max_dimension() {
    let tall_png = create_tall_test_png();

    let result = resize_image(&tall_png).unwrap();

    let resized_img = image::load_from_memory(&result).unwrap();
    let (new_width, new_height) = resized_img.dimensions();

    // Both dimensions should be within MAX_IMAGE_DIMENSION
    assert!(
        (new_width as f64) <= MAX_IMAGE_DIMENSION,
        "width {} exceeds max {}",
        new_width,
        MAX_IMAGE_DIMENSION
    );
    assert!(
        (new_height as f64) <= MAX_IMAGE_DIMENSION,
        "height {} exceeds max {}",
        new_height,
        MAX_IMAGE_DIMENSION
    );
}
