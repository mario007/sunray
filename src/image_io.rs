extern crate image;
use image::io::Reader as ImageReader;

use std::path::Path;
use std::error::Error;

pub struct Image {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl Image {
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Self {
        Image{width, height, pixels}
    }
}

pub fn read_image<P: AsRef<Path>>(path: P) -> Result<Image, Box<dyn Error>> {
    let ext = Path::new(path.as_ref()).extension();
    match ext {
        None => Err(format!("Extension is missing! {}", path.as_ref().to_str().expect("Filename error!")).into()),
        Some(os_str) => match os_str.to_str() {
            Some("png") => load_rgb_image(path),
            _ => load_rgb_image(path),
        }
    }
}

fn load_rgb_image<P: AsRef<Path>>(path: P) -> Result<Image, Box<dyn Error>> {

    let img = ImageReader::open(path)?.decode()?;
    let img = img.to_rgb8();
    let (width, height) = img.dimensions();

    // TODO allocate memory in advance
    let mut pixels = Vec::<u8>::new();

    for y in 0..height {
        for x in 0..width {
            let pix = img.get_pixel(x, height - y - 1);
            pixels.push(pix[0]);
            pixels.push(pix[1]);
            pixels.push(pix[2]);
        }
    }
    
    let image = Image::new(width, height, pixels);
    Ok(image)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_image_loading() {
        let image = read_image("E:\\pbrt v3 scenes\\pbrt-v3-scenes\\barcelona-pavilion\\textures\\url.jpeg");
        match image {
            Ok(img) => println!("Image dimenstions {} {}", img.width, img.height),
            Err(err) => println!("{}", err),
        }
    }
}
