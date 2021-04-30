extern crate image;
use std::path::Path;


pub struct ColorBufferRGB {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl ColorBufferRGB {
    pub fn new(width: usize, height: usize) -> ColorBufferRGB {
        ColorBufferRGB { width: width,
                         height: height,
                         pixels: vec![0; width*height*3]
        }
    }
    
    pub fn get_color(&self, x: usize, y: usize) -> (u8, u8, u8) {
       let index = y * self.width * 3 + x * 3;
       (self.pixels[index], self.pixels[index+1], self.pixels[index+2])
    }
    
    pub fn set_color(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8) {
        let index = y * self.width * 3 + x * 3;
        self.pixels[index] = r;
        self.pixels[index + 1] = g;
        self.pixels[index + 2] = b;
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> image::error::ImageResult<()> {
        image::save_buffer(path,
                           &self.pixels[0..self.pixels.len()],
                           self.width as u32,
                           self.height as u32,
                           image::ColorType::Rgb8)
    }
}

pub enum TMOType {
    Linear,
    Gamma,
    Reinhard,

}

// http://filmicworlds.com/blog/filmic-tonemapping-operators/
fn tone_map(tmo_type: &TMOType, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    match tmo_type {
        TMOType::Linear => (r, g, b),
        TMOType::Gamma => (r.powf(1.0/2.2), g.powf(1.0/2.2), b.powf(1.0/2.2)),
        TMOType::Reinhard => {
            ((r / (r + 1.0)).powf(1.0/2.2),
            (g / (g + 1.0)).powf(1.0/2.2),
            (b / (b + 1.0)).powf(1.0/2.2))
        }
    }
}

pub struct ColorBuffer {
    width: usize,
    height: usize,
    pixels: Vec<f32>,
}

impl ColorBuffer {
    pub fn new(width: usize, height: usize) -> ColorBuffer {
        ColorBuffer{ width: width,
                     height: height,
                     pixels: vec![0.0; width*height*4]
        }
    }

    pub fn get_color(&self, x: usize, y:usize) -> (f32, f32, f32, f32) {
        let index = y * self.width * 4 + x * 4;
        (self.pixels[index], self.pixels[index+1], self.pixels[index+2], self.pixels[index+3])
    }

    pub fn add_color(&mut self, x: usize, y:usize, r: f32, g: f32, b: f32, weight: f32) {
        let index = y * self.width * 4 + x * 4;
        self.pixels[index] += r;
        self.pixels[index + 1] += g;
        self.pixels[index + 2] += b;
        self.pixels[index + 3] += weight;
    }

    pub fn to_rgb(&self, tmo_type: TMOType) -> ColorBufferRGB {
        let mut buf = ColorBufferRGB::new(self.width, self.height);

        for y in 0..self.height {
            for x in 0..self.width {
                let (mut r, mut g, mut b, weight) = self.get_color(x, y);
                if weight != 0.0 {
                    let factor = 1.0 / weight;
                    r = factor * r;
                    g = factor * g;
                    b = factor * b;
                }
                let (mut r, mut g, mut b) = tone_map(&tmo_type, r, g, b);
                r = r * 256.0;
                g = g * 256.0;
                b = b * 256.0;
                buf.set_color(x, y, r as u8, g as u8, b as u8);
            }
        }
        return buf;
    }
}
