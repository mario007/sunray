extern crate image;
use std::path::Path;
use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;


#[derive(Debug, Copy, Clone)]
pub struct Color {
    pub red: u8,
    pub green: u8,
    pub blue: u8
}

pub struct ColorBufferRGBA {
    width: usize,
    height: usize,
    pixels: Vec<Color>,
}

impl ColorBufferRGBA {
    pub fn new(width: usize, height: usize) -> ColorBufferRGBA {
        ColorBufferRGBA {width,
                         height,
                         pixels: vec![Color{red:0, green:0, blue:0}; width * height]
        }
    }
    
    pub fn get_color(&self, x: usize, y: usize) -> Color {
        return self.pixels[y * self.width + x]
    }
    
    pub fn set_color(&mut self, x: usize, y: usize, rgba: &Color) {
        self.pixels[y * self.width + x] = *rgba;
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        let output: Vec<u8> = self.pixels.iter().flat_map(|val| [val.red, val.green, val.blue]).collect();
        let result = image::save_buffer(path,
                                        &output[0..output.len()],
                                        self.width as u32,
                                        self.height as u32,
                                        image::ColorType::Rgb8);
        match result {
            Ok(_) => Ok(()),
            Err(err) => Err(err.into()),
        }
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
        ColorBuffer{ width,
                     height,
                     pixels: vec![0.0; width * height * 4]
        }
    }

    pub fn get_color(&self, x: usize, y:usize) -> (f32, f32, f32) {
        let index = y * self.width * 4 + x * 4;
        let r = self.pixels[index];
        let g = self.pixels[index+1];
        let b = self.pixels[index+2];
        let weight = self.pixels[index+3];
        if weight > 0.0 {
            let factor = 1.0 / weight;
            (r * factor, g * factor, b * factor)
        } else {
            (r, g, b)
        }
    }

    pub fn add_color(&mut self, x: usize, y:usize, r: f32, g: f32, b: f32, weight: f32) {
        // if r.is_nan() || g.is_nan() || b.is_nan() {
        //     println!("Nan {} {} {} {} {}", x, y, r, g, b);
        //     return;
        // }
        let index = y * self.width * 4 + x * 4;
        self.pixels[index] += r;
        self.pixels[index + 1] += g;
        self.pixels[index + 2] += b;
        self.pixels[index + 3] += weight;
    }

    pub fn to_rgb(&self, tmo_type: TMOType) -> ColorBufferRGBA {
        let mut buf = ColorBufferRGBA::new(self.width, self.height);

        for y in 0..self.height {
            for x in 0..self.width {
                let (r, g, b) = self.get_color(x, y);
                let (mut r, mut g, mut b) = tone_map(&tmo_type, r, g, b); 
                r *= 256.0;
                g *= 256.0;
                b *= 256.0;
                buf.set_color(x, y, &Color{red: r as u8, green: g as u8, blue: b as u8});
            }
        }
        buf
    }

    pub fn to_rgb_vector(&self, tmo_type: TMOType) -> Vec<u32> {
        let mut buffer: Vec<u32> = vec![0; self.width * self.height];

        for y in 0..self.height {
            for x in 0..self.width {
                let (r, g, b) = self.get_color(x, y);
                let (mut r, mut g, mut b) = tone_map(&tmo_type, r, g, b);
                r *= 256.0;
                g *= 256.0;
                b *= 256.0;
                let (r, g, b) = (r as u8, g as u8, b as u8);
                let color = ((r as u32) << 16) | ((g as u32) << 8) | b as u32;
                buffer[y * self.width + x] = color;
            }
        }
        buffer
    }

    pub fn save<P: AsRef<Path>>(&self, path: P, tmo_type: TMOType) -> Result<(), Box<dyn Error>> {
        let ext = Path::new(path.as_ref()).extension();
        match ext {
            None => Err("There is no filename or embedded .".into()),
            Some(os_str) => match os_str.to_str() {
                Some("pfm") => self.save_as_pfm(path),
                Some("hdr") => self.save_as_hdr(path),
                _ => self.to_rgb(tmo_type).save(path),
            }
        }
    }

    fn save_as_pfm<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {

        let file = File::create(&path)?;
        let mut buf_writer = BufWriter::new(file);
        let _result = buf_writer.write_all(b"PF\n");
        let _result = buf_writer.write_fmt(format_args!("{} {}\n", self.width, self.height));
        let _result = buf_writer.write_all(b"-1\n");
        for y in 0..self.height {
            for x in 0..self.width {
                let (r, g, b) = self.get_color(x, self.height - y - 1);
                    let _result = buf_writer.write_all(&f32::to_le_bytes(r));
                    let _result = buf_writer.write_all(&f32::to_le_bytes(g));
                    let _result = buf_writer.write_all(&f32::to_le_bytes(b));
            }
        }
        Ok(())
    }

    fn save_as_hdr<P: AsRef<Path>>(&self, _path: P) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}
