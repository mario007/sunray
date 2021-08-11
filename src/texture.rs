use std::path::Path;
use std::error::Error;


use crate::vec::f32x3;
use crate::shapes::IsectPoint;


trait Evaluate {
    fn evaluate(&self, isect: &IsectPoint) -> f32x3;
}

struct TextureCache<'a> {
    textures: Vec<Box<dyn Evaluate + 'a>>,
}

impl<'a> TextureCache<'a> {
    pub fn new() -> Self {
        TextureCache{textures: Vec::new()}
    }

    pub fn add_texture<T>(&mut self, texture: T) -> usize 
        where T: Evaluate + 'a {
        self.textures.push(Box::new(texture));
        self.textures.len() - 1
    }

    pub fn evaluate(&self, texture_id: usize, isect: &IsectPoint) -> f32x3 {
        self.textures[texture_id].evaluate(isect)
    }
}

pub struct ConstantTexture {
    value: f32x3
}

impl ConstantTexture {
    pub fn new(value: f32x3) -> ConstantTexture {
        ConstantTexture{value}
    }
}

impl Evaluate for ConstantTexture {
    fn evaluate(&self, isect: &IsectPoint) -> f32x3 {
        self.value
    }
}

pub struct ImageTexture {
    texels: Vec<u8>
}

impl ImageTexture {
    pub fn from<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn Error>> {
        
        Ok(())
    }
}


#[cfg(test)]
mod tests {

    use super::*;
    use crate::shapes::{IsectPoint, ShapeType};

    #[test]
    fn test_texture_cache() {
        let mut cache = TextureCache::new();
        let texture = ConstantTexture::new(f32x3(0.5, 0.3, 0.5));
        let tex_id = cache.add_texture(texture);
        let isect = IsectPoint::new(f32x3::from(0.0), f32x3::from(0.0), 0.0, 0, ShapeType::None, 0, 0, 0);
        let val = cache.evaluate(tex_id, &isect);
        println!("Evaluate: {:?}", val);
    }
}
