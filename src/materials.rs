use std::f32;
use crate::vec::f32x3;


pub struct MatteMaterial {
    kd: f32x3
}

impl MatteMaterial {
    pub fn new(kd: f32x3) -> Self {
        Self {kd}
    }

    pub fn eval(&self, _wo: f32x3, _normal: f32x3, _wi: f32x3) -> f32x3 {
        self.kd * (1.0 / f32::consts::PI) 
    }
}

pub enum Material {
    Matte(MatteMaterial)
}

impl Material {
    pub fn eval(&self, wo: f32x3, normal: f32x3, wi: f32x3) -> f32x3 {
        match self {
            Material::Matte(matte) => matte.eval(wo, normal, wi), 
        }
    }
}
