use std::f32;

use crate::vec::f32x3;
use crate::sampler::PathSampler;
use crate::sampling;

pub struct MaterialSample {
    pub value: f32x3,
    pub wi: f32x3,
    pub pdfw: f32,
}

impl MaterialSample {
    pub fn new(value: f32x3, wi: f32x3, pdfw: f32) -> Self {
        Self {value, wi, pdfw}
    }
}

pub struct MatteMaterial {
    kd: f32x3
}

impl MatteMaterial {
    pub fn new(kd: f32x3) -> Self {
        Self {kd}
    }

    pub fn eval(&self, _wo: f32x3, normal: f32x3, wi: f32x3) -> (f32x3, f32) {
        let value = self.kd * f32::consts::FRAC_1_PI;
        let pdfw = sampling::cosine_hemi_pdf(normal.dot(wi).abs());
        (value, pdfw)
    }

    pub fn sample(&self, _wo: f32x3, normal: f32x3, path_sampler: &mut PathSampler) -> MaterialSample {
        let wi = sampling::cosine_hemi_direction(normal, path_sampler.next_1d(), path_sampler.next_1d());
        let pdfw = sampling::cosine_hemi_pdf(normal.dot(wi).abs());
        let value = self.kd * f32::consts::FRAC_1_PI;
        MaterialSample::new(value, wi, pdfw)
    }
}

pub enum Material {
    Matte(MatteMaterial)
}

impl Material {
    pub fn eval(&self, wo: f32x3, normal: f32x3, wi: f32x3) -> (f32x3, f32) {
        match self {
            Material::Matte(matte) => matte.eval(wo, normal, wi), 
        }
    }

    pub fn sample(&self, wo: f32x3, normal: f32x3, path_sampler: &mut PathSampler) -> MaterialSample {
        match self {
            Material::Matte(matte) => matte.sample(wo, normal, path_sampler), 
        }
    }
}
