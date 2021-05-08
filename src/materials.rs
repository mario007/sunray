use std::f32;

use crate::vec::f32x3;
use crate::sampler::PathSampler;
use crate::bsdf;
use crate::sampling;


pub struct MaterialSample {
    pub value: f32x3,
    pub wi: f32x3,
    pub pdfw: f32,
    pub valid: bool,
}

impl MaterialSample {
    pub fn new(value: f32x3, wi: f32x3, pdfw: f32, valid: bool) -> Self {
        Self {value, wi, pdfw, valid}
    }
}

pub struct MatteMaterial {
    kd: f32x3,
    roughness: f32,
}

impl MatteMaterial {
    pub fn new(kd: f32x3, roughness: f32) -> Self {
        Self {kd, roughness}
    }

    pub fn eval(&self, wo: f32x3, normal: f32x3, wi: f32x3) -> (f32x3, f32) {
        let mut value = self.kd * bsdf::lambertian();
        if self.roughness > 0.0 {
            value = self.kd * bsdf::oren_nayar(wo, normal, wi, self.roughness);
        }
        let pdfw = sampling::cosine_hemi_pdf(normal.dot(wi).abs());
        (value, pdfw)
    }

    pub fn sample(&self, wo: f32x3, normal: f32x3, path_sampler: &mut PathSampler) -> MaterialSample {
        let wi = sampling::cosine_hemi_direction(normal, path_sampler.next_1d(), path_sampler.next_1d());
        let (value, pdfw) = self.eval(wo, normal, wi);
        MaterialSample::new(value, wi, pdfw, true)
    }
}


pub struct PhongMaterial {
    kd: f32x3,
    ks: f32x3,
    shininess: f32,
}

impl PhongMaterial {
    pub fn new(kd: f32x3, ks: f32x3, shininess: f32) -> Self {
        Self {kd, ks, shininess}
    }

    pub fn eval(&self, wo: f32x3, normal: f32x3, wi: f32x3) -> (f32x3, f32) {
        let diffuse = self.kd * bsdf::lambertian();
        let specular = self.ks * bsdf::phong(wo, normal, wi, self.shininess);
        let pdfw = 0.5 * sampling::cosine_hemi_pdf(normal.dot(wi).abs()) + 0.5 * bsdf::phong_pdf(wo, normal, wi, self.shininess);
        (diffuse + specular, pdfw)
    }

    pub fn sample(&self, wo: f32x3, normal: f32x3, path_sampler: &mut PathSampler) -> MaterialSample {
        let mut valid = true;
        let wi: f32x3;
        if path_sampler.next_1d() > 0.5 {
            wi = bsdf::sample_phong(normal, self.shininess, path_sampler.next_1d(), path_sampler.next_1d());
            if wi.dot(normal) < 0.001 {
                valid = false;
            }
        } else {
            wi = sampling::cosine_hemi_direction(normal, path_sampler.next_1d(), path_sampler.next_1d());
        }
        let (value, pdfw) = self.eval(wo, normal, wi);
        MaterialSample::new(value, wi, pdfw, valid)
    }
}


pub enum Material {
    Matte(MatteMaterial),
    Phong(PhongMaterial)
}

impl Material {
    pub fn eval(&self, wo: f32x3, normal: f32x3, wi: f32x3) -> (f32x3, f32) {
        match self {
            Material::Matte(matte) => matte.eval(wo, normal, wi),
            Material::Phong(phong) => phong.eval(wo, normal, wi),
        }
    }

    pub fn sample(&self, wo: f32x3, normal: f32x3, path_sampler: &mut PathSampler) -> MaterialSample {
        match self {
            Material::Matte(matte) => matte.sample(wo, normal, path_sampler),
            Material::Phong(phong) => phong.sample(wo, normal, path_sampler),
        }
    }
}
