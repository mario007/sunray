use std::f32;

use crate::vec::f32x3;
use crate::sampler::PathSampler;
use crate::bsdf;
use crate::sampling;
use crate::spectrum::luminance;


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

        let specular_weight =  luminance(self.ks)  / (luminance(self.kd) + luminance(self.ks));
        let specular_weight = specular_weight.clamp(0.001, 0.999);
        let diffuse_weight = 1.0 - specular_weight;
        let pdfw = diffuse_weight * sampling::cosine_hemi_pdf(normal.dot(wi).abs()) + specular_weight * bsdf::phong_pdf(wo, normal, wi, self.shininess);

        (diffuse + specular, pdfw)
    }

    pub fn sample(&self, wo: f32x3, normal: f32x3, path_sampler: &mut PathSampler) -> MaterialSample {
        let mut valid = true;
        let wi: f32x3;

        let specular_weight =  luminance(self.ks)  / (luminance(self.kd) + luminance(self.ks));
        if path_sampler.next_1d() < specular_weight {
            wi = bsdf::sample_phong(wo, normal, self.shininess, path_sampler.next_1d(), path_sampler.next_1d());
            if wi.dot(normal) < 0.001 {
                valid = false;
            }
        } else {
            wi = sampling::cosine_hemi_direction(normal, path_sampler.next_1d(), path_sampler.next_1d());
        }
        let (value, pdfw) = self.eval(wo, normal, wi);
        if pdfw < 1e-10 { valid = false; }
        MaterialSample::new(value, wi, pdfw, valid)
    }
}

pub struct WardMaterial {
    kd: f32x3,
    ks: f32x3,
    alpha_x: f32,
    alpha_y: f32,
}

impl WardMaterial {
    pub fn new(kd: f32x3, ks: f32x3, alpha_x: f32, alpha_y: f32) -> Self {
        Self {kd, ks, alpha_x, alpha_y}
    }

    pub fn eval(&self, wo: f32x3, normal: f32x3, wi: f32x3) -> (f32x3, f32) {
        let diffuse = self.kd * bsdf::lambertian();
        let specular = self.ks * bsdf::ward(wo, normal, wi, self.alpha_x, self.alpha_y);

        let specular_weight =  luminance(self.ks)  / (luminance(self.kd) + luminance(self.ks));
        let diffuse_weight = 1.0 - specular_weight;
        let pdfw = diffuse_weight * sampling::cosine_hemi_pdf(normal.dot(wi).abs()) + specular_weight * bsdf::ward_pdf(wo, normal, wi, self.alpha_x, self.alpha_y);

        (diffuse + specular, pdfw)
    }

    pub fn sample(&self, wo: f32x3, normal: f32x3, path_sampler: &mut PathSampler) -> MaterialSample {
        let mut valid = true;
        let wi: f32x3;

        let specular_weight =  luminance(self.ks)  / (luminance(self.kd) + luminance(self.ks));
        if path_sampler.next_1d() < specular_weight {
            wi = bsdf::sample_ward(wo, normal, self.alpha_x, self.alpha_y, path_sampler.next_1d(), path_sampler.next_1d());
            if wi.dot(normal) < 0.001 {
                valid = false;
            }
        } else {
            wi = sampling::cosine_hemi_direction(normal, path_sampler.next_1d(), path_sampler.next_1d());
        }
        let (value, pdfw) = self.eval(wo, normal, wi);
        if pdfw < 1e-10 { valid = false; }
        MaterialSample::new(value, wi, pdfw, valid)
    }
}


pub enum Material {
    Matte(MatteMaterial),
    Phong(PhongMaterial),
    Ward(WardMaterial),
}

impl Material {
    pub fn eval(&self, wo: f32x3, normal: f32x3, wi: f32x3) -> (f32x3, f32) {
        match self {
            Material::Matte(matte) => matte.eval(wo, normal, wi),
            Material::Phong(phong) => phong.eval(wo, normal, wi),
            Material::Ward(ward) => ward.eval(wo, normal, wi),
        }
    }

    pub fn sample(&self, wo: f32x3, normal: f32x3, path_sampler: &mut PathSampler) -> MaterialSample {
        match self {
            Material::Matte(matte) => matte.sample(wo, normal, path_sampler),
            Material::Phong(phong) => phong.sample(wo, normal, path_sampler),
            Material::Ward(ward) => ward.sample(wo, normal, path_sampler),
        }
    }
}
