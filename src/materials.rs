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


pub enum ConductorParams {
    F0(f32x3),
    IOR(f32x3, f32x3),
}

pub enum MicrofacetDistType {
    GGX,
    Beckmann,
}

pub struct MetalMaterial {
    con_params: ConductorParams,
    alpha: f32,
    dist_type: MicrofacetDistType,
}

impl MetalMaterial {
    pub fn new(con_params: ConductorParams, alpha: f32, dist_type: MicrofacetDistType) -> Self {
        let alpha = alpha.max(0.001);
        Self {con_params, alpha, dist_type}
    }

    pub fn eval(&self, wo: f32x3, normal: f32x3, wi: f32x3) -> (f32x3, f32) {
        let h = (wo + wi).normalize();
        let fresnel = match self.con_params {
            ConductorParams::F0(val) => bsdf::fresnel_schlick(val, wi, h),
            ConductorParams::IOR(eta, etak) => bsdf::fresnel_conductor(eta, etak, h.dot(wi)),
        };

        let lambda_wo: f32;
        let lambda_wi: f32;
        let pdfw: f32;
        let d: f32;

        match self.dist_type {
            MicrofacetDistType::Beckmann => {
                d = bsdf::beckmann_dist(self.alpha, wo, normal, wi);
                lambda_wo = bsdf::beckmann_lambda(self.alpha, normal, wo);
                lambda_wi = bsdf::beckmann_lambda(self.alpha, normal, wi);
                pdfw = d * normal.dot(h) / (4.0 * wo.dot(h));
            }
            MicrofacetDistType::GGX => {
                d = bsdf::ggx_dist(self.alpha, wo, normal, wi);
                lambda_wo = bsdf::ggx_lambda(self.alpha, normal, wo);
                lambda_wi = bsdf::ggx_lambda(self.alpha, normal, wi);
                //pdfw = d * normal.dot(h) / (4.0 * wo.dot(h));
                // D_Ve(Ne) = G1(Ve) * max(0, dot(Ve, Ne)) * D(Ne) / Ve.z
                pdfw = bsdf::smith_g1(lambda_wo) * d / (4.0 * normal.dot(wo));
            }
        } 

        let g2 = bsdf::smith_g2(lambda_wo, lambda_wi);
        let denom = 4.0 * normal.dot(wi) * normal.dot(wo);
        let specular = fresnel * d * g2 * denom.recip();

        (specular, pdfw)
    }

    pub fn sample(&self, wo: f32x3, normal: f32x3, path_sampler: &mut PathSampler) -> MaterialSample {
        let mut valid = true;
    
        let wi = match self.dist_type {
            MicrofacetDistType::Beckmann => bsdf::sample_beckmann(wo, normal, self.alpha, path_sampler.next_1d(), path_sampler.next_1d()),
            MicrofacetDistType::GGX => bsdf::sample_ggxvndf(wo, normal, self.alpha, self.alpha, path_sampler.next_1d(), path_sampler.next_1d())
            //bsdf::sample_ggx(wo, normal, self.alpha, path_sampler.next_1d(), path_sampler.next_1d());
        };
        if wi.dot(normal) < 0.001 {
            valid = false;
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
    Metal(MetalMaterial),
}

impl Material {
    pub fn eval(&self, wo: f32x3, normal: f32x3, wi: f32x3) -> (f32x3, f32) {
        match self {
            Material::Matte(matte) => matte.eval(wo, normal, wi),
            Material::Phong(phong) => phong.eval(wo, normal, wi),
            Material::Ward(ward) => ward.eval(wo, normal, wi),
            Material::Metal(metal) => metal.eval(wo, normal, wi),
        }
    }

    pub fn sample(&self, wo: f32x3, normal: f32x3, path_sampler: &mut PathSampler) -> MaterialSample {
        match self {
            Material::Matte(matte) => matte.sample(wo, normal, path_sampler),
            Material::Phong(phong) => phong.sample(wo, normal, path_sampler),
            Material::Ward(ward) => ward.sample(wo, normal, path_sampler),
            Material::Metal(metal) => metal.sample(wo, normal, path_sampler),
        }
    }
}

pub trait SurfaceMaterial {
    fn eval(&self, wo: f32x3, normal: f32x3, wi: f32x3) -> (f32x3, f32);
    fn sample(&self, wo: f32x3, normal: f32x3, path_sampler: &mut PathSampler) -> MaterialSample;
}

impl SurfaceMaterial for MatteMaterial {
    fn eval(&self, wo: f32x3, normal: f32x3, wi: f32x3) -> (f32x3, f32) {
        self.eval(wo, normal, wi)
    }

    fn sample(&self, wo: f32x3, normal: f32x3, path_sampler: &mut PathSampler) -> MaterialSample{
        self.sample(wo, normal, path_sampler)
    }
}


impl SurfaceMaterial for PhongMaterial {
    fn eval(&self, wo: f32x3, normal: f32x3, wi: f32x3) -> (f32x3, f32) {
        self.eval(wo, normal, wi)
    }

    fn sample(&self, wo: f32x3, normal: f32x3, path_sampler: &mut PathSampler) -> MaterialSample{
        self.sample(wo, normal, path_sampler)
    }
}

impl SurfaceMaterial for WardMaterial {
    fn eval(&self, wo: f32x3, normal: f32x3, wi: f32x3) -> (f32x3, f32) {
        self.eval(wo, normal, wi)
    }

    fn sample(&self, wo: f32x3, normal: f32x3, path_sampler: &mut PathSampler) -> MaterialSample{
        self.sample(wo, normal, path_sampler)
    }
}

impl SurfaceMaterial for MetalMaterial {
    fn eval(&self, wo: f32x3, normal: f32x3, wi: f32x3) -> (f32x3, f32) {
        self.eval(wo, normal, wi)
    }

    fn sample(&self, wo: f32x3, normal: f32x3, path_sampler: &mut PathSampler) -> MaterialSample{
        self.sample(wo, normal, path_sampler)
    }
}
