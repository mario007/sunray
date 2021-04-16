use crate::vec::f32x3;
use crate::math;


pub struct LightSample {
    pub intensity: f32x3,
    pub position: f32x3,
    pub wi: f32x3,
    pub pdf: f32,
}

impl LightSample {
    pub fn new(intensity: f32x3, position: f32x3, wi: f32x3, pdf: f32) -> Self {
        Self{intensity, position, wi, pdf}
    }
}

pub struct PointLight {
    position: f32x3,
    intensity: f32x3,
}

impl PointLight {
    pub fn new(position: f32x3, intensity: f32x3) -> Self {
        Self{position, intensity}
    }

    pub fn sample_li(&self, _normal: f32x3, point: f32x3) -> LightSample {
        let wi = (self.position - point).normalize();
        let pdf = 1.0f32;
        let dst_sqr = math::distance_sqr(self.position, point);
        let intensity = self.intensity.div(f32x3(dst_sqr, dst_sqr, dst_sqr));
        LightSample::new(intensity, self.position, wi, pdf)
    }
}

pub enum Light {
    Point(PointLight)
}

impl Light {
    pub fn sample_li(&self, normal: f32x3, point: f32x3) -> LightSample {
        match self {
            Light::Point(point_light) => point_light.sample_li(normal, point),
        }

    }
}
