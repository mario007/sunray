use crate::vec::f32x3;
use crate::shapes::ShapeType;
use crate::isects::AABB;
use crate::math;
use crate::sampling;


pub struct LightSample {
    pub intensity: f32x3,
    pub position: f32x3,
    pub wi: f32x3,
    pub pdfa: f32,
    pub cos_theta: f32,
    pub delta_distribution: bool,
    pub valid: bool,
}

impl LightSample {
    pub fn new(intensity: f32x3, position: f32x3, wi: f32x3, pdfa: f32, cos_theta: f32, delta_distribution: bool, valid: bool) -> Self {
        Self {intensity, position, wi, pdfa, cos_theta, delta_distribution, valid}
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

    pub fn sample_li(&self, point: f32x3) -> LightSample {
        let wi = (self.position - point).normalize();
        let pdfa = 1.0f32;
        let cos_theta = 1.0f32;
        let intensity = self.intensity;
        LightSample::new(intensity, self.position, wi, pdfa, cos_theta, true, true)
    }

    pub fn precompute(&self, _world_bound: &AABB) {

    }
}

pub struct AreaLight {
    pub shape_type: ShapeType,
    pub shape_id: u32,
    pub intensity: f32x3,
}

impl AreaLight {
    pub fn new(shape_type: ShapeType, shape_id: u32, intensity: f32x3) -> Self {
        Self {shape_type, shape_id, intensity}
    }

    pub fn precompute(&self, _world_bound: &AABB) {
        
    }
}

pub struct DistantLight {
    wi_light: f32x3,
    intensity: f32x3,
    world_center: f32x3,
    world_radius: f32,
}

impl DistantLight {
    pub fn new(wi_light: f32x3, intensity: f32x3) -> Self {
        Self {wi_light, intensity, world_center: f32x3(0.0, 0.0, 0.0), world_radius: 1.0}
    }

    pub fn sample_li(&self, point: f32x3) -> LightSample {
        let wi = self.wi_light;
        let position = point + 2.0 * self.world_radius * wi;
        let cos_theta = 1.0f32;
        let pdfa = sampling::pdfw_to_a(1.0f32, math::distance(position, point), cos_theta);
        let intensity = self.intensity;
        LightSample::new(intensity, position, wi, pdfa, cos_theta, true, true)
    }

    pub fn precompute(&mut self, world_bound: &AABB) {
        self.world_center = (world_bound.min_p + world_bound.max_p) * 0.5;
        self.world_radius = math::distance(world_bound.min_p, world_bound.max_p) * 0.5;
    }
}

pub enum Light {
    Point(PointLight),
    Area(AreaLight),
    Distant(DistantLight),
}


impl Light {
    pub fn precompute(&mut self, world_bound: &AABB) {
        match self {
            Light::Point(point) => point.precompute(world_bound),
            Light::Area(area) => area.precompute(world_bound),
            Light::Distant(distant) => distant.precompute(world_bound),
        }
    }
}
