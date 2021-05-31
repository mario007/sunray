use crate::scene_data::SceneData;
use crate::sampler::PathSampler;
use crate::vec::f32x3;
use crate::ray::Ray;
use crate::shapes::IsectPoint;
use crate::math;
use crate::sampling;


pub fn radiance_isect (ray: &Ray, scene_data: &SceneData, _path_sampler: &PathSampler) -> f32x3 {

    let isect_p = match scene_data.intersect(ray) {
        Some(isect_p) => isect_p,
        None => return f32x3(0.0, 0.0, 0.0)
    };
    let wo = -ray.direction;
    f32x3(1.0, 1.0, 1.0) * wo.dot(isect_p.normal).abs()
}

pub fn radiance_direct_lgt (ray: &Ray, scene_data: &SceneData, path_sampler: &mut PathSampler) -> f32x3 {
    let isect_p = match scene_data.intersect(ray) {
        Some(isect_p) => isect_p,
        None => return f32x3(0.0, 0.0, 0.0)
    };

    let wo = -ray.direction;
    let mat = scene_data.material(isect_p.material_id);
    let mut acum = f32x3(0.0, 0.0, 0.0);

    for light_id in 0..scene_data.lights.len() {
        let ls = scene_data.light_sample_li(light_id, &isect_p, path_sampler);
        if scene_data.visible(isect_p.position, ls.position) {
            let wi = (ls.position - isect_p.position).normalize();
            let (mat_spectrum, _pdfw) = mat.eval(wo, isect_p.normal, wi);
            let cosa = isect_p.normal.dot(wi).abs();
            let dist = math::distance(isect_p.position, ls.position);
            let pdf = sampling::pdfa_to_w(ls.pdfa, dist, ls.cos_theta);
            if wi.dot(isect_p.normal) > 0.0 && wo.dot(isect_p.normal) > 0.0 {
                acum = acum + mat_spectrum.mul(ls.intensity) * (cosa / pdf);
            }
        }
    }
    acum
}

pub fn radiance_path_tracer(ray: &Ray, scene_data: &SceneData, path_sampler: &mut PathSampler) -> f32x3 {
    let path_treshold = 0.001f32;
    let max_depth = 10;
    
    let mut path_t = f32x3(1.0, 1.0, 1.0);
    let mut depth = 0;

    let mut acum = f32x3(0.0, 0.0, 0.0);
    let mut origin = ray.origin;
    let mut direction = ray.direction;
    let mut last_pdfw = 1.0f32;
    let mut cos_theta = 1.0f32;

    loop {
        depth += 1;

        let loc_ray = Ray::new(origin, direction);

        let isect_p = match scene_data.intersect(&loc_ray) {
            Some(isect_p) => isect_p,
            None => return acum
        };

        let wo = -loc_ray.direction;
        if isect_p.light_id != -1 {
            let last_pdfa = sampling::pdfw_to_a(last_pdfw, math::distance(ray.origin, isect_p.position), cos_theta);
            acum = acum + path_t.mul(make_implicit_connection(scene_data, &isect_p, ray.direction, depth, last_pdfa));
        }
        acum = acum + path_t.mul(make_explict_connection(scene_data, &isect_p, wo, path_sampler));

        let bs = scene_data.material_sample(isect_p.material_id, wo, isect_p.normal, path_sampler);
        if !bs.valid { break; }
        cos_theta = bs.wi.dot(isect_p.normal).abs();
        path_t = path_t.mul(bs.value) * (cos_theta / bs.pdfw);
        last_pdfw = bs.pdfw;

        if depth >= max_depth { break; }
        if path_t.dot(path_t) < path_treshold { break; }

        origin = isect_p.position;
        direction = bs.wi;
    }
    acum
}

fn make_implicit_connection(scene_data: &SceneData, isect: &IsectPoint, direction_to_light: f32x3, depth: i32, last_pdfa: f32) -> f32x3 {
    let ls = scene_data.light_get_le(isect, direction_to_light);
    if !ls.valid { return f32x3(0.0, 0.0, 0.0); }
    let mut weight = 1.0f32;
    if depth > 1 {
        weight = last_pdfa / (last_pdfa + ls.pdfa);
    }
    weight * ls.intensity
}

fn make_explict_connection(scene_data: &SceneData, isect: &IsectPoint, wo: f32x3, path_sampler: &mut PathSampler) -> f32x3 {

    let ls = scene_data.lights_sample_li(isect, path_sampler);
    if !ls.valid { return f32x3(0.0, 0.0, 0.0); }
    if scene_data.visible(isect.position, ls.position) {
        let (value, pdfw) = scene_data.material_eval(isect.material_id, wo, isect.normal, ls.wi);
        let cos_theta = isect.normal.dot(ls.wi);
        let g = (ls.cos_theta * cos_theta).abs() / math::distance_sqr(isect.position, ls.position);
        let mut weight = 1.0f32;
        //let weight = ls.pdfa / (ls.pdfa + ((pdfw / cos_theta) * g));
        if !ls.delta_distribution {
            let bs_pdfa = sampling::pdfw_to_a(pdfw, math::distance(isect.position, ls.position), cos_theta);
            weight = ls.pdfa / (ls.pdfa + bs_pdfa);
        }
        if ls.wi.dot(isect.normal) > 0.0 && wo.dot(isect.normal) > 0.0 {
            return (weight * g / ls.pdfa) * value.mul(ls.intensity);
        }
    }
    f32x3(0.0, 0.0, 0.0)
}

