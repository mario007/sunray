use std::thread;
use std::sync::mpsc;
use std::sync::Arc;
use std::mem::drop;

use crate::scene::{Scene, IntegratorType};
use crate::scene_data::SceneData;
use crate::sampler::PathSampler;
use crate::vec::f32x3;
use crate::ray::Ray;
use crate::shapes::IsectPoint;
use crate::math;
use crate::sampling;


struct PixelValue {
    x: u32,
    y: u32,
    r: f32,
    g: f32,
    b: f32,
}

impl PixelValue {
    pub fn new(x: u32, y: u32, r: f32, g: f32, b: f32) -> PixelValue {
        PixelValue{x, y, r, g, b}
    }
}

pub fn render(scene: &mut Scene, rendering_pass: u32) {

    let n_threads = 16;
    let (tx, rx) = mpsc::channel();

    for cur_y_value in 0..n_threads {
        let start_y = cur_y_value;
        let t_sender = tx.clone();
        let options = Arc::clone(&scene.options);
        let camera = Arc::clone(&scene.camera);
        let scene_data = Arc::clone(&scene.scene_data);
        let step = n_threads as usize;
        thread::spawn(move || {
            let seed = start_y as u64 * 123456789 + 123456 * rendering_pass as u64;
            let mut sampler = PathSampler::new(options.sampler_type, options.xres, options.yres, options.n_samples, seed);
            for y in (start_y..options.yres).step_by(step) {
                for x in 0..options.xres {
                    let (xp, yp) = sampler.sample_pixel(x, y, rendering_pass);
                    let ray = camera.generate_ray(x as f32 + xp, y as f32 + yp);
                    let rad = match options.integrator_type {
                        IntegratorType::DirectLighting => radiance_direct_lgt(&ray, &scene_data, &mut sampler),
                        IntegratorType::PathTracer => radiance_path_tracer(&ray, &scene_data, &mut sampler),
                        IntegratorType::Isect => radiance_isect(&ray, &scene_data, &mut sampler),
                    };
                    let pixel = PixelValue::new(x, y, rad.0, rad.1, rad.2);
                    t_sender.send(pixel).expect("Pixel value not send!");
                }
            }
        });
    }

    drop(tx);
    for pix in rx {
        scene.color_buffer.add_color(pix.x as usize, pix.y as usize, pix.r, pix.g, pix.b, 1.0);
    }
}

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

fn radiance_path_tracer2(ray: &Ray, scene_data: &SceneData, path_sampler: &mut PathSampler) -> f32x3 {
    let path_treshold = 0.001f32;
    let max_depth = 6;
    
    let mut path_t = f32x3(1.0, 1.0, 1.0);
    let mut depth = 0;

    let mut acum = f32x3(0.0, 0.0, 0.0);
    let mut origin = ray.origin;
    let mut direction = ray.direction;

    loop {
        depth += 1;

        let loc_ray = Ray::new(origin, direction);
        let isect_p = match scene_data.intersect(&loc_ray) {
            Some(isect_p) => isect_p,
            None => return acum
        };

        let wo = -loc_ray.direction;
        acum = acum + path_t.mul(make_explict_connection2(scene_data, &isect_p, wo, path_sampler));

        let bs = scene_data.material_sample(isect_p.material_id, wo, isect_p.normal, path_sampler);
        if !bs.valid { break; }
        let cos_theta = bs.wi.dot(isect_p.normal).abs();
        path_t = path_t.mul(bs.value) * (cos_theta / bs.pdfw);

        if depth >= max_depth { break; }
        if path_t.dot(path_t) < path_treshold { break; }

        origin = isect_p.position;
        direction = bs.wi;
    }
    acum
}

fn make_explict_connection2(scene_data: &SceneData, isect: &IsectPoint, wo: f32x3, path_sampler: &mut PathSampler) -> f32x3 {

    let ls = scene_data.lights_sample_li(isect, path_sampler);
    if !ls.valid { return f32x3(0.0, 0.0, 0.0); }
    if scene_data.visible(isect.position, ls.position) {
        let (value, _pdfw) = scene_data.material_eval(isect.material_id, wo, isect.normal, ls.wi);
        let dist = math::distance(isect.position, ls.position);
        let pdf = sampling::pdfa_to_w(ls.pdfa, dist, ls.cos_theta);
        let cos_theta = isect.normal.dot(ls.wi).abs();
        if ls.wi.dot(isect.normal) > 0.0 && wo.dot(isect.normal) > 0.0 {
            return (cos_theta / pdf) * value.mul(ls.intensity);
        }
    }
    f32x3(0.0, 0.0, 0.0)
}
