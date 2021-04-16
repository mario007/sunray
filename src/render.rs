use std::thread;
use std::sync::mpsc;
use std::sync::Arc;
use std::mem::drop;

use crate::scene::{Scene, IntegratorType};
use crate::scene_data::SceneData;
use crate::sampler::PathSampler;
use crate::vec::f32x3;
use crate::ray::Ray;


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

fn calc_rendering_blocks(y_res: u32, n_threads: u32) -> Vec<u32> {

    let step_size = y_res / n_threads;
    let mut cur_y: u32 = 0;
    let mut y_values = vec![cur_y]; 

    while cur_y < y_res {
        cur_y += step_size;
        if cur_y < y_res {
            y_values.push(cur_y);
        } else {
            y_values.push(y_res);
        }
    }
    return y_values;
}

pub fn render(scene: &mut Scene, rendering_pass: u32) {

    let n_threads = 1;
    let rend_blocks = calc_rendering_blocks(scene.options.yres, n_threads);
    let (tx, rx) = mpsc::channel();

    let mut prev_y_value = rend_blocks[0];
    for cur_y_value in rend_blocks.iter().skip(1) {
        let y_val = *cur_y_value;
        let t_sender = tx.clone();
        let options = Arc::clone(&scene.options);
        let camera = Arc::clone(&scene.camera);
        let scene_data = Arc::clone(&scene.scene_data);
        thread::spawn(move || {
            let seed = y_val as u64 * 123456789 + 123456 * rendering_pass as u64;
            let mut sampler = PathSampler::new(options.sampler_type, options.xres, options.yres, options.n_samples, seed);
            
            for y in prev_y_value..y_val {
                for x in 0..options.xres {
                    let (xp, yp) = sampler.sample_pixel(x, y, rendering_pass);
                    let ray = camera.generate_ray(x as f32 + xp, y as f32 + yp);
                    let rad = match options.integrator_type {
                        IntegratorType::DirectLighting => radiance_direct_lgt(&ray, &scene_data, &sampler),
                        IntegratorType::Isect => radiance_isect(&ray, &scene_data, &sampler),
                    };
                    let pixel = PixelValue::new(x, y, rad.0, rad.1, rad.2);
                    t_sender.send(pixel).expect("Pixel value not send!");
                }
            }
        });
        prev_y_value = *cur_y_value;
    }

    drop(tx);
    for pix in rx {
        scene.color_buffer.add_color(pix.x as usize, pix.y as usize, pix.r, pix.g, pix.b, 1.0);
    }
}

fn radiance_isect (ray: &Ray, scene_data: &SceneData, _path_sampler: &PathSampler) -> f32x3 {

    let isect_p = match scene_data.intersect(ray) {
        Some(isect_p) => isect_p,
        None => return f32x3(0.0, 0.0, 0.0)
    };
    let wo = -ray.direction;
    f32x3(1.0, 1.0, 1.0) * wo.dot(isect_p.normal).abs()
}

fn radiance_direct_lgt (ray: &Ray, scene_data: &SceneData, _path_sampler: &PathSampler) -> f32x3 {
    let isect_p = match scene_data.intersect(ray) {
        Some(isect_p) => isect_p,
        None => return f32x3(0.0, 0.0, 0.0)
    };

    let wo = -ray.direction;
    let mat = scene_data.material(isect_p.material_id);
    let mut acum = f32x3(0.0, 0.0, 0.0);

    for light in scene_data.lights.iter() {
        let ls = light.sample_li(isect_p.normal, isect_p.position);
        if scene_data.visible(isect_p.position, ls.position) {
            let mat_spectrum = mat.eval(wo, isect_p.normal, ls.wi);
            let cosa = isect_p.normal.dot(ls.wi).abs();
            acum = acum + mat_spectrum.mul(ls.intensity) * cosa * (1.0 / ls.pdf);
        }
    }
    acum
}
