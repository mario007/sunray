use std::sync::Arc;

use crate::matrix::Matrix4x4;
use crate::camera::{Camera, PerspectiveCamera};
use crate::sampler::SamplerType;
use crate::buffers::ColorBuffer;
use crate::shapes::{Sphere, ShapeInstance, TransformShape, Mesh, ShapeType};
use crate::materials::Material;
use crate::lights::Light;
use crate::scene_data::SceneData;


pub enum IntegratorType {
    Isect,
    DirectLighting,
    PathTracer,
}

pub struct SceneOptions {
    pub xres: u32,
    pub yres: u32,
    pub sampler_type: SamplerType,
    pub n_samples: u32,
    pub integrator_type: IntegratorType
}

impl SceneOptions {
    pub fn new(xres: u32, yres: u32, sampler_type: SamplerType, n_samples: u32, integrator_type: IntegratorType) -> Self {
        Self {xres, yres, sampler_type, n_samples, integrator_type}
    }
}

pub struct Scene {
    pub output_filename: String,
    pub color_buffer: ColorBuffer,

    pub options: Arc<SceneOptions>,
    pub camera: Arc<Camera>,
    pub scene_data: Arc<SceneData>
}

impl Scene {
    pub fn new() -> Self {
        Scene {
            output_filename: String::new(),
            color_buffer: ColorBuffer::new(200, 200),
            options: Arc::new(SceneOptions::new(200, 200, SamplerType::Sobol, 16, IntegratorType::DirectLighting)),
            camera: Arc::new(Camera::Perspective(PerspectiveCamera::new(200, 200, 90.0))),
            scene_data: Arc::new(SceneData::default()),
        }        
    }

    pub fn set_camera_to_world(&mut self, matrix: Matrix4x4) {
        let camera = Arc::get_mut(&mut self.camera).expect("Camera cannot be aquired!");
        camera.set_camera_to_world(matrix);
    }

    pub fn set_integrator_type(&mut self, integrator_type: IntegratorType) {
        let options = Arc::get_mut(&mut self.options).expect("Scene data cannot be aquired!");
        options.integrator_type = integrator_type;
    }

    pub fn set_resolution(&mut self, xres: u32, yres: u32) {
        let options = Arc::get_mut(&mut self.options).expect("Scene data cannot be aquired!");
        options.xres = xres;
        options.yres = yres;
    }

    pub fn get_resolution(&self) -> (u32, u32) {
        (self.options.xres, self.options.yres)
    }

    pub fn set_output_filename(&mut self, filename: String) {
        self.output_filename = filename;
    }

    pub fn set_camera(&mut self, camera: Camera) {
        self.camera = Arc::new(camera);
    }

    pub fn set_camera_aspect_ratio(&mut self, aspect_ratio: Option<f32>) {
        let camera = Arc::get_mut(&mut self.camera).expect("Camera cannot be aquired!");
        camera.set_aspect_ratio(aspect_ratio);
    }

    pub fn set_sampler(&mut self, sampler_type: SamplerType, n_samples: u32) {
        let options = Arc::get_mut(&mut self.options).expect("Scene data cannot be aquired!");
        options.sampler_type = sampler_type;
        options.n_samples = n_samples;
    }

    pub fn add_named_sphere(&mut self, _sphere: Sphere, _material_name: String) {
        // for object instance approach
    }

    pub fn add_sphere(&mut self, sphere: ShapeInstance<Sphere>) -> u32 {
        let data = Arc::get_mut(&mut self.scene_data).expect("Scene data cannot be aquired!");
        data.add_sphere(sphere)
    }

    pub fn add_transformd_sphere(&mut self, sphere: ShapeInstance<TransformShape<Sphere>>) -> u32 {
        let data = Arc::get_mut(&mut self.scene_data).expect("Scene data cannot be aquired!");
        data.add_transformed_sphere(sphere)
    }

    pub fn add_mesh(&mut self, mesh: ShapeInstance<Mesh>) -> u32 {
        let data = Arc::get_mut(&mut self.scene_data).expect("Scene data cannot be aquired!");
        data.add_mesh(mesh)
    }

    pub fn add_transformed_mesh(&mut self, mesh: ShapeInstance<TransformShape<Mesh>>) -> u32 {
        let data = Arc::get_mut(&mut self.scene_data).expect("Scene data cannot be aquired!");
        data.add_transformed_mesh(mesh)
    }

    pub fn add_material(&mut self, mat: Material) -> u32 {
        let data = Arc::get_mut(&mut self.scene_data).expect("Scene data cannot be aquired!");
        data.add_material(mat)
    }

    pub fn add_light(&mut self, light: Light) -> u32 {
        let data = Arc::get_mut(&mut self.scene_data).expect("Scene data cannot be aquired!");
        data.add_light(light)
    }

    pub fn set_area_light(&mut self, shape_type: &ShapeType, shape_id: u32, light_id: i32) {
        let data = Arc::get_mut(&mut self.scene_data).expect("Scene data cannot be aquired!");
        data.set_area_light(shape_type, shape_id, light_id);
    }

    pub fn prepare(&mut self) {
        let camera = Arc::get_mut(&mut self.camera).expect("Camera cannot be aquired!");
        camera.set_resolution(self.options.xres, self.options.yres);
        camera.prepare_camera();
        self.color_buffer = ColorBuffer::new(self.options.xres as usize, self.options.yres as usize);
        let scene_data = Arc::get_mut(&mut self.scene_data).expect("Scene data cannot be aquired!");
        scene_data.prepare();
    }
}
