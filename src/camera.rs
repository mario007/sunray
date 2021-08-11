use crate::vec::{f32x2, f32x3};
use crate::matrix::Matrix4x4;
use crate::ray::Ray;


#[derive(Debug, Copy, Clone, PartialEq)]
pub struct BaseCamera {
    xres: u32,
    yres: u32,
    camera_to_world: Matrix4x4,

    aspect_ratio: Option<f32>,
    screen_pmin: Option<f32x2>,
    screen_pmax: Option<f32x2>,
   
}

impl BaseCamera {
    pub fn new(xres: u32, yres: u32) -> BaseCamera {
        BaseCamera {
            xres,
            yres,
            camera_to_world: Matrix4x4::identity(),

            aspect_ratio: None,
            screen_pmin: None,
            screen_pmax: None,
        }
    }

    pub fn set_camera_to_world(&mut self, matrix: Matrix4x4) {
        self.camera_to_world = matrix;
    }

    pub fn set_resolution(&mut self, xres: u32, yres: u32) {
        self.xres = xres;
        self.yres = yres;
    }

    pub fn set_aspect_ratio(&mut self, aspect_ratio: Option<f32>) {
        self.aspect_ratio = aspect_ratio;
    }
    
    fn compute_aspect_ratio(&self) -> f32 {
        match self.aspect_ratio {
            Some(aspect) => aspect,
            None => self.xres as f32 / self.yres as f32,
        }
    }

    fn compute_screen_bounds(&self, frame_aspect_ratio: f32) -> (f32x2, f32x2) {

        let mut pmin: f32x2;
        let mut pmax: f32x2;
        if frame_aspect_ratio > 1.0 {
            pmin = f32x2(-frame_aspect_ratio, -1.0);
            pmax = f32x2(frame_aspect_ratio, 1.0);
        } else {
            pmin = f32x2(-1.0, -1.0 / frame_aspect_ratio);
            pmax = f32x2(1.0, 1.0 / frame_aspect_ratio);
        }

        pmin = match self.screen_pmin {
            Some(screen_pmin) => screen_pmin,
            None => pmin,
        };

        pmax = match self.screen_pmax {
            Some(screen_pmax) => screen_pmax,
            None => pmax,
        };
        (pmin, pmax)
    }

    fn compute_screen_to_raster(&self) -> Matrix4x4 {
        let aspect_ratio = self.compute_aspect_ratio();
        let (pmin, pmax) = self.compute_screen_bounds(aspect_ratio);
        Matrix4x4::scale(self.xres as f32, self.yres as f32, 1.0) *
        Matrix4x4::scale(1.0 / (pmax.0 - pmin.0), 1.0 / (pmin.1 - pmax.1), 1.0) *
        Matrix4x4::translate(-pmin.0, -pmax.1, 0.0)
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PerspectiveCamera {  
    base: BaseCamera,
    fov: f32,
    raster_to_camera: Matrix4x4,
}

impl PerspectiveCamera {
    pub fn new(xres: u32, yres: u32, fov: f32) -> PerspectiveCamera {
        PerspectiveCamera {
            base: BaseCamera::new(xres, yres),
            fov,
            raster_to_camera: Matrix4x4::identity(),
        }
    }

    pub fn set_fov(&mut self, fov: f32) {
        self.fov = fov;
    }

    pub fn prepare(&mut self) {
        let camera_to_screen = Matrix4x4::perspective(self.fov, 1e-2, 1000.0);
        let screen_to_raster = self.base.compute_screen_to_raster();
        self.raster_to_camera = camera_to_screen.inverse() * screen_to_raster.inverse();
    }

    pub fn generate_ray(&self, x: f32, y: f32) -> Ray {
        // TODO depth of field
        let camera_dir = self.raster_to_camera.transform_point(f32x3(x, y, 0.0)).normalize();
        let camera_pos = f32x3::from(0.0);

        let dir = self.base.camera_to_world.transform_vector(camera_dir).normalize();
        let pos = self.base.camera_to_world.transform_point(camera_pos);
        Ray::new(pos, dir)
    }
}

pub enum Camera {
    Perspective(PerspectiveCamera)
}

impl Camera {

    pub fn set_camera_to_world(&mut self, matrix: Matrix4x4) {
        match self {
            Camera::Perspective(persp_data) => persp_data.base.set_camera_to_world(matrix),
        }
    }

    pub fn set_resolution(&mut self, xres: u32, yres: u32) {
        match self {
            Camera::Perspective(persp_data) => persp_data.base.set_resolution(xres, yres),
        }
    }

    pub fn set_aspect_ratio(&mut self, aspect_ratio: Option<f32>) {
        match self {
            Camera::Perspective(persp_data) => persp_data.base.set_aspect_ratio(aspect_ratio),
        }
    }

    pub fn prepare_camera(&mut self) {
        match self {
            Camera::Perspective(persp_data) => persp_data.prepare(),
        }
    }
    
    pub fn generate_ray(&self, sample_x: f32, sample_y: f32) -> Ray {
        match self {
            Camera::Perspective(persp_data) => persp_data.generate_ray(sample_x, sample_y),
        }
    }
}
