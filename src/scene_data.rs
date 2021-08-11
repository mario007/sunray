use crate::vec::f32x3;
use crate::shapes::{Sphere, Shapes, ShapeInstance, TransformShape, Mesh, ShapeType, IsectPoint, triangle_pos_and_pdf, triangle_pdf};
use crate::materials::{Material, MaterialSample};
use crate::isects::{CollectionIntersect, NPrimitives, CalculateNormal, AABB, BBox};
use crate::ray::Ray;
use crate::lights::{Light, LightSample};
use crate::math;
use crate::sampler::PathSampler;


pub struct ShapeSample {
    pub position: f32x3,
    pub wi: f32x3,
    pub pdfa: f32,
    pub cos_theta: f32,
    pub valid: bool,
}

impl ShapeSample {
    pub fn new(position: f32x3, wi: f32x3, pdfa: f32, cos_theta: f32, valid: bool) -> Self {
        Self {position, wi, pdfa, cos_theta, valid}
    }
}

pub struct SceneData {

    pub materials: Vec<Material>,
    pub lights: Vec<Light>,

    pub spheres: Shapes<Sphere>,
    pub tran_spheres: Shapes<TransformShape<Sphere>>,

    pub meshes: Shapes<Mesh>,
    pub tran_meshes: Shapes<TransformShape<Mesh>>,
} 


impl SceneData {
    pub fn new() -> Self {
        Self {
            materials: Vec::new(),
            lights: Vec::new(),

            spheres: Shapes::new(),
            tran_spheres: Shapes::new(),

            meshes: Shapes::new(),
            tran_meshes: Shapes::new(),
        }
    }

    pub fn add_material(&mut self, mat: Material) -> u32 {
        let id = self.materials.len();
        self.materials.push(mat);
        id as u32
    }

    pub fn material(&self, material_id: u32) -> &Material {
        &self.materials[material_id as usize]
    }

    pub fn add_light(&mut self, light: Light) -> u32 {
        let id = self.lights.len();
        self.lights.push(light);
        id as u32
    }

    pub fn add_sphere(&mut self, sphere: ShapeInstance<Sphere>) -> u32 {
        self.spheres.add_shape(sphere)
    }

    pub fn add_transformed_sphere(&mut self, sphere: ShapeInstance<TransformShape<Sphere>>) -> u32 {
        self.tran_spheres.add_shape(sphere)
    }

    pub fn add_mesh(&mut self, mesh: ShapeInstance<Mesh>)-> u32 {
        self.meshes.add_shape(mesh)
    }

    pub fn add_transformed_mesh(&mut self, mesh: ShapeInstance<TransformShape<Mesh>>) -> u32 {
        self.tran_meshes.add_shape(mesh)
    }

    pub fn prepare(&mut self) {
        self.spheres.precompute();
        self.tran_spheres.precompute();
        self.meshes.precompute();
        self.tran_meshes.precompute();

        let mut world_bound = AABB::new(f32x3::from(1e30), f32x3::from(-1e30));
        if self.spheres.len() > 0 {
            world_bound = world_bound.update(&self.spheres.bbox());
        }
        if self.tran_spheres.len() > 0 {
            world_bound = world_bound.update(&self.tran_spheres.bbox());
        }
        if self.meshes.len() > 0 {
            world_bound = world_bound.update(&self.meshes.bbox());
        }
        if self.tran_meshes.len() > 0 {
            world_bound = world_bound.update(&self.tran_meshes.bbox());
        }
        for light in &mut self.lights {
           light.precompute(&world_bound)
        }
    }

    pub fn intersect_t(&self, ray: &Ray, min_dist: f32) -> (i32, i32, f32, ShapeType) {
        let mut cur_min_dist = min_dist;
        let mut shape_idx = -1;
        let mut triangle_idx = -1;
        let mut shape_type = ShapeType::None;

        if self.spheres.nprimitives() > 0 {
            let (idx, t, _) = self.spheres.collection_intersect(ray, cur_min_dist);
            if idx != -1 && t < cur_min_dist {
                shape_type = ShapeType::Sphere;
                cur_min_dist = t;
                shape_idx = idx;
            }
        }
        if self.tran_spheres.nprimitives() > 0 {
            let (idx, t, _) = self.tran_spheres.collection_intersect(ray, cur_min_dist);
            if idx != -1 && t < cur_min_dist {
                shape_type = ShapeType::TransformSphere;
                cur_min_dist = t;
                shape_idx = idx;
            }
        }
        if self.meshes.nprimitives() > 0 {
            let (idx, t, triangle) = self.meshes.collection_intersect(ray, cur_min_dist);
            if idx != -1 && t < cur_min_dist {
                shape_type = ShapeType::Mesh;
                cur_min_dist = t;
                shape_idx = idx;
                triangle_idx = triangle;
            }
        }
        if self.tran_meshes.nprimitives() > 0 {
            let (idx, t, triangle) = self.tran_meshes.collection_intersect(ray, cur_min_dist);
            if idx != -1 && t < cur_min_dist {
                shape_type = ShapeType::TransformMesh;
                cur_min_dist = t;
                shape_idx = idx;
                triangle_idx = triangle;
            }
        }
        (shape_idx, triangle_idx, cur_min_dist, shape_type)
    }

    pub fn intersect(&self, ray: &Ray) -> Option<IsectPoint> {
        let (shape_idx, triangle_idx, min_dist, shape_type) = self.intersect_t(ray, 1e30f32);
        if shape_idx == -1 {
            return None
        }

        match shape_type {
            ShapeType::Sphere => {
                Some(self.spheres.generate_isect(shape_idx as usize, ray, min_dist, shape_type, -1))
            }
            ShapeType::TransformSphere => {
                Some(self.tran_spheres.generate_isect(shape_idx as usize, ray, min_dist, shape_type, -1))
            }
            ShapeType::Mesh => {
                Some(self.meshes.generate_isect(shape_idx as usize, ray, min_dist, shape_type, triangle_idx))
            }
            ShapeType::TransformMesh => {
                Some(self.tran_meshes.generate_isect(shape_idx as usize, ray, min_dist, shape_type, triangle_idx))
            }
            ShapeType::None => None
        }
    }

    pub fn visible(&self, point1: f32x3, point2: f32x3) -> bool {
        let eps = 0.00001;
        let dist = math::distance(point1, point2) - eps;
        let direction = (point2 - point1).normalize();
        let ray = Ray::new(point1, direction);
        // TODO: early return
        let (shape_idx, _triangle_idx, _min_dist, _shape_type) = self.intersect_t(&ray, dist);
        shape_idx == -1
    }

    pub fn light_sample_li(&self, light_id: usize, isect: &IsectPoint, path_sampler: &mut PathSampler) -> LightSample{
        let light = &self.lights[light_id];
        match light {
            Light::Point(point_light) => point_light.sample_li(isect.position),
            Light::Distant(distant) => distant.sample_li(isect.position),
            Light::Area(area_light) => {
                let ss = self.sample_shape(isect.position, &area_light.shape_type, area_light.shape_id, path_sampler);
                LightSample::new(area_light.intensity, ss.position, ss.wi, ss.pdfa, ss.cos_theta, false, ss.valid)
            }
        }
    }

    pub fn light_get_le(&self, isect: &IsectPoint, direction_to_light: f32x3) -> LightSample {
        let light = &self.lights[isect.light_id as usize];
        match light {
            Light::Point(_) => LightSample::new(f32x3(0.0, 0.0, 0.0), isect.position, direction_to_light, 0.0, 1.0, true, false),
            Light::Distant(_) => LightSample::new(f32x3(0.0, 0.0, 0.0), isect.position, direction_to_light, 0.0, 1.0, true, false),
            Light::Area(area_light) => {
                let cos_theta = isect.normal.dot(-direction_to_light);
                let mut valid = true;
                if cos_theta < 0.0 { valid = false; }
                let pdfa = self.get_shape_pdf(&isect.shape_type, isect.shape_id, isect.sub_shape);
                LightSample::new(area_light.intensity, isect.position, direction_to_light, pdfa, cos_theta, false, valid)
            }
        }
    }

    pub fn sample_shape(&self, hitpoint: f32x3, shape_type: &ShapeType, shape_id: u32, path_sampler: &mut PathSampler) -> ShapeSample {
        match shape_type {
            ShapeType::Sphere => {
                ShapeSample::new(f32x3(0.0, 0.0, 0.0), f32x3(0.0, 0.0, 0.0), 0.0, 0.0, false)
            }
            ShapeType::TransformSphere => {
                ShapeSample::new(f32x3(0.0, 0.0, 0.0), f32x3(0.0, 0.0, 0.0), 0.0, 0.0, false)
            }
            ShapeType::Mesh => {
                let mesh = &self.meshes.shapes[shape_id as usize].shape;
                let triangle_idx = self.calc_index(mesh.nprimitives(), path_sampler);
                let (v1, v2, v3) = mesh.get_vertices(triangle_idx);
                let (position, pdf) = triangle_pos_and_pdf(v1, v2, v3, path_sampler.next_1d(), path_sampler.next_1d());
                let wi = (position - hitpoint).normalize();
                let normal = mesh.calculate_normal(position, triangle_idx);
                let cos_theta = normal.dot(-wi);
                let mut valid = true;
                if cos_theta < 0.0 { valid = false; }
                let triangle_picking_pdf = 1.0 / mesh.nprimitives() as f32;
                ShapeSample::new(position, wi, pdf * triangle_picking_pdf, cos_theta, valid)
            }
            ShapeType::TransformMesh => {
                let tran_mesh = &self.tran_meshes.shapes[shape_id as usize].shape;
                let triangle_idx = self.calc_index(tran_mesh.shape.nprimitives(), path_sampler);
                let (v1, v2, v3) = tran_mesh.shape.get_vertices(triangle_idx);
                let v1 = tran_mesh.obj_to_world.transform_point(v1);
                let v2 = tran_mesh.obj_to_world.transform_point(v2);
                let v3 = tran_mesh.obj_to_world.transform_point(v3);
                let (position, pdf) = triangle_pos_and_pdf(v1, v2, v3, path_sampler.next_1d(), path_sampler.next_1d());
                let wi = (position - hitpoint).normalize();
                let normal = tran_mesh.calculate_normal(position, triangle_idx);
                let cos_theta = normal.dot(-wi);
                let mut valid = true;
                if cos_theta < 0.0 { valid = false; }
                let triangle_picking_pdf = 1.0 / tran_mesh.shape.nprimitives() as f32;
                ShapeSample::new(position, wi, pdf * triangle_picking_pdf, cos_theta, valid)
            }
            ShapeType::None => ShapeSample::new(f32x3(0.0, 0.0, 0.0), f32x3(0.0, 0.0, 0.0), 0.0, 0.0, false)
        }
    }

    fn get_shape_pdf(&self, shape_type: &ShapeType, shape_id: usize, sub_shape: i32) -> f32 {
        match shape_type {
            ShapeType::Sphere => {
                0.0
            }
            ShapeType::TransformSphere => {
                0.0
            }
            ShapeType::Mesh => {
                let mesh = &self.meshes.shapes[shape_id].shape;
                let (v1, v2, v3) = mesh.get_vertices(sub_shape as usize);
                let triangle_picking_pdf = 1.0 / mesh.nprimitives() as f32;
                triangle_pdf(v1, v2, v3) * triangle_picking_pdf
            }
            ShapeType::TransformMesh => {
                let tran_mesh = &self.tran_meshes.shapes[shape_id].shape;
                let (v1, v2, v3) = tran_mesh.shape.get_vertices(sub_shape as usize);
                let v1 = tran_mesh.obj_to_world.transform_point(v1);
                let v2 = tran_mesh.obj_to_world.transform_point(v2);
                let v3 = tran_mesh.obj_to_world.transform_point(v3);
                let triangle_picking_pdf = 1.0 / tran_mesh.shape.nprimitives() as f32;
                triangle_pdf(v1, v2, v3) * triangle_picking_pdf
            }
            ShapeType::None => 0.0
        }
    }

    fn calc_index(&self, n: usize, path_sampler: &mut PathSampler) -> usize {
        let index = path_sampler.next_1d() * n as f32;
        (index as usize).min(n - 1)
    }

    pub fn lights_sample_li(&self, isect: &IsectPoint, path_sampler: &mut PathSampler) -> LightSample {
        let nlights = self.lights.len();
        let light_picking_pdf = 1.0 / nlights as f32;
        let light_id = self.calc_index(nlights, path_sampler);
        let mut ls = self.light_sample_li(light_id, isect, path_sampler);
        ls.pdfa *= light_picking_pdf;
        ls
    }

    pub fn material_eval(&self, material_id: u32, wo: f32x3, normal: f32x3, wi: f32x3) -> (f32x3, f32) {
        let mat = &self.materials[material_id as usize];
        mat.eval(wo, normal, wi)
    }

    pub fn material_sample(&self, material_id: u32, wo: f32x3, normal: f32x3, path_sampler: &mut PathSampler) -> MaterialSample {
        let mat = &self.materials[material_id as usize];
        mat.sample(wo, normal, path_sampler)
    }

    pub fn set_area_light(&mut self, shape_type: &ShapeType, shape_id: u32, light_id: i32) {
        match shape_type {
            ShapeType::Sphere => {
                self.spheres.set_area_light(shape_id, light_id)
            }
            ShapeType::TransformSphere => {
                self.tran_spheres.set_area_light(shape_id, light_id)
            }
            ShapeType::Mesh => {
                self.meshes.set_area_light(shape_id, light_id)
            }
            ShapeType::TransformMesh => {
                self.tran_meshes.set_area_light(shape_id, light_id)
            }
            ShapeType::None => {}
        }
    }
}

impl Default for SceneData {
    fn default() -> Self {
        SceneData::new()
    }
}
