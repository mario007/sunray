use crate::vec::f32x3;
use crate::shapes::{Sphere, Shapes, ShapeInstance, TransformShape, Mesh};
use crate::materials::Material;
use crate::isects::{IsectPoint, CollectionIntersect, NPrimitives};
use crate::ray::Ray;
use crate::lights::Light;
use crate::math;


pub struct SceneData {

    pub materials: Vec<Material>,
    pub lights: Vec<Light>,

    pub spheres: Shapes<Sphere>,
    pub tran_spheres: Shapes<TransformShape<Sphere>>,

    pub meshes: Shapes<Mesh>,
    pub tran_meshes: Shapes<TransformShape<Mesh>>,
} 

pub enum ShapeType {
    Sphere,
    TransformSphere,
    Mesh,
    TransformMesh,
    None,
}

impl SceneData {
    pub fn new() -> Self {
        SceneData {
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

    pub fn add_sphere(&mut self, sphere: ShapeInstance<Sphere>) {
        self.spheres.add_shape(sphere);
    }

    pub fn add_transformed_sphere(&mut self, sphere: ShapeInstance<TransformShape<Sphere>>) {
        self.tran_spheres.add_shape(sphere);
    }

    pub fn add_mesh(&mut self, mesh: ShapeInstance<Mesh>) {
        self.meshes.add_shape(mesh);
    }

    pub fn add_transformed_mesh(&mut self, mesh: ShapeInstance<TransformShape<Mesh>>) {
        self.tran_meshes.add_shape(mesh);
    }

    pub fn prepare(&mut self) {
        self.spheres.precompute();
        self.tran_spheres.precompute();
        self.meshes.precompute();
        self.tran_meshes.precompute();
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
                return Some(self.spheres.generate_isect(shape_idx as usize, ray, min_dist, -1))
            }
            ShapeType::TransformSphere => {
                return Some(self.tran_spheres.generate_isect(shape_idx as usize, ray, min_dist, -1))
            }
            ShapeType::Mesh => {
                return Some(self.meshes.generate_isect(shape_idx as usize, ray, min_dist, triangle_idx))
            }
            ShapeType::TransformMesh => {
                return Some(self.tran_meshes.generate_isect(shape_idx as usize, ray, min_dist, triangle_idx))
            }
            ShapeType::None => return None
        }
    }

    pub fn visible(&self, point1: f32x3, point2: f32x3) -> bool {
        let eps = 0.00001;
        let dist = math::distance(point1, point2) - eps;
        let direction = (point2 - point1).normalize();
        let ray = Ray::new(point1, direction);
        let (shape_idx, _triangle_idx, _min_dist, _shape_type) = self.intersect_t(&ray, dist);
        shape_idx == -1
    }
}
