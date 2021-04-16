
use crate::vec::{u32x3, f32x2, f32x3, f64x3};
use crate::matrix;
use crate::matrix::Matrix4x4;
use crate::ray::Ray;
use crate::isects;
use crate::isects::{CollectionIntersect, NPrimitives, BBoxPrimitive, PrimitiveIntersect,
    LinearIntersect, ShapeIntersect, IsectPoint, CalculateNormal, Precompute};
use crate::isects::{AABB, BBox};
use crate::grid::{UniGrid, UniGridBuild};
use crate::grid;
use crate::math;
use crate::ply_reader::PlyModel;


pub struct Sphere {
    pub position: f32x3,
    pub radius: f32,
}

impl Sphere {
    pub fn new(position: f32x3, radius: f32) -> Self {
        Self{position, radius}
    }
}

impl BBox for Sphere {
    fn bbox(&self) -> AABB {
        let r = self.radius;
        let min_p = self.position - f32x3(r, r, r);
        let max_p = self.position + f32x3(r, r, r);
        AABB::new(min_p, max_p)
    }
}

impl ShapeIntersect for Sphere {
    fn intersect(&self, ray: &Ray, min_dist: f32) -> (f32, i32) {
        let pos = f64x3::from(self.position);
        let radius = self.radius as f64;
        let origin = f64x3::from(ray.origin);
        let dir = f64x3::from(ray.direction);
        (isects::isect_sphere(origin, dir, pos, radius , min_dist as f64) as f32, -1)
    }
}

impl CalculateNormal for Sphere {
    fn calculate_normal(&self, hitpoint: f32x3, _ray: &Ray, _sub_shape: i32) -> f32x3 {
        (hitpoint - self.position).normalize()
    }
}

impl Precompute for Sphere {}

pub struct Mesh {
    vertices: Vec<f32x3>,
    indices: Vec<u32x3>,
    vertex_normals: Vec<f32x3>,
    uv_coords: Vec<f32x2>,
    grid: Option<UniGrid>,
    bbox: Option<AABB>,
}

impl Mesh {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_normals: Vec::new(),
            uv_coords: Vec::new(),
            grid: None,
            bbox: None,
        }
    }

    pub fn add_vertex(&mut self, x: f32, y: f32, z: f32) {
        let eps = 0.00001f32;
        self.bbox = match self.bbox {
            Some(bbox) => Some(bbox.update(&AABB::new(f32x3(x-eps, y-eps, z-eps), f32x3(x+eps, y+eps, z+eps)))),
            None => Some(AABB::new(f32x3(x-eps, y-eps, z-eps), f32x3(x+eps, y+eps, z+eps)))
        };
        self.vertices.push(f32x3(x, y, z));
    }

    pub fn add_vertex_normal(&mut self, x: f32, y: f32, z: f32) {
        self.vertex_normals.push(f32x3(x, y, z));
    }

    pub fn add_uv(&mut self, u: f32, v: f32) {
        self.uv_coords.push(f32x2(u, v));
    }

    pub fn add_indices(&mut self, i: u32, j: u32, k: u32) {
        self.indices.push(u32x3(i, j, k));
    }

    pub fn get_vertices(&self, index: usize) -> (f32x3, f32x3, f32x3) {
        let indices = self.indices[index];
        let v1 = self.vertices[indices.0 as usize];
        let v2 = self.vertices[indices.1 as usize];
        let v3 = self.vertices[indices.2 as usize];
        (v1, v2, v3)
    }

    pub fn normal(&self, index: usize) -> f32x3 {
        // TODO ih we have vertex normals use them
        let (v1, v2, v3) = self.get_vertices(index);
        (v2 - v1).cross(v3 - v1).normalize()
    }
}

impl Precompute for Mesh {
    fn precompute(&mut self) {
        if self.indices.len() > 2 {
            self.grid = Some(grid::create_uni_grid(self));
        }
    }
}

impl BBox for Mesh {
    fn bbox(&self) -> AABB {
        match self.bbox {
            Some(bbox) => bbox,
            None => AABB::new(f32x3(0.0, 0.0, 0.0), f32x3(0.0, 0.0, 0.0))
        }
    }
}

impl BBoxPrimitive for Mesh {
    fn bbox_primitive(&self, index: usize) -> AABB {
        let indices = self.indices[index];
        let eps = f32x3(0.00001, 0.00001, 0.00001);
        let mut min_p = self.vertices[indices.0 as usize] - eps;
        let mut max_p = self.vertices[indices.0 as usize] + eps;
        min_p = min_p.min(self.vertices[indices.1 as usize] - eps);
        max_p = max_p.max(self.vertices[indices.1 as usize] + eps);
        min_p = min_p.min(self.vertices[indices.2 as usize] - eps);
        max_p = max_p.max(self.vertices[indices.2 as usize] + eps);
        AABB::new(min_p, max_p)
    }
}

impl ShapeIntersect for Mesh {
    fn intersect(&self, ray: &Ray, min_dist: f32) -> (f32, i32) {
        let (idx, t, _) = self.collection_intersect(ray, min_dist);
        (t, idx)
    }
}

impl CalculateNormal for Mesh {
    fn calculate_normal(&self, _hitpoint: f32x3, _ray: &Ray, sub_shape: i32) -> f32x3 {
        // TODO ih we have vertex normals use them
        let (v1, v2, v3) = self.get_vertices(sub_shape as usize);
        (v2 - v1).cross(v3 - v1).normalize()
    }
}

impl NPrimitives for Mesh {
    fn nprimitives(&self) -> usize {
        self.indices.len()
    }
}

impl UniGridBuild for Mesh {}

impl PrimitiveIntersect for Mesh {
    fn primitive_intersect(&self, index: usize, ray: &Ray, min_dist: f32) -> (f32, i32) {
        let (v0, v1, v2) = self.get_vertices(index as usize);
        let t = isects::ray_triangle_isect(f64x3::from(v0), f64x3::from(v1), f64x3::from(v2),
                                           f64x3::from(ray.origin), f64x3::from(ray.direction));
        if t > min_dist as f64 {
            return (0.0, -1);
        }
        return (t as f32, -1)
    }
}

impl LinearIntersect for Mesh {}

impl CollectionIntersect for Mesh {
    fn has_acceleration_structure(&self) -> bool {
        self.grid.is_some()
    }
    fn accel_intersect(&self, ray: &Ray, min_dist: f32) -> (i32, f32, i32) {
        match &self.grid {
            Some(grid) => grid::intersect(grid, self, ray, min_dist),
            None => (-1, 0.0, -1)
        }
    }
}

impl PlyModel for Mesh {
    fn set_number_of_vertices(&mut self, n: usize) {
        self.vertices.reserve_exact(n);
    }

    fn set_number_of_faces(&mut self, n: usize) {
        self.indices.reserve_exact(n);
    }

    fn add_vertex(&mut self, v0: f32, v1: f32, v2: f32) {
        self.add_vertex(v0, v1, v2);
    }

    fn add_normal(&mut self, n0: f32, n1: f32, n2: f32) {
        self.add_vertex_normal(n0, n1, n2);
    }

	fn add_uv_coord(&mut self, u: f32, v: f32) {
        self.add_uv(u, v);
    }
	
	fn add_face(&mut self, f0: u32, f1: u32, f2: u32) {
        self.add_indices(f0, f1, f2);
    }
}

pub struct TransformShape<T> {
    shape: T,
    obj_to_world: Matrix4x4,
    world_to_obj: Matrix4x4,
}

impl<T> TransformShape<T> {

    pub fn new(shape: T, obj_to_world: Matrix4x4) -> Self {
        Self{shape, obj_to_world, world_to_obj: obj_to_world.inverse()}
    }
}

impl<T: Precompute> Precompute for TransformShape<T> {
    fn precompute(&mut self) {
        self.shape.precompute()
    }
}

impl<T: BBox> BBox for TransformShape<T> {
    fn bbox(&self) -> AABB {
        let bb = self.shape.bbox();
        let delta = bb.max_p - bb.min_p;
        let p1 = self.obj_to_world.transform_point(bb.min_p);
        let p2 = self.obj_to_world.transform_point(bb.max_p);
        let p3 = self.obj_to_world.transform_point(bb.min_p + f32x3(delta.0, 0.0, 0.0));
        let p4 = self.obj_to_world.transform_point(bb.min_p + f32x3(0.0, delta.1, 0.0));
        let p5 = self.obj_to_world.transform_point(bb.min_p + f32x3(delta.0, delta.1, 0.0));
        let p6 = self.obj_to_world.transform_point(bb.max_p + f32x3(delta.0, 0.0, 0.0));
        let p7 = self.obj_to_world.transform_point(bb.max_p + f32x3(0.0, delta.1, 0.0));
        let p8 = self.obj_to_world.transform_point(bb.max_p + f32x3(delta.0, delta.1, 0.0));
        let min_p = p1.min(p2).min(p3).min(p4).min(p5).min(p6).min(p7).min(p8);
        let max_p = p1.max(p2).max(p3).max(p4).max(p5).max(p6).max(p7).max(p8);
        AABB::new(min_p, max_p)
    }
}

impl<T: ShapeIntersect> ShapeIntersect for TransformShape<T> {
    fn intersect(&self, ray: &Ray, min_dist: f32) -> (f32, i32) {
        let world_p = ray.origin + min_dist * ray.direction;
        let local_p = self.world_to_obj.transform_point(world_p);
        let origin = self.world_to_obj.transform_point(ray.origin);
        let dir = self.world_to_obj.transform_vector(ray.direction).normalize();
        let local_min_dist = math::distance(local_p, origin);

        let local_ray = Ray::new(origin, dir);
        let (t, sub_shape) = self.shape.intersect(&local_ray, local_min_dist);

        if t > 0.0 {
            let local_hit = origin + t * dir;
            let world_hit = self.obj_to_world.transform_point(local_hit);
            let world_t = math::distance(world_hit, ray.origin);
            return (world_t, sub_shape);
        }
        (0.0, -1)
    }
}

impl<T: CalculateNormal> CalculateNormal for TransformShape<T> {
    fn calculate_normal(&self, hitpoint: f32x3, ray: &Ray, sub_shape: i32) -> f32x3 {
        let local_p = self.world_to_obj.transform_point(hitpoint);
        let origin = self.world_to_obj.transform_point(ray.origin);
        let dir = self.world_to_obj.transform_vector(ray.direction).normalize();
        let local_ray = Ray::new(origin, dir);
        let normal = self.shape.calculate_normal(local_p, &local_ray, sub_shape);
        matrix::transform_normal(&self.world_to_obj, normal).normalize()
    }
}


pub struct ShapeInstance<T> {
    pub shape: T,
    pub material_id: u32,
}

impl<T> ShapeInstance<T> {
    pub fn new(shape: T, material_id: u32) -> Self {
        Self{shape, material_id}
    }
}

pub struct Shapes<T> {
    shapes: Vec<ShapeInstance<T>>,
    shapes_bbox: Vec<AABB>,
    grid: Option<UniGrid>,
    bbox: Option<AABB>,
}

impl<T> Shapes<T> {
    pub fn new() -> Self {
        Self {
            shapes: Vec::new(),
            shapes_bbox: Vec::new(),
            grid: None,
            bbox: None,
        }
    }

    pub fn add_shape(&mut self, shape: ShapeInstance<T>) where T: BBox {

        self.bbox = match self.bbox {
            Some(bbox) => Some(bbox.update(&shape.shape.bbox())),
            None => Some(shape.shape.bbox())
        };
        self.shapes_bbox.push(shape.shape.bbox());
        self.shapes.push(shape);
    }

    pub fn generate_isect(&self, index: usize, ray: &Ray, min_dist: f32, sub_shape: i32) -> IsectPoint where T: CalculateNormal {
        let hitpoint = ray.origin + min_dist * ray.direction;
        let normal = self.shapes[index].shape.calculate_normal(hitpoint, ray, sub_shape);
        let material_id = *&self.shapes[index].material_id;
        let isect_point = IsectPoint::new(hitpoint, normal, min_dist, material_id);
        return isect_point;
    }

    pub fn material(&self, index: usize) -> u32 {
        *&self.shapes[index].material_id
    }

    pub fn precompute(&mut self) where T: Precompute {
        for shape in self.shapes.iter_mut() {
            shape.shape.precompute();
        }

        if self.shapes.len() > 4 {
            self.grid = Some(grid::create_uni_grid(self));
        }
    }
}

impl<T> BBox for Shapes<T> {
    fn bbox(&self) -> AABB {
        match self.bbox {
            Some(bbox) => bbox,
            None => AABB::new(f32x3(0.0, 0.0, 0.0), f32x3(0.0, 0.0, 0.0))
        }
    }
}

impl<T> BBoxPrimitive for Shapes<T> {
    fn bbox_primitive(&self, index: usize) -> AABB {
        self.shapes_bbox[index]
    }
}

impl<T> NPrimitives for Shapes<T> {
    fn nprimitives(&self) -> usize {
        self.shapes.len()
    }
}

impl<T: ShapeIntersect> PrimitiveIntersect for Shapes<T> {
    fn primitive_intersect(&self, index: usize, ray: &Ray, min_dist: f32) -> (f32, i32) {
        let shape_inst = &self.shapes[index];
        shape_inst.shape.intersect(ray, min_dist)
    }
}

impl<T: ShapeIntersect> LinearIntersect for Shapes<T> {}

impl<T> UniGridBuild for Shapes<T> {}

impl<T: ShapeIntersect> CollectionIntersect for Shapes<T> {
    fn has_acceleration_structure(&self) -> bool {
        self.grid.is_some()
    }
    fn accel_intersect(&self, ray: &Ray, min_dist: f32) -> (i32, f32, i32) {
        match &self.grid {
            Some(grid) => grid::intersect(grid, self, ray, min_dist),
            None => (-1, 0.0, -1)
        }
    }
}
