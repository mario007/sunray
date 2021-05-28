
use crate::vec::{u32x3, f32x2, f32x3, f64x3};
use crate::matrix;
use crate::matrix::Matrix4x4;
use crate::ray::Ray;
use crate::isects;
use crate::isects::{CollectionIntersect, NPrimitives, BBoxPrimitive, PrimitiveIntersect,
    LinearIntersect, ShapeIntersect, CalculateNormal, Precompute};
use crate::isects::{AABB, BBox};
use crate::grid::{UniGrid, UniGridBuild};
use crate::grid;
use crate::math;
use crate::ply_reader::PlyModel;
use crate::sampling;


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
    fn calculate_normal(&self, hitpoint: f32x3, _sub_shape: usize) -> f32x3 {
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

    pub fn get_normals(&self, index: usize) -> (f32x3, f32x3, f32x3) {
        let indices = self.indices[index];
        let n1 = self.vertex_normals[indices.0 as usize];
        let n2 = self.vertex_normals[indices.1 as usize];
        let n3 = self.vertex_normals[indices.2 as usize];
        (n1, n2, n3)
    }

    pub fn generate_position(&self, u1: f32, u2: f32, u3: f32) -> f32x3 {
        let ntriangles = self.indices.len();
        let triangle_idx = u1 * ntriangles as f32;
        let triangle_idx = (triangle_idx as usize).min(ntriangles - 1);
        let (u, v, w) = sampling::uniform_sampling_triangle(u2, u3);
        let (v1, v2, v3) = self.get_vertices(triangle_idx);
        u * v1 + v * v2 + w * v3
    }
}

pub fn triangle_pdf(v1: f32x3, v2: f32x3, v3: f32x3) -> f32 {
    let v = (v2 - v1).cross(v3 - v1);
    let length = (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt();
    let area = 0.5 * length;
    1.0 / area
}

pub fn triangle_pos_and_pdf(v1: f32x3, v2: f32x3, v3: f32x3, u1: f32, u2: f32) -> (f32x3, f32) {
    let (u, v, w) = sampling::uniform_sampling_triangle(u1, u2);
    let point = u * v1 + v * v2 + w * v3;
    (point, triangle_pdf(v1, v2, v3))
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
    fn calculate_normal(&self, hitpoint: f32x3, sub_shape: usize) -> f32x3 {
        let (v1, v2, v3) = self.get_vertices(sub_shape);
        if self.vertex_normals.len() > 0 {
            let (u, v, w) = math::barycentric(hitpoint, v1, v2, v3);
            let (n1, n2, n3) = self.get_normals(sub_shape);
            return (u * n1 + v * n2 + w * n3).normalize();
        } else { 
            return (v2 - v1).cross(v3 - v1).normalize();
        }
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
        let (v0, v1, v2) = self.get_vertices(index);
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
    pub shape: T,
    pub obj_to_world: Matrix4x4,
    pub world_to_obj: Matrix4x4,
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
        if t > 0.0 && t < local_min_dist {
            let local_hit = origin + t * dir;
            let world_hit = self.obj_to_world.transform_point(local_hit);
            let world_t = math::distance(world_hit, ray.origin);
            
            return (world_t, sub_shape);
        }
        (0.0, -1)
    }
}

impl<T: CalculateNormal> CalculateNormal for TransformShape<T> {
    fn calculate_normal(&self, hitpoint: f32x3, sub_shape: usize) -> f32x3 {
        let local_p = self.world_to_obj.transform_point(hitpoint);
        let normal = self.shape.calculate_normal(local_p, sub_shape);
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

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ShapeType {
    Sphere,
    TransformSphere,
    Mesh,
    TransformMesh,
    None,
}

pub struct IsectPoint {
    pub position: f32x3,
    pub normal: f32x3,
    pub t: f32,
    pub material_id: u32,
    pub shape_type: ShapeType,
    pub shape_id: usize,
    pub sub_shape: i32,
    pub light_id: i32,
}

impl IsectPoint {
    pub fn new(position: f32x3, normal: f32x3, t: f32, material_id: u32, shape_type: ShapeType, shape_id: usize, sub_shape: i32, light_id: i32) -> Self {
        Self {position, normal, t, material_id, shape_type, shape_id, sub_shape, light_id}
    }
}

pub struct Shapes<T> {
    pub shapes: Vec<ShapeInstance<T>>,
    shapes_bbox: Vec<AABB>,
    grid: Option<UniGrid>,
    bbox: Option<AABB>,
    area_lights_mapping: Vec<i32>,
}

impl<T> Shapes<T> {
    pub fn new() -> Self {
        Self {
            shapes: Vec::new(),
            shapes_bbox: Vec::new(),
            grid: None,
            bbox: None,
            area_lights_mapping: Vec::new(),
        }
    }

    pub fn add_shape(&mut self, shape: ShapeInstance<T>) -> u32 where T: BBox {
        let id = self.shapes.len();
        self.bbox = match self.bbox {
            Some(bbox) => Some(bbox.update(&shape.shape.bbox())),
            None => Some(shape.shape.bbox())
        };
        self.shapes_bbox.push(shape.shape.bbox());
        self.shapes.push(shape);
        self.area_lights_mapping.push(-1);
        id as u32
    }

    pub fn generate_isect(&self, index: usize, ray: &Ray, min_dist: f32, shape_type: ShapeType, sub_shape: i32) -> IsectPoint where T: CalculateNormal {
        let hitpoint = ray.origin + min_dist * ray.direction;
        let mut normal = self.shapes[index].shape.calculate_normal(hitpoint, sub_shape as usize);
        if normal.dot(ray.direction * -1.0) < 0.0 {
            normal = normal * -1.0;
        }
        let material_id = *&self.shapes[index].material_id;
        let light_id = self.area_lights_mapping[index];
        let isect_point = IsectPoint::new(hitpoint, normal, min_dist, material_id, shape_type, index, sub_shape, light_id);
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

    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    pub fn set_area_light(&mut self, shape_id: u32, light_id: i32) {
        self.area_lights_mapping[shape_id as usize] = light_id;
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
