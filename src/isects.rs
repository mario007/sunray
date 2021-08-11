use crate::vec::{f32x3, f64x3};
use crate::ray::Ray;


#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AABB {
    pub min_p: f32x3,
    pub max_p: f32x3,
}

impl AABB {
    pub fn new(min_p: f32x3, max_p: f32x3) -> Self {
        AABB{min_p, max_p}
    }

    pub fn update(&self, bbox: &AABB) -> AABB {
        AABB::new(self.min_p.min(bbox.min_p), self.max_p.max(bbox.max_p))
    }
}

pub trait BBox {
    fn bbox(&self) -> AABB;
}

pub trait BBoxPrimitive {
    fn bbox_primitive(&self, index: usize) -> AABB;
}

pub trait NPrimitives {
    fn nprimitives(&self) -> usize;
}

pub trait ShapeIntersect {
    fn intersect(&self, ray: &Ray, min_dist: f32) -> (f32, i32);
}

pub trait Precompute {
    fn precompute(&mut self) {}
}

pub trait PrimitiveIntersect {
    fn primitive_intersect(&self, index: usize, ray: &Ray, min_dist: f32) -> (f32, i32);
}

pub trait CalculateNormal {
    fn calculate_normal(&self, hitpoint: f32x3, sub_shape: usize) -> f32x3;
}

pub trait LinearIntersect : NPrimitives + BBoxPrimitive + PrimitiveIntersect {
    
    fn linear_intersect(&self, ray: &Ray, min_dist: f32) -> (i32, f32, i32) {
        if self.nprimitives() == 0 { return (-1, 0.0, -1); }
        let mut cur_min_dist = min_dist;
        let mut shape_idx = -1;
        let mut sub_shape_idx = -1;
        for idx in 0..self.nprimitives() {
            let bbox = self.bbox_primitive(idx);
            let tb = intersection_aabb(bbox.min_p, bbox.max_p, ray.origin, ray.inv_direction);
            if tb > 0.0 {
                let (t, sub_shape) = self.primitive_intersect(idx, ray, cur_min_dist);
                if t > 0.0 && t < cur_min_dist {
                    cur_min_dist = t;
                    shape_idx = idx as i32;
                    sub_shape_idx = sub_shape;
                }
            }
       }
       (shape_idx, cur_min_dist, sub_shape_idx)
    }
}

pub trait CollectionIntersect : LinearIntersect {
    fn has_acceleration_structure(&self) -> bool { false }
    fn accel_intersect(&self, _ray: &Ray, _min_dist: f32) -> (i32, f32, i32) { (-1, 0.0, -1) }
    fn collection_intersect(&self, ray: &Ray, min_dist: f32) -> (i32, f32, i32) {
        if self.has_acceleration_structure() {
            self.accel_intersect(ray, min_dist)
        } else {
            self.linear_intersect(ray, min_dist)
        }
    }
}

pub fn intersection_aabb(min_p: f32x3, max_p: f32x3, origin: f32x3, inv_dir: f32x3) -> f32 {
    let t1 = (min_p.0 - origin.0) * inv_dir.0;
    let t2 = (max_p.0 - origin.0) * inv_dir.0;

    let tmin = t1.min(t2);
    let tmax = t1.max(t2);

    let t1 = (min_p.1 - origin.1) * inv_dir.1;
    let t2 = (max_p.1 - origin.1) * inv_dir.1;

    let tmin = tmin.max(t1.min(t2).min(tmax));
    let tmax = tmax.min(t1.max(t2).max(tmin));

    let t1 = (min_p.2 - origin.2) * inv_dir.2;
    let t2 = (max_p.2 - origin.2) * inv_dir.2;

    let tmin = tmin.max(t1.min(t2).min(tmax));
    let tmax = tmax.min(t1.max(t2).max(tmin));

    if tmax > tmin.max(0.0) {
        if tmin < 0.0 {
            return tmax;
        } else {
            return tmin;
        }
    }
    0.0

    //return tmax > tmin.max(0.0);
    // if tmax <= tmin.max(0.0) { return 0.0; } // no intersection
    // if tmin < 0.0 { // if tmin < 0 you are inside of box t = tmax
    //     return tmax;
    // } else {
    //     return tmin;
    // }
}

pub fn isect_sphere(r_origin: f64x3, r_dir: f64x3, position: f64x3, radius: f64, cur_min_dist: f64) -> f64
{
    let temp = r_origin - position;
    let a = r_dir.dot(r_dir);
    let b = temp.dot(r_dir) * 2.0;
    let c = temp.dot(temp) - radius * radius;

    let disc = b * b - 4.0 * a * c;

    let eps = 0.000005;

    if disc < 0.0 {
        return 0.0;
    } else {
        let e = disc.sqrt();
        let denom = 2.0 * a;
        let t = (-b - e) / denom; // smaller root
        if t > eps && t < cur_min_dist {
            return t;
        }

        let t = (-b + e) / denom; // larger root
        if t > eps && t < cur_min_dist {
            return t;
        }
    }
    0.0
}
    
pub fn ray_triangle_isect(v0: f64x3, v1: f64x3, v2: f64x3, origin: f64x3, dir: f64x3) -> f64 {
    
    let a = v0.0 - v1.0;
    let b = v0.0 - v2.0;
    let c = dir.0;
    let d = v0.0 - origin.0;
    let e = v0.1 - v1.1;
    let f = v0.1 - v2.1;
    let g = dir.1;
    let h = v0.1 - origin.1;
    let i = v0.2 - v1.2;
    let j = v0.2 - v2.2;
    let k = dir.2;
    let l = v0.2 - origin.2;

    let m = f * k - g * j;
    let n = h * k - g * l;
    let p = f * l - h * j;
    let q = g * i - e * k;
    let s = e * j - f * i;

    let temp3 =  a * m + b * q + c * s;

    if temp3 == 0.0 { return 0.0; }
    let inv_denom = 1.0 / temp3;

    let e1 = d * m - b * n - c * p;
    let beta = e1 * inv_denom;

    if beta < 0.0 { return 0.0;}

    let r = e * l - h * i;
    let e2 = a * n + d * q + c * r;
    let gamma = e2 * inv_denom;

    if gamma < 0.0 { return 0.0}

    if beta + gamma > 1.0 { return 0.0;}

    let e3 = a * p - b * r + d * s;
    let t = e3 * inv_denom;

    if t < 0.00001 { // self-intersection
        return 0.0;
    }
    t
}
