use crate::vec::{f64x3, u32x3};
use crate::ray::Ray;
use crate::isects::{PrimitiveIntersect, BBox, BBoxPrimitive, NPrimitives};


pub struct UniGrid {
    dim: u32x3,
    inv_dim: f64x3,
    pmin: f64x3,
    pmax: f64x3,
    inv_cell_dim: f64x3,
    primitive_map: Vec<i32>,
    pub primitives: Vec<i32>,
}

impl UniGrid {
    pub fn new(pmin: f64x3, pmax: f64x3, dim: u32x3) -> UniGrid {
        let inv_dim = f64x3::from(1.0).div(f64x3::from(dim));
        let cell_dim = (pmax - pmin).div(f64x3::from(dim));
        let inv_cell_dim = f64x3(1.0, 1.0, 1.0).div(cell_dim);
        UniGrid {
            dim, inv_dim, pmin, pmax, inv_cell_dim,
            primitive_map: Vec::<i32>::new(),
            primitives: Vec::<i32>::new(),
        }
    }

    #[allow(dead_code)]
    pub fn memory(&self) -> usize {
        let val1 = self.primitive_map.len() * 4;
        let val2 = self.primitives.len() * 4;
        let rest: usize = 86;
        val1 + val2 + rest
    }
}

fn calc_grid_dim(pmin: f64x3, pmax: f64x3, n: usize) -> u32x3 {
    let w = pmax - pmin;
    let multiplier = 2.0; // about 8 times more cells than objects if multiplier is 2!
    let s = (w.0 * w.1 * w.2 / (n as f64)).powf(0.33333333);
    let nx = ((multiplier * w.0 / s) + 1.0) as u32;
    let ny = ((multiplier * w.1 / s) + 1.0) as u32;
    let nz = ((multiplier * w.2 / s) + 1.0) as u32;
    let max_width = u32x3(512, 512, 512);
    u32x3(nx, ny, nz).min(max_width)
}

pub trait UniGridBuild : BBox + BBoxPrimitive + NPrimitives {}

fn populate_uni_grid_cells(pmin: f64x3, pmax: f64x3, dim: u32x3,
    primitives: &[u32], mesh: &impl UniGridBuild) -> Vec<Vec<u32>> {

    let cell_dim = (pmax - pmin).div(f64x3(dim.0 as f64, dim.1 as f64, dim.2 as f64));
    let inv_cell_dim = f64x3(1.0, 1.0, 1.0).div(cell_dim);
    let num_cels = dim.0 * dim.1 * dim.2;

    let mut prim_indices: Vec<Vec<u32>> = Vec::with_capacity(num_cels as usize);
    for _i in 0..num_cels {
        prim_indices.push(Vec::new())
    }

    let max_indices = u32x3(dim.0 - 1, dim.1 - 1, dim.2 - 1);

    for i in primitives {
    let aabb = mesh.bbox_primitive(*i as usize);
    let idx = u32x3::from((f64x3::from(aabb.min_p) - pmin).mul(inv_cell_dim));
    let start_idx = idx.min(max_indices);
    let idx = u32x3::from((f64x3::from(aabb.max_p) - pmin).mul(inv_cell_dim));
    let end_idx = idx.min(max_indices);
        for z in start_idx.2..end_idx.2 + 1 {
            for y in start_idx.1..end_idx.1 + 1  {
                for x in start_idx.0..end_idx.0 + 1 {
                    let index = z * dim.0 * dim.1 + y * dim.0 + x;
                    // TODO triangle box overlap              
                    prim_indices[index as usize].push(*i as u32);
                }
            }
        }
    }
    prim_indices
}

fn compress_uni_grid(grid: &mut UniGrid, cell_primitives: &[Vec<u32>]){

    let dim = grid.dim;

    for z in 0..dim.2 {
        for y in 0..dim.1 {
            for x in 0..dim.0 {
                let index = z * dim.0 * dim.1 + y * dim.0 + x;
                let arr = &cell_primitives[index as usize]; 
                if !arr.is_empty() {
                    grid.primitive_map.push(grid.primitives.len() as i32);
                    for k in arr {
                        grid.primitives.push(*k as i32);
                    }
                    grid.primitives.push(-1);
                } else {
                    grid.primitive_map.push(-1);
                }
            }
        }
    }
}

pub fn create_uni_grid(shapes: &impl UniGridBuild) -> UniGrid {

    let bbox = shapes.bbox();
    let pmin = f64x3::from(bbox.min_p);
    let pmax = f64x3::from(bbox.max_p);
    let dim = calc_grid_dim(pmin, pmax, shapes.nprimitives());
    let primitives: Vec<u32> = (0..shapes.nprimitives() as u32).collect();
    let cell_primitives = populate_uni_grid_cells(pmin, pmax, dim, &primitives, shapes);
    let mut grid = UniGrid::new(pmin, pmax, dim);
    compress_uni_grid(&mut grid, &cell_primitives);
    grid
}


pub trait GridIntersect {
    fn pmin(&self) -> f64x3;
    fn pmax(&self) -> f64x3;
    fn inv_cell_dim(&self) -> f64x3;
    fn dim(&self) -> u32x3;
    fn inv_dim(&self) -> f64x3;
    fn has_primitives(&self, index: usize) -> bool;
    fn isect_primitives(&self, index: usize, ray: &Ray, prim: &impl PrimitiveIntersect) -> (i32, f32, i32);
}

impl GridIntersect for UniGrid {

    fn pmin(&self) -> f64x3 {
        self.pmin
    }
    fn pmax(&self) -> f64x3 {
        self.pmax
    }
    fn inv_cell_dim(&self) -> f64x3 {
        self.inv_cell_dim
    }
    fn dim(&self) -> u32x3 {
        self.dim
    }
    fn inv_dim(&self) -> f64x3 {
        self.inv_dim
    }
    fn has_primitives(&self, index: usize) -> bool {
        self.primitive_map[index] != -1
    }

    fn isect_primitives(&self, index: usize, ray: &Ray, primitives: &impl PrimitiveIntersect) -> (i32, f32, i32) {
        let prim_offset = self.primitive_map[index];
        if prim_offset == -1 {
            return (-1, 1e30, -1);
        }

        let mut offset = prim_offset as usize;
        let mut idx = -1;
        let mut sub_shape_idx = -1;
        let mut min_t = 1e30;
        loop {
            let prim_idx = self.primitives[offset];
            if prim_idx == -1 { break; }
            let (t, sub_shape) = primitives.primitive_intersect(prim_idx as usize, ray, min_t);

            if t > 0.0 && t < min_t {
                idx = prim_idx;
                sub_shape_idx = sub_shape;
                min_t = t;
            }
            offset += 1;
        }
        (idx, min_t, sub_shape_idx)
    }
}

fn point_inside_bbox(pmin: f64x3, pmax: f64x3, point: f64x3) -> bool {
    point.0 >= pmin.0 && point.0 <= pmax.0 &&
    point.1 >= pmin.1 && point.1 <= pmax.1 &&
    point.2 >= pmin.2 && point.2 <= pmax.2
}

pub fn intersect(grid: &impl GridIntersect, prim: &impl PrimitiveIntersect, ray: &Ray, min_dist: f32) -> (i32, f32, i32) {
    let origin = f64x3::from(ray.origin);
    let inv_direction = f64x3::from(ray.inv_direction);
    let direction = f64x3::from(ray.direction);

    let pmin = grid.pmin();
    let pmax = grid.pmax();
    let inv_cell_dim = grid.inv_cell_dim();
    let dim = grid.dim();
    let inv_dim = grid.inv_dim();

    let t1 = (pmin - origin).mul(inv_direction);
    let t2 = (pmax - origin).mul(inv_direction);

    let tx_min = t1.0.min(t2.0);
    let tx_max = t1.0.max(t2.0);
    let ty_min = t1.1.min(t2.1);
    let ty_max = t1.1.max(t2.1);
    let tmin = tx_min.max(ty_min);
    let tmax = tx_max.min(ty_max);
    let tz_min = t1.2.min(t2.2);
    let tz_max = t1.2.max(t2.2);
    let tmin = tmin.max(tz_min);
    let tmax = tmax.min(tz_max);

    if tmax <= tmin.max(0.0) { return (-1, 0.0, -1); } // no intersection
    if tmin > 0.0 && tmin > min_dist as f64 {
        return (-1, 0.0, -1);
    } 

    let mut t = tmin; 
    if tmin < 0.0 { t = tmax; } // if tmin < 0 you are inside of box t = tmax

    let mut start_point = origin + t * direction;
    if point_inside_bbox(pmin, pmax, origin) {
        start_point = origin;
    }

    fn int_coord(start_p: f64, end_p: f64, inv_dim: f64, dim: u32) -> i32 {
        let mut p = ((end_p - start_p) * inv_dim) as u32;
        if p > dim { p = dim; }
        p as i32
    }
    let mut ix = int_coord(pmin.0, start_point.0, inv_cell_dim.0, dim.0 - 1);
    let mut iy = int_coord(pmin.1, start_point.1, inv_cell_dim.1, dim.1 - 1);
    let mut iz = int_coord(pmin.2, start_point.2, inv_cell_dim.2, dim.2 - 1);

    let delta_x = (tx_max - tx_min) * inv_dim.0;
    let delta_y = (ty_max - ty_min) * inv_dim.1;
    let delta_z = (tz_max - tz_min) * inv_dim.2;

    fn setup_loop(t_min: f64, index: i32, delta: f64, dir: f64, n: u32) -> (f64, i32, i32) {
        if dir > 0.0 {
            let t_next = t_min + (index + 1) as f64 * delta;
            (t_next, 1, n as i32)
        } else if dir < 0.0 {
            let t_next = t_min + (n - index as u32) as f64 * delta;
            (t_next, -1, -1)
        } else {
            (1e30, -1, -1)
        }
    }

    let (mut tx_next, ix_step, ix_stop) = setup_loop(tx_min, ix, delta_x, direction.0, dim.0);
    let (mut ty_next, iy_step, iy_stop) = setup_loop(ty_min, iy, delta_y, direction.1, dim.1);
    let (mut tz_next, iz_step, iz_stop) = setup_loop(tz_min, iz, delta_z, direction.2, dim.2);
    let (nx, ny, _nz) = (dim.0 as i32, dim.1 as i32, dim.2 as i32);

    let mut prim_isect_idx = -1;
    let mut min_t_isect = 1e30;
    let mut sub_prim_isect_idx = -1;


    loop {
        let index = iz * nx * ny + iy * nx + ix;
        if grid.has_primitives(index as usize) {
            let (isect_idx, t_isect, sub_shape_idx) = grid.isect_primitives(index as usize, ray, prim);
            if isect_idx != -1 && t_isect > min_dist {
                return (-1, 0.0, -1);
            }
            prim_isect_idx = isect_idx;
            min_t_isect = t_isect as f64;
            sub_prim_isect_idx = sub_shape_idx;
        }

        if tx_next < ty_next && tx_next < tz_next {
            if prim_isect_idx != -1 && min_t_isect <= tx_next + f64::EPSILON {
                return (prim_isect_idx, min_t_isect as f32, sub_prim_isect_idx);
            }
            //println!("x {} {} {} {}", ix, iy, iz, min_t_isect);
            tx_next += delta_x;
            ix += ix_step;
            if ix == ix_stop { 
                return (-1, 0.0, -1);
            }
        } else if ty_next < tz_next {
            if prim_isect_idx != -1 && min_t_isect <= ty_next + f64::EPSILON {
                return (prim_isect_idx, min_t_isect as f32, sub_prim_isect_idx);
            }
            //println!("y {} {} {} {} {}", ix, iy, iz, min_t_isect, ty_next);
            ty_next += delta_y;
            iy += iy_step;
            if iy == iy_stop {
                return (-1, 0.0, -1);
            }
        }  else {
            if prim_isect_idx != -1 && min_t_isect <= tz_next + f64::EPSILON {
                return (prim_isect_idx, min_t_isect as f32, sub_prim_isect_idx);
            }
            //println!("z {} {} {} {}", ix, iy, iz, min_t_isect);
            tz_next += delta_z;
            iz += iz_step;
            if iz == iz_stop { 
             return (-1, 0.0, -1);
            }
        }
    }
}
