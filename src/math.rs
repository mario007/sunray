use crate::vec::f32x3;


pub fn distance(p1: f32x3, p2: f32x3) -> f32 {
    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    let dz = p2.2 - p1.2;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

pub fn distance_sqr(p1: f32x3, p2: f32x3) -> f32 {
    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    let dz = p2.2 - p1.2;
    dx * dx + dy * dy + dz * dz
}
