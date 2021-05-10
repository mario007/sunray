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

pub fn frisvad_revised_onb(normal: f32x3) -> (f32x3, f32x3) {
    if normal.2 < 0. {
        let a = 1.0 / (1.0 - normal.2);
        let b = normal.0 * normal.1 * a;
        let b1 = f32x3(1.0 - normal.0 * normal.0 * a, -b, normal.0);
        let b2 = f32x3(b, normal.1 * normal.1 * a - 1.0, -normal.1);
        return (b1, b2);
    } else {
        let a = 1.0 / (1.0 + normal.2);
        let b = -normal.0 * normal.1 * a;
        let b1 = f32x3(1.0 - normal.0 * normal.0 * a, b, -normal.0);
        let b2 = f32x3(b, 1.0 - normal.1 * normal.1 * a, -normal.1);
        return (b1, b2);
    }
}

pub fn reflect(v: f32x3, normal: f32x3) -> f32x3 {
    2.0 * v.dot(normal) * normal - v
}


pub fn barycentric(p: f32x3, a: f32x3, b: f32x3, c: f32x3) -> (f32, f32, f32) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0f32 - v - w;
    (u, v, w)
}
