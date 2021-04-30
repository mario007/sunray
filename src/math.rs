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
