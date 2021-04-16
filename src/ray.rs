use crate::vec::f32x3;


#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Ray {
    pub origin: f32x3,
    pub direction: f32x3,
    pub inv_direction: f32x3,
}

impl Ray {
    pub fn new(origin: f32x3, direction: f32x3) -> Ray {
        let inv_direction = f32x3(1.0, 1.0, 1.0).div(direction);
        return Ray {origin, direction, inv_direction}
    }
}
