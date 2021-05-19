use crate::vec::f32x3;


pub fn luminance(color: f32x3) -> f32 {
    0.2126 * color.0 + 0.7152 * color.1 + 0.0722 * color.2
}
