use std::f32;

use crate::vec::f32x3;
use crate::math::frisvad_revised_onb;

// WtoA = PdfW * cosine / distance_squared
// AtoW = PdfA * distance_squared / cosine

pub fn pdfw_to_a(pdfw: f32, dist: f32, cos_there: f32) -> f32 {
    pdfw * cos_there.abs() / (dist * dist)
}

pub fn pdfa_to_w(pdfa: f32, dist: f32, cos_there: f32) -> f32 {
    pdfa * (dist * dist) / cos_there.abs()
}

pub fn cosine_hemi_pdf(n_dot_wi: f32) -> f32 {
    n_dot_wi * f32::consts::FRAC_1_PI
}

pub fn cosine_hemi_direction(normal: f32x3, u1: f32, u2: f32) -> f32x3 {
    //let u1_sqrt = u1.sqrt();
    //let x = u1_sqrt * (2.0 * f32::consts::PI * u2).cos();
    //let y = u1_sqrt * (2.0 * f32::consts::PI * u2).sin();
    //let z = (1.0 - u1).sqrt();

    let term1 = 2.0 * f32::consts::PI * u1;
    let term2 = (1.0 - u2).sqrt();
    let x = (term1).cos() * term2;
    let y = (term1).sin() * term2;
    let z = u2.sqrt();

    let (b1, b2) = frisvad_revised_onb(normal);
    (x * b1 + y * b2 + z * normal).normalize()
}

pub fn hemi_pdf() -> f32 {
    0.5 * f32::consts::FRAC_1_PI
}

pub fn hemi_direction(normal: f32x3, u1: f32, u2: f32) -> f32x3 {
    let x = (2.0 * f32::consts::PI * u2).cos() * (1.0 - u1 * u1).sqrt();
    let y = (2.0 * f32::consts::PI * u2).sin() * (1.0 - u1 * u1).sqrt();
    let z = u1;

    let (b1, b2) = frisvad_revised_onb(normal);
    (x * b1 + y * b2 + z * normal).normalize()
}

/*pub fn uniform_sampling_triangle(u1: f32, u2: f32) -> (f32, f32, f32) {
    let u_sqrt = u1.sqrt();
    let u = 1.0 - u_sqrt;
    let v = u2 * u_sqrt;
    let w = 1.0 - u - v;
    (u, v, w)
}*/

// A Low-Distortion Map Between Triangle and Square   -   Eric Heitz
pub fn uniform_sampling_triangle(u1: f32, u2: f32) -> (f32, f32, f32) {
    let u: f32;
    let v: f32;
    if u2 > u1 {
        u = u1 * 0.5;
        v = u2 - u;
    } else {
        v = u2 * 0.5;
        u = u1 - v;
    }
    let w = 1.0 - u - v;
    (u, v, w)
}
