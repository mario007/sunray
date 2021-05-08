use std::f32;

use crate::vec::f32x3;
use crate::math;


pub fn lambertian() -> f32 {
    f32::consts::FRAC_1_PI
}

pub fn oren_nayar(wo: f32x3, normal: f32x3, wi: f32x3, sigma: f32) -> f32 {

    let n_dot_v = normal.dot(wo);
    let n_dot_l = normal.dot(wi); 

    let theta_v = n_dot_v.acos();
    let theta_l = n_dot_l.acos();

	let alpha = theta_v.max(theta_l);
	let beta = theta_v.min(theta_l);

	// Calculate cosine of azimuth angles difference - by projecting L and V onto plane.
	let l = wi - n_dot_l * normal;
	let v = wo - n_dot_v * normal;
	let cos_phi_difference = v.normalize().dot(l.normalize());

	let sigma2 = sigma * sigma;
	let a = 1.0 - 0.5 * (sigma2 / (sigma2 + 0.33));
	let b = 0.45 * (sigma2 / (sigma2 + 0.09));

	return (a + b * cos_phi_difference.max(0.0) * alpha.sin() * beta.tan()) * f32::consts::FRAC_1_PI;
}

pub fn phong(wo: f32x3, normal: f32x3, wi: f32x3, shininess: f32) -> f32 {
	let r = math::reflect(wi, normal);
	let r_dot_wo = r.dot(wo).max(0.0);
	let coef = (shininess + 2.0) * f32::consts::FRAC_1_PI * 0.5; 
	coef * r_dot_wo.powf(shininess)
}

pub fn phong_pdf(wo: f32x3, normal: f32x3, wi: f32x3, shininess: f32) -> f32 {
	let r = math::reflect(wi, normal);
	let r_dot_wo = r.dot(wo).max(0.0);
	let coef = (shininess + 1.0) * f32::consts::FRAC_1_PI * 0.5; 
	coef * r_dot_wo.powf(shininess)
}

pub fn sample_phong(normal: f32x3, shininess: f32, u1: f32, u2: f32) -> f32x3 {
	let term1 = (1.0 - u1.powf(2.0 / (shininess + 1.0))).sqrt();
	let term2 = 2.0 * f32::consts::PI * u2;

	let x = term1 * term2.cos();
	let y = term1 * term2.sin();
	let z = u1.powf(1.0 / (shininess + 1.0));

	let (b1, b2) = math::frisvad_revised_onb(normal);
    (x * b1 + y * b2 + z * normal).normalize()
}
