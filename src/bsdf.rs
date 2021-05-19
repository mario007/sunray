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

pub fn sample_phong(wo: f32x3, normal: f32x3, shininess: f32, u1: f32, u2: f32) -> f32x3 {
	let term1 = (1.0 - u1.powf(2.0 / (shininess + 1.0))).sqrt();
	let term2 = 2.0 * f32::consts::PI * u2;

	let x = term1 * term2.cos();
	let y = term1 * term2.sin();
	let z = u1.powf(1.0 / (shininess + 1.0));

	let r = math::reflect(wo, normal);

	let (b1, b2) = math::frisvad_revised_onb(r);
    (x * b1 + y * b2 + z * r).normalize()
}

pub fn ward(wo: f32x3, normal: f32x3, wi: f32x3, alpha_x: f32, alpha_y: f32) -> f32 {

	fn sqr(x: f32) -> f32 {x * x}

	let h = (wo + wi).normalize();

	if wo.dot(normal) * wi.dot(normal) <= 0.0 { return 0.0; }
	if alpha_x * alpha_y <= 0.0 {return 0.0; }

	let denom = 4.0 * f32::consts::PI * alpha_x * alpha_y * (wo.dot(normal) * wi.dot(normal)).sqrt();

	let (b1, b2) = math::frisvad_revised_onb(normal);

	let exponent = -(sqr(h.dot(b1) / alpha_x) + sqr(h.dot(b2) / alpha_y)) / sqr(h.dot(normal));

	exponent.exp() / denom
}

pub fn ward_pdf(wo: f32x3, normal: f32x3, wi: f32x3, alpha_x: f32, alpha_y: f32) -> f32 {

	fn sqr(x: f32) -> f32 {x * x}

	let h = (wo + wi).normalize();

	let theta = h.dot(normal);
	let denom = 4.0 * f32::consts::PI * alpha_x * alpha_y * h.dot(wi) * theta * theta * theta;

	let (b1, b2) = math::frisvad_revised_onb(normal);

	let exponent = -(sqr(h.dot(b1) / alpha_x) + sqr(h.dot(b2) / alpha_y)) / sqr(h.dot(normal));

	exponent.exp() / denom
}

pub fn sample_ward(wo: f32x3, normal: f32x3, alpha_x: f32, alpha_y: f32, u1: f32, u2: f32) -> f32x3 {

	fn sqr(x: f32) -> f32 {x * x}

	let mut phi_h = (alpha_y / alpha_x * (2.0 * f32::consts::PI * u2).tan()).atan();
	// phi must be in the same quadrant as angle 2*pi*u2
	if u2 > 0.5 {phi_h += f32::consts::PI};

	let cos_phi_h = phi_h.cos();
	let sin_phi_h = (1.0 - cos_phi_h*cos_phi_h).sqrt();

	let f = -(u1.log(f32::consts::E)) / (sqr(cos_phi_h)/sqr(alpha_x) + sqr(sin_phi_h)/sqr(alpha_y));
	let theta_h = f.sqrt().atan();
	
	let sin_theta = theta_h.sin();
	let cos_theta = theta_h.cos();

	let wh = f32x3(sin_theta*cos_phi_h, sin_theta*sin_phi_h, cos_theta);

	let (b1, b2) = math::frisvad_revised_onb(normal);
	let wh = (wh.0 * b1 + wh.1 * b2 + wh.2 * normal).normalize();

	let wi = math::reflect(wo, wh);
	wi
}