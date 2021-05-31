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
	let z = u1.powf((shininess + 1.0).recip());

	let r = math::reflect(wo, normal);

	let (b1, b2) = math::frisvad_revised_onb(r);
    (x * b1 + y * b2 + z * r).normalize()
}

pub fn ward(wo: f32x3, normal: f32x3, wi: f32x3, alpha_x: f32, alpha_y: f32) -> f32 {

	let h = (wo + wi).normalize();

	if wo.dot(normal) * wi.dot(normal) <= 0.0 { return 0.0; }
	if alpha_x * alpha_y <= 0.0 { return 0.0; }

	let denom = 4.0 * f32::consts::PI * alpha_x * alpha_y * (wo.dot(normal) * wi.dot(normal)).sqrt();

	let (b1, b2) = math::frisvad_revised_onb(normal);

	let exponent = -(math::sqr(h.dot(b1) / alpha_x) + math::sqr(h.dot(b2) / alpha_y)) / math::sqr(h.dot(normal));

	exponent.exp() / denom
}

pub fn ward_pdf(wo: f32x3, normal: f32x3, wi: f32x3, alpha_x: f32, alpha_y: f32) -> f32 {

	let h = (wo + wi).normalize();

	let theta = h.dot(normal);
	let denom = 4.0 * f32::consts::PI * alpha_x * alpha_y * h.dot(wi) * theta * theta * theta;

	let (b1, b2) = math::frisvad_revised_onb(normal);

	let exponent = -(math::sqr(h.dot(b1) / alpha_x) + math::sqr(h.dot(b2) / alpha_y)) / math::sqr(h.dot(normal));

	exponent.exp() / denom
}

pub fn sample_ward(wo: f32x3, normal: f32x3, alpha_x: f32, alpha_y: f32, u1: f32, u2: f32) -> f32x3 {
	let cos_u2 = (2.0 * f32::consts::PI * u2).cos();
	let sin_u2 = (2.0 * f32::consts::PI * u2).sin();
	let denom = (math::sqr(alpha_x) * math::sqr(cos_u2) + math::sqr(alpha_y) * math::sqr(sin_u2)).sqrt();
	let cos_phi_h = alpha_x * cos_u2 / denom;
	let sin_phi_h = alpha_y * sin_u2 / denom;

	let f = -(u1.ln()) / (math::sqr(cos_phi_h) / math::sqr(alpha_x) + math::sqr(sin_phi_h) / math::sqr(alpha_y));
	let theta_h = f.sqrt().atan();
	
	let sin_theta = theta_h.sin();
	let cos_theta = theta_h.cos();

	let wh = f32x3(sin_theta * cos_phi_h, sin_theta * sin_phi_h, cos_theta);

	let (b1, b2) = math::frisvad_revised_onb(normal);
	let wh = (wh.0 * b1 + wh.1 * b2 + wh.2 * normal).normalize();

	let wi = math::reflect(wo, wh);
	wi
}

pub fn beckmann_dist(alpha: f32, wo: f32x3, normal: f32x3, wi: f32x3) -> f32 {

	// microsurface normal
	let m = (wo + wi).normalize();

	let ndotm = normal.dot(m);
	let denom = f32::consts::PI * math::sqr(alpha) * math::sqr(ndotm) * math::sqr(ndotm);
	if denom == 0.0 { return 0.0; }

	let exponent = (math::sqr(ndotm) - 1.0) / (math::sqr(alpha) * math::sqr(ndotm));
	exponent.exp() / denom
}

pub fn beckmann_lambda(alpha: f32, normal: f32x3, v: f32x3) -> f32 {
	
	let cos = normal.dot(v);
	if cos > 0.999999 { return 0.0; }
	let sin = (1.0 - cos * cos).sqrt();
	let a = cos / (alpha * sin);

	if a < 1.6 {
		return (1.0 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a); 
	} else {
		return 0.0;
	}
}

pub fn ggx_dist(alpha: f32, wo: f32x3, normal: f32x3, wi: f32x3) -> f32 {

	// microsurface noraml
	let m = (wo + wi).normalize();

	let ndotm = normal.dot(m);

	let f = math::sqr(ndotm) * (math::sqr(alpha) - 1.0) + 1.0;

	math::sqr(alpha) / (f32::consts::PI * math::sqr(f))
}

pub fn ggx_lambda(alpha: f32, normal: f32x3, v: f32x3) -> f32 {
	
	let cos = normal.dot(v);
	if cos > 0.999999 { return 0.0; }
	let sin = (1.0 - cos * cos).sqrt();
	let a = cos / (alpha * sin);

	(-1.0 + (1.0 + 1.0 / (a * a)).sqrt()) * 0.5
}

pub fn smith_g1(lambda: f32) -> f32 {
	(1.0 + lambda).recip()
}

pub fn smith_g2(lambda_wo: f32, lambda_wi: f32) -> f32 {
	(1.0 + lambda_wo + lambda_wi).recip()
}

#[allow(dead_code)]
pub fn spherical_direction(theta: f32, phi: f32, v: f32x3) -> f32x3 {
	let (b1, b2) = math::frisvad_revised_onb(v);
	let phi_cos = phi.cos();
	let phi_sin = phi.sin();
	let theta_sin = theta.sin();
	let theta_cos = theta.cos();
	let x = phi_cos * theta_sin;
	let y = phi_sin * theta_sin;
	let z = theta_cos;
	(x * b1 + y * b2 + z * v).normalize()
}

#[allow(dead_code)]
pub fn beckmann_g1(alpha: f32, normal: f32x3, v: f32x3) -> f32 {
	let ndotv = normal.dot(v);
	let a = ndotv / (alpha * (1.0 - ndotv * ndotv).sqrt());

	if a < 1.6 {
		(3.535 * a + 2.181 * a * a) / (1.0 + 2.276 * a + 2.577 * a * a)
	} else {
		1.0
	}
}

pub fn sample_beckmann(wo: f32x3, normal: f32x3, alpha: f32, u1: f32, u2: f32) -> f32x3 {

	let theta = (1.0 / (1.0 - alpha * alpha * (1.0 - u1).ln())).sqrt().acos();
	let phi = 2.0 * f32::consts::PI * u2;

	let wh = spherical_direction(theta, phi, normal);
	let wi = math::reflect(wo, wh);
	wi
}

// Note: For microfacet use microfacet normal(half vector) instead of geometric normal
pub fn fresnel_schlick(f0: f32x3, wi: f32x3, normal: f32x3) -> f32x3 {
	let a = 1.0 - normal.dot(wi);
	f0 + (f32x3::from(1.0) - f0) * (a * a * a * a * a)
}

// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
pub fn fresnel_conductor(eta: f32x3, etak: f32x3, cos_theta: f32) -> f32x3
{  
   let cos_theta2 = cos_theta * cos_theta;
   let sin_theta2 = 1.0 - cos_theta2;
   let eta2 = eta.mul(eta);
   let etak2 = etak.mul(etak);

   let t0 = eta2 - etak2 - f32x3::from(sin_theta2);
   let tmp = t0.mul(t0) + 4.0 * eta2.mul(etak2);
   let a2plusb2 = f32x3(tmp.0.sqrt(), tmp.1.sqrt(), tmp.2.sqrt());
   let t1 = a2plusb2 + f32x3::from(cos_theta2);
   let tmp = 0.5 * (a2plusb2 + t0);
   let a = f32x3(tmp.0.sqrt(), tmp.1.sqrt(), tmp.2.sqrt());
   let t2 = 2.0 * a.mul(f32x3::from(cos_theta));
   let rs = (t1 - t2).div(t1 + t2);

   let t3 = cos_theta2 * a2plusb2 + f32x3::from(sin_theta2 * sin_theta2);
   let t4 = t2 * sin_theta2;   
   let rp = rs.mul((t3 - t4).div(t3 + t4));

   return 0.5 * (rp + rs);
}

#[allow(dead_code)]
pub fn sample_ggx(wo: f32x3, normal: f32x3, alpha: f32, u1: f32, u2: f32) -> f32x3 {

	let theta = ((1.0 - u1) / ((alpha*alpha - 1.0) * u1 + 1.0)).sqrt().acos();
	let phi = 2.0 * f32::consts::PI * u2;

	let wh = spherical_direction(theta, phi, normal);
	let wi = math::reflect(wo, wh);
	wi
}

// Input Ve: view direction
// Input alpha_x, alpha_y: roughness parameters
// Input U1, U2: uniform random numbers
// Output Ne: normal sampled with PDF D_Ve(Ne) = G1(Ve) * max(0, dot(Ve, Ne)) * D(Ne) / Ve.z
#[allow(non_snake_case)]
pub fn sample_ggxvndf(wo: f32x3, normal: f32x3, alpha_x: f32, alpha_y: f32, u1: f32, u2: f32) -> f32x3
{
	fn inversesqrt(x: f32) -> f32 { 1.0 / x.sqrt() }

	let (b1, b2) = math::frisvad_revised_onb(normal);
	let wo_local = f32x3(b1.dot(wo), b2.dot(wo), normal.dot(wo));

	// Section 3.2: transforming the view direction to the hemisphere configuration
	let Vh = f32x3(alpha_x * wo_local.0, wo_local.1 * alpha_y, wo_local.2).normalize();

	// Section 4.1: orthonormal basis (with special case if cross product is zero)
	let lensq = Vh.0 * Vh.0 + Vh.1 * Vh.1;
	let T1: f32x3;
	if lensq > 0.0 {
		T1 = f32x3(-Vh.1, Vh.0, 0.0) * inversesqrt(lensq)
	} else {
		T1 = f32x3(1.0, 0.0, 0.0);
	}
	let T2 = Vh.cross(T1);

	// Section 4.2: parameterization of the projected area
	let r = u1.sqrt();
	let phi = 2.0 * f32::consts::PI * u2;
	let t1 = r * phi.cos();
	let mut t2 = r * phi.sin();
	let s = 0.5 * (1.0 + Vh.2);
	t2 = (1.0 - s) * (1.0 - t1 * t1).sqrt() + s * t2;

	// Section 4.3: reprojection onto hemisphere
	let Nh = t1 * T1 + t2 * T2 + (1.0 - t1 * t1 - t2 * t2).max(0.0).sqrt() * Vh;

	// Section 3.4: transforming the normal back to the ellipsoid configuration
	let Ne = f32x3(alpha_x * Nh.0, alpha_y * Nh.1, Nh.2.max(0.0)).normalize();

	//let wi_local = math::reflect(wo_local, Ne);
	//let wi = (wi_local.0 * b1 + wi_local.1 * b2 + wi_local.2 * normal).normalize();

	let Ne_world = (Ne.0 * b1 + Ne.1 * b2 + Ne.2 * normal).normalize();
	let wi = math::reflect(wo, Ne_world); 
	wi
}

// If used, roughness values are expected to be in the range [0,1]
pub fn ggx_and_beckmann_roughness_to_alpha(roughness: f32) -> f32 {
    let x = roughness.max(0.001).ln();
    1.62142 + 0.819955 * x + 0.1734 * x * x + 0.0171201 * x * x * x + 0.000640711 * x * x * x * x
}


#[cfg(test)]
mod tests {

    use super::*;
	use crate::pcg::PCGRandom;
	use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn weak_white_furnace_test_beckman() {
		//integrate_isotropic_dist(beckmann_dist, beckmann_lambda);
		let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
		let mut rnd = PCGRandom::new(t, 0);
		let alpha = rnd.random_f32();
		let theta_o = rnd.random_f32();
		let normal = f32x3(rnd.random_f32(), rnd.random_f32(), rnd.random_f32()).normalize();

		let view_vector = spherical_direction(theta_o, 0.0, normal);
		let lambda = beckmann_lambda(alpha, normal, view_vector);
		let g1 = smith_g1(lambda);

		let mut integral = 0.0;
		let dtheta = 0.005;
		let dphi = 0.005;

		let mut theta = 0.0;
		let mut phi;

		while theta < f32::consts::PI {
			phi = 0.0;
			while phi < 2.0 * f32::consts::PI {
				// reflected vector
				let l = spherical_direction(theta, phi, normal);
				let d = beckmann_dist(alpha, view_vector, normal, l);
				integral += theta.sin() * d * g1 / (4.0 * view_vector.dot(normal).abs());
				phi += dphi;
			}
			theta += dtheta;
		}

		integral *= dphi * dtheta;
		let diff = (1.0 - integral).abs();
		println!("diff {} {} {}", diff, alpha, theta_o);
		println!("normal {:?}", normal);
		assert!(diff < 0.02, "Microfacet distribution failed")
    }

	#[test]
    fn weak_white_furnace_test_ggx() {
		//integrate_isotropic_dist(beckmann_dist, beckmann_lambda);
		let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
		let mut rnd = PCGRandom::new(t, 0);
		let alpha = rnd.random_f32();
		let theta_o = rnd.random_f32();
		let normal = f32x3(rnd.random_f32(), rnd.random_f32(), rnd.random_f32()).normalize();
		let view_vector = spherical_direction(theta_o, 0.0, normal);
		let lambda = ggx_lambda(alpha, normal, view_vector);
		let g1 = smith_g1(lambda);

		let mut integral = 0.0;
		let dtheta = 0.005;
		let dphi = 0.005;

		let mut theta = 0.0;
		let mut phi;

		while theta < f32::consts::PI {
			phi = 0.0;
			while phi < 2.0 * f32::consts::PI {
				// reflected vector
				let l = spherical_direction(theta, phi, normal);
				let d = ggx_dist(alpha, view_vector, normal, l);
				integral += theta.sin() * d * g1 / (4.0 * view_vector.dot(normal).abs());
				phi += dphi;
			}
			theta += dtheta;
		}

		integral *= dphi * dtheta;
		let diff = (1.0 - integral).abs();
		println!("diff {} {} {}", diff, alpha, theta_o);
		assert!(diff < 0.02, "Microfacet distribution failed")
    }

	#[test]
    fn test_roughness_to_alpha() {

		println!("{}", ggx_and_beckmann_roughness_to_alpha(0.0));
		println!("{}", ggx_and_beckmann_roughness_to_alpha(0.01));
		println!("{}", ggx_and_beckmann_roughness_to_alpha(0.1));
		println!("{}", ggx_and_beckmann_roughness_to_alpha(0.2));
		println!("{}", ggx_and_beckmann_roughness_to_alpha(0.5));
		println!("{}", ggx_and_beckmann_roughness_to_alpha(0.9));
		println!("{}", ggx_and_beckmann_roughness_to_alpha(1.0));

	}

}
