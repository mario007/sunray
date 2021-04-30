use std::ops:: {Add, Sub, Mul, Neg};


#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct u32x3(pub u32, pub u32, pub u32);

impl From<f64x3> for u32x3 {
    fn from(src: f64x3) -> u32x3 {
        u32x3(src.0 as u32, src.1 as u32, src.2 as u32)
    }
}

impl u32x3 {
    pub fn min(self: u32x3, v2: u32x3) -> u32x3 {
        u32x3(self.0.min(v2.0), self.1.min(v2.1), self.2.min(v2.2))
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct f32x2(pub f32, pub f32);


#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct f32x3(pub f32, pub f32, pub f32);

impl Add for f32x3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self (self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }
}

impl Sub for f32x3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self (self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

impl Mul<f32> for f32x3 {
    type Output = Self;

    fn mul(self, other: f32) -> Self {
        Self (self.0 * other, self.1 * other, self.2 * other)
    }
}

impl Mul<f32x3> for f32 {
    type Output = f32x3;

    fn mul(self, other: f32x3) -> f32x3 {
        f32x3 (self * other.0, self * other.1, self * other.2)
    }
}

impl Neg for f32x3 {
    type Output = Self;

    fn neg(self) -> Self {
        Self (-self.0, -self.1, -self.2)
    }
}

impl From<f64x3> for f32x3 {
    fn from(src: f64x3) -> f32x3 {
        f32x3(src.0 as f32, src.1 as f32, src.2 as f32)
    }
}

impl From<f32> for f32x3 {
    fn from(src: f32) -> f32x3 {
        f32x3(src, src, src)
    }
}

impl f32x3 {

    pub fn dot(self: f32x3, v2: f32x3) -> f32 {
        self.0 * v2.0 + self.1 * v2.1 + self.2 * v2.2 
    }

    pub fn mul(self: f32x3, v2: f32x3) -> f32x3 {
        f32x3(self.0 * v2.0, self.1 * v2.1, self.2 * v2.2)
    }

    pub fn div(self: f32x3, v2: f32x3) -> f32x3 {
        f32x3(self.0 / v2.0, self.1 / v2.1, self.2 / v2.2)
    }

    pub fn min(self: f32x3, v2: f32x3) -> f32x3 {
        f32x3(self.0.min(v2.0), self.1.min(v2.1), self.2.min(v2.2))
    }

    pub fn max(self: f32x3, v2: f32x3) -> f32x3 {
        f32x3(self.0.max(v2.0), self.1.max(v2.1), self.2.max(v2.2))
    }

    pub fn cross(self: f32x3, v2: f32x3) -> f32x3 {
        let (vx, vy, vz) = (self.0 as f64, self.1 as f64, self.2 as f64);
        let (wx, wy, wz) = (v2.0 as f64, v2.1 as f64, v2.2 as f64);
        f32x3((vy*wz - vz*wy) as f32, (vz*wx - vx*wz) as f32, (vx*wy - vy*wx) as f32)
    }

    pub fn normalize(self: f32x3) -> f32x3 {
        let len = (self.0 * self.0 + self.1 * self.1 + self.2 * self.2).sqrt();
        f32x3(self.0 / len, self.1 / len, self.2 / len)
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct f64x3(pub f64, pub f64, pub f64);

impl Add for f64x3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self (self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }
}

impl Sub for f64x3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self (self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

impl Mul<f64> for f64x3 {
    type Output = Self;

    fn mul(self, other: f64) -> Self {
        Self (self.0 * other, self.1 * other, self.2 * other)
    }
}

impl Mul<f64x3> for f64 {
    type Output = f64x3;

    fn mul(self, other: f64x3) -> f64x3 {
        f64x3 (self * other.0, self * other.1, self * other.2)
    }
}

impl From<f32x3> for f64x3 {
    fn from(src: f32x3) -> f64x3 {
        f64x3 (src.0 as f64, src.1 as f64, src.2 as f64)
    }
}

impl From<f64> for f64x3 {
    fn from(src: f64) -> f64x3 {
        f64x3(src, src, src)
    }
}

impl From<u32x3> for f64x3 {
    fn from(src: u32x3) -> f64x3 {
        f64x3(src.0 as f64, src.1 as f64, src.2 as f64)
    }
}

impl f64x3 {

    pub fn dot(self: f64x3, v2: f64x3) -> f64 {
        self.0 * v2.0 + self.1 * v2.1 + self.2 * v2.2 
    }

    pub fn mul(self: f64x3, v2: f64x3) -> f64x3 {
        f64x3(self.0 * v2.0, self.1 * v2.1, self.2 * v2.2)
    }

    pub fn div(self: f64x3, v2: f64x3) -> f64x3 {
        f64x3(self.0 / v2.0, self.1 / v2.1, self.2 / v2.2)
    }

    pub fn min(self: f64x3, v2: f64x3) -> f64x3 {
        f64x3(self.0.min(v2.0), self.1.min(v2.1), self.2.min(v2.2))
    }

    pub fn max(self: f64x3, v2: f64x3) -> f64x3 {
        f64x3(self.0.max(v2.0), self.1.max(v2.1), self.2.max(v2.2))
    }

    pub fn cross(self: f64x3, v2: f64x3) -> f64x3 {
        f64x3(self.1 * v2.2 - self.2 * v2.1,
              self.2 * v2.0 - self.0 * v2.2,
              self.0 * v2.1 - self.1 * v2.0)
    }

    pub fn normalize(self: f64x3) -> f64x3 {
        let len = (self.0 * self.0 + self.1 * self.1 + self.2 * self.2).sqrt();
        f64x3(self.0 / len, self.1 / len, self.2 / len)
    }
}


#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn midpoint_f32() {
        let v1 = f32x3(2.0, 3.0, 4.0);
        let v2 = f32x3(2.0, 2.0, 5.0);
        let midpoint = 0.5 * (v1 + v2);
        assert_eq!(midpoint.0, 2.0);
        assert_eq!(midpoint.1, 2.5);
        assert_eq!(midpoint.2, 4.5);
    }

    #[test]
    fn bbox_f32() {
        let v1 = f32x3(1.0, 2.0, 3.0);
        let v2 = f32x3(0.0, 3.0, 2.0);
        let p1 = v1.min(v2);
        let p2 = v1.max(v2);

        assert_eq!(p1.0, 0.0);
        assert_eq!(p1.1, 2.0);
        assert_eq!(p1.2, 2.0);

        assert_eq!(p2.0, 1.0);
        assert_eq!(p2.1, 3.0);
        assert_eq!(p2.2, 3.0);
    }

    #[test]
    fn midpoint_f64() {
        let v1 = f64x3(2.0, 3.0, 4.0);
        let v2 = f64x3(2.0, 2.0, 5.0);
        let midpoint = 0.5 * (v1 + v2);
        assert_eq!(midpoint.0, 2.0);
        assert_eq!(midpoint.1, 2.5);
        assert_eq!(midpoint.2, 4.5);
    }

    #[test]
    fn bbox_f64() {
        let v1 = f64x3(1.0, 2.0, 3.0);
        let v2 = f64x3(0.0, 3.0, 2.0);
        let p1 = v1.min(v2);
        let p2 = v1.max(v2);

        assert_eq!(p1.0, 0.0);
        assert_eq!(p1.1, 2.0);
        assert_eq!(p1.2, 2.0);

        assert_eq!(p2.0, 1.0);
        assert_eq!(p2.1, 3.0);
        assert_eq!(p2.2, 3.0);
    }

    #[test]
    fn conversion() {
        let v1 = f64x3(1.0, 2.0, 3.0);
        let v2: f32x3 = v1.into();

        assert_eq!(v2.0, 1.0);
        assert_eq!(v2.1, 2.0);
        assert_eq!(v2.2, 3.0);

        let v3 = f32x3(2.0, 3.0, 4.0);
        let v4 = f64x3::from(v3);

        assert_eq!(v4.0, 2.0);
        assert_eq!(v4.1, 3.0);
        assert_eq!(v4.2, 4.0);
    }
}
