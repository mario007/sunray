use std::ops::Mul;
use crate::vec::f32x3;


#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Matrix4x4 {
    m: [[f32; 4]; 4]
}

impl Matrix4x4 {
    pub fn new() -> Matrix4x4 {
        Matrix4x4::identity()
    }

    pub fn identity() -> Matrix4x4 {
        Matrix4x4 {
            m: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        }
    }

    #[allow(clippy::float_cmp)]
    pub fn is_identity(&self) -> bool {
        self.m[0][0] == 1.0 && self.m[0][1] == 0.0 && self.m[0][2] == 0.0 && self.m[0][3] == 0.0 && 
        self.m[1][0] == 0.0 && self.m[1][1] == 1.0 && self.m[1][2] == 0.0 && self.m[1][3] == 0.0 &&
        self.m[2][0] == 0.0 && self.m[2][1] == 0.0 && self.m[2][2] == 1.0 && self.m[2][3] == 0.0 &&
        self.m[3][0] == 0.0 && self.m[3][1] == 0.0 && self.m[3][2] == 0.0 && self.m[3][3] == 1.0
    }

    pub fn from(v: [f32; 16]) -> Matrix4x4 {
        Matrix4x4 {
            m: [
                [v[0],  v[1],  v[2],  v[3]],
                [v[4],  v[5],  v[6],  v[7]],
                [v[8],  v[9],  v[10], v[11]],
                [v[12], v[13], v[14], v[15]]
            ]
        }
    }

    pub fn transpose(&self) -> Matrix4x4 {
        Matrix4x4 {
            m: [
                [self.m[0][0],   self.m[1][0], self.m[2][0], self.m[3][0]],
                [self.m[0][1],   self.m[1][1], self.m[2][1], self.m[3][1]],
                [self.m[0][2],   self.m[1][2], self.m[2][2], self.m[3][2]],
                [self.m[0][3],   self.m[1][3], self.m[2][3], self.m[3][3]],
            ]
        }
    }

    pub fn scale(x: f32, y: f32, z: f32) -> Matrix4x4 {
        Matrix4x4 {
            m: [
                [x,   0.0, 0.0, 0.0],
                [0.0, y,   0.0, 0.0],
                [0.0, 0.0, z,   0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        
        }
    }

    pub fn translate(delta_x: f32, delta_y: f32, delta_z:f32) -> Matrix4x4 {
        Matrix4x4 {
            m: [
                [1.0, 0.0, 0.0, delta_x],
                [0.0, 1.0, 0.0, delta_y],
                [0.0, 0.0, 1.0, delta_z],
                [0.0, 0.0, 0.0, 1.0]
            ]
        
        }
    }

    pub fn rotate_x(theta: f32) -> Matrix4x4 {

        let sin_theta = theta.to_radians().sin();
        let cos_theta = theta.to_radians().cos();

        Matrix4x4 {
            m: [
                [1.0, 0.0,          0.0,        0.0],
                [0.0, cos_theta,    -sin_theta, 0.0],
                [0.0, sin_theta,    cos_theta,  0.0],
                [0.0, 0.0,          0.0,        1.0]
            ]
        
        }
    }

    pub fn rotate_y(theta: f32) -> Matrix4x4 {

        let sin_theta = theta.to_radians().sin();
        let cos_theta = theta.to_radians().cos();

        Matrix4x4 {
            m: [
                [cos_theta,     0.0,    sin_theta,  0.0],
                [0.0,           1.0,    0.0,        0.0],
                [-sin_theta,    0.0,    cos_theta,  0.0],
                [0.0,           0.0,    0.0,        1.0]
            ]
        }
    }

    pub fn rotate_z(theta: f32) -> Matrix4x4 {

        let sin_theta = theta.to_radians().sin();
        let cos_theta = theta.to_radians().cos();

        Matrix4x4 {
            m: [
                [cos_theta,     -sin_theta, 0.0,    0.0],
                [sin_theta,     cos_theta,  0.0,    0.0],
                [0.0,           0.0,        1.0,    0.0],
                [0.0,           0.0,        0.0,    1.0]
            ]
        }
    }

    pub fn rotate_around_axis(theta: f32, x: f32, y: f32, z: f32) -> Matrix4x4 {
        let axis = f32x3(x, y, z).normalize();
        let sin_theta = theta.to_radians().sin();
        let cos_theta = theta.to_radians().cos();

        let mut m = Matrix4x4::new();
        // Compute rotation of first basis vector
        m.m[0][0] = axis.0 * axis.0 + (1.0 - axis.0 * axis.0) * cos_theta;
        m.m[0][1] = axis.0 * axis.1 * (1.0 - cos_theta) - axis.2 * sin_theta;
        m.m[0][2] = axis.0 * axis.2 * (1.0 - cos_theta) + axis.1 * sin_theta;
        m.m[0][3] = 0.0;

        // Compute rotations of second and third basis vectors
        m.m[1][0] = axis.0 * axis.1 * (1.0 - cos_theta) + axis.2 * sin_theta;
        m.m[1][1] = axis.1 * axis.1 + (1.0 - axis.1 * axis.1) * cos_theta;
        m.m[1][2] = axis.1 * axis.2 * (1.0 - cos_theta) - axis.0 * sin_theta;
        m.m[1][3] = 0.0;

        m.m[2][0] = axis.0 * axis.2 * (1.0 - cos_theta) - axis.1 * sin_theta;
        m.m[2][1] = axis.1 * axis.2 * (1.0 - cos_theta) + axis.0 * sin_theta;
        m.m[2][2] = axis.2 * axis.2 + (1.0 - axis.2 * axis.2) * cos_theta;
        m.m[2][3] = 0.0;
        m
    }

    pub fn look_at(pos: f32x3, look: f32x3, up: f32x3) -> Matrix4x4 {

        // left coordinate system

        // todo error logger - can be zero
        let dir = (look - pos).normalize();
        let right = up.normalize().cross(dir).normalize();
        let new_up = dir.cross(right);

        let mat = Matrix4x4 {
            m: [
                [right.0,   new_up.0,   dir.0,  pos.0],
                [right.1,   new_up.1,   dir.1,  pos.1],
                [right.2,   new_up.2,   dir.2,  pos.2],
                [0.0,       0.0,        0.0,    1.0]
            ]
        };
        // Note: we are returning WorldToCamera matrix
        mat.inverse()
    }

    pub fn perspective(fov: f32, near: f32, far: f32) -> Matrix4x4 {

        let persp = Matrix4x4 {
            m: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, far / (far - near), -far * near / (far - near)],
                [0.0, 0.0, 1.0, 0.0]
            ]
        };

        // Scale canonical perspective view to specified field of view
        let inv_tan = 1.0 / ((fov.to_radians() / 2.0).tan());
        Matrix4x4::scale(inv_tan, inv_tan, 1.0) * persp
    }

    #[allow(clippy::float_cmp)]
    pub fn transform_point(self, p: f32x3) -> f32x3 {
        let x = self.m[0][0] * p.0 + self.m[0][1] * p.1 + self.m[0][2] * p.2 + self.m[0][3];
        let y = self.m[1][0] * p.0 + self.m[1][1] * p.1 + self.m[1][2] * p.2 + self.m[1][3];
        let z = self.m[2][0] * p.0 + self.m[2][1] * p.1 + self.m[2][2] * p.2 + self.m[2][3];
        let w = self.m[3][0] * p.0 + self.m[3][1] * p.1 + self.m[3][2] * p.2 + self.m[3][3];

        if w == 1.0 {
            f32x3(x, y, z)
        } else {
            f32x3(x, y, z).div(f32x3(w, w, w))
        }
    }

    pub fn transform_vector(self, v: f32x3) -> f32x3 {
        let x = self.m[0][0] * v.0 + self.m[0][1] * v.1 + self.m[0][2] * v.2;
        let y = self.m[1][0] * v.0 + self.m[1][1] * v.1 + self.m[1][2] * v.2;
        let z = self.m[2][0] * v.0 + self.m[2][1] * v.1 + self.m[2][2] * v.2;
        f32x3(x, y, z)
    }

    pub fn inverse(self: Matrix4x4) -> Matrix4x4 {
        let a2323 = self.m[2][2] * self.m[3][3] - self.m[2][3] * self.m[3][2];
        let a1323 = self.m[2][1] * self.m[3][3] - self.m[2][3] * self.m[3][1];
        let a1223 = self.m[2][1] * self.m[3][2] - self.m[2][2] * self.m[3][1];
        let a0323 = self.m[2][0] * self.m[3][3] - self.m[2][3] * self.m[3][0];
        let a0223 = self.m[2][0] * self.m[3][2] - self.m[2][2] * self.m[3][0];
        let a0123 = self.m[2][0] * self.m[3][1] - self.m[2][1] * self.m[3][0];
        let a2313 = self.m[1][2] * self.m[3][3] - self.m[1][3] * self.m[3][2];
        let a1313 = self.m[1][1] * self.m[3][3] - self.m[1][3] * self.m[3][1];
        let a1213 = self.m[1][1] * self.m[3][2] - self.m[1][2] * self.m[3][1];
        let a2312 = self.m[1][2] * self.m[2][3] - self.m[1][3] * self.m[2][2];
        let a1312 = self.m[1][1] * self.m[2][3] - self.m[1][3] * self.m[2][1];
        let a1212 = self.m[1][1] * self.m[2][2] - self.m[1][2] * self.m[2][1];
        let a0313 = self.m[1][0] * self.m[3][3] - self.m[1][3] * self.m[3][0];
        let a0213 = self.m[1][0] * self.m[3][2] - self.m[1][2] * self.m[3][0];
        let a0312 = self.m[1][0] * self.m[2][3] - self.m[1][3] * self.m[2][0];
        let a0212 = self.m[1][0] * self.m[2][2] - self.m[1][2] * self.m[2][0];
        let a0113 = self.m[1][0] * self.m[3][1] - self.m[1][1] * self.m[3][0];
        let a0112 = self.m[1][0] * self.m[2][1] - self.m[1][1] * self.m[2][0];

        let det = self.m[0][0] * ( self.m[1][1] * a2323 - self.m[1][2] * a1323 + self.m[1][3] * a1223 )
        - self.m[0][1] * ( self.m[1][0] * a2323 - self.m[1][2] * a0323 + self.m[1][3] * a0223 )
        + self.m[0][2] * ( self.m[1][0] * a1323 - self.m[1][1] * a0323 + self.m[1][3] * a0123 )
        - self.m[0][3] * ( self.m[1][0] * a1223 - self.m[1][1] * a0223 + self.m[1][2] * a0123 );

        // TODO error logger det can be zero
        let det = 1.0 / det;

        let mut im = Matrix4x4::new();

        im.m[0][0] = det *   ( self.m[1][1] * a2323 - self.m[1][2] * a1323 + self.m[1][3] * a1223 );
        im.m[0][1] = det * - ( self.m[0][1] * a2323 - self.m[0][2] * a1323 + self.m[0][3] * a1223 );
        im.m[0][2] = det *   ( self.m[0][1] * a2313 - self.m[0][2] * a1313 + self.m[0][3] * a1213 );
        im.m[0][3] = det * - ( self.m[0][1] * a2312 - self.m[0][2] * a1312 + self.m[0][3] * a1212 );
        im.m[1][0] = det * - ( self.m[1][0] * a2323 - self.m[1][2] * a0323 + self.m[1][3] * a0223 );
        im.m[1][1] = det *   ( self.m[0][0] * a2323 - self.m[0][2] * a0323 + self.m[0][3] * a0223 );
        im.m[1][2] = det * - ( self.m[0][0] * a2313 - self.m[0][2] * a0313 + self.m[0][3] * a0213 );
        im.m[1][3] = det *   ( self.m[0][0] * a2312 - self.m[0][2] * a0312 + self.m[0][3] * a0212 );
        im.m[2][0] = det *   ( self.m[1][0] * a1323 - self.m[1][1] * a0323 + self.m[1][3] * a0123 );
        im.m[2][1] = det * - ( self.m[0][0] * a1323 - self.m[0][1] * a0323 + self.m[0][3] * a0123 );
        im.m[2][2] = det *   ( self.m[0][0] * a1313 - self.m[0][1] * a0313 + self.m[0][3] * a0113 );
        im.m[2][3] = det * - ( self.m[0][0] * a1312 - self.m[0][1] * a0312 + self.m[0][3] * a0112 );
        im.m[3][0] = det * - ( self.m[1][0] * a1223 - self.m[1][1] * a0223 + self.m[1][2] * a0123 );
        im.m[3][1] = det *   ( self.m[0][0] * a1223 - self.m[0][1] * a0223 + self.m[0][2] * a0123 );
        im.m[3][2] = det * - ( self.m[0][0] * a1213 - self.m[0][1] * a0213 + self.m[0][2] * a0113 );
        im.m[3][3] = det *   ( self.m[0][0] * a1212 - self.m[0][1] * a0212 + self.m[0][2] * a0112 );
        im
    }
}

impl Mul<Matrix4x4> for Matrix4x4 {
    type Output = Self;

    fn mul(self, m2: Matrix4x4) -> Self {
            let mut m = Matrix4x4::new();
            for i in 0..4 {
                for j in 0..4 {
                    m.m[i][j] = self.m[i][0] * m2.m[0][j] + self.m[i][1] * m2.m[1][j] +
                                self.m[i][2] * m2.m[2][j] + self.m[i][3] * m2.m[3][j];
            }
        }
        m
    }
}

// Note: to transform normal inverse matrix is needed
pub fn transform_normal(inv_matrix: &Matrix4x4, vec: f32x3) -> f32x3 {
    f32x3(inv_matrix.m[0][0] * vec.0 + inv_matrix.m[1][0] * vec.1 + inv_matrix.m[2][0] * vec.2,
          inv_matrix.m[0][1] * vec.0 + inv_matrix.m[1][1] * vec.1 + inv_matrix.m[2][1] * vec.2,
          inv_matrix.m[0][2] * vec.0 + inv_matrix.m[1][2] * vec.1 + inv_matrix.m[2][2] * vec.2)
}


#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn mul_matrix() {
        let m1 = Matrix4x4::identity();
        let m2 = Matrix4x4::identity();
        let m = m1 * m2;
        assert_eq!(m.m, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]);

        let m3 = Matrix4x4::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
        let m4 = Matrix4x4::from([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0]);
        let m = m3 * m4;
        assert_eq!(m.m, [[30.0, 30.0, 30.0, 30.0], [70.0, 70.0, 70.0, 70.0], [110.0, 110.0, 110.0, 110.0], [150.0, 150.0, 150.0, 150.0]]);

    }

    #[test]
    fn inverse_mat() {
        let m = Matrix4x4::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 0.0, 5.0, 16.0]);
        let inv = m.inverse();
        assert_eq!(inv.m, [[-0.4047619, -0.14285715, 0.16666667, 0.04761905], [-0.39285716, 0.71428573, -0.25, -0.071428575],
                        [0.5, -1.0, 0.5, 0.0], [0.17261904, 0.42857143, -0.2916667, 0.023809524]])
    }

    #[test]
    fn transpose_mat() {
        let m = Matrix4x4::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
        let tran = m.transpose();
        assert_eq!(tran.m,[[1.0, 5.0, 9.0, 13.0], [2.0, 6.0, 10.0, 14.0], [3.0, 7.0, 11.0, 15.0], [4.0, 8.0, 12.0, 16.0]])
    }
}