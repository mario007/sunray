use crate::pcg::PCGRandom;
use crate::sobol::SobolSequences;


pub struct UniformSampler {
    pcg_rng: PCGRandom,
}

impl UniformSampler {
    pub fn new(xres: u32, yres: u32, seed: u64) -> UniformSampler {
        let pcg_rng = PCGRandom::new(seed, (xres * yres) as u64);
        UniformSampler{pcg_rng}
    }

    pub fn next_1d(&mut self) -> f32 {
        self.pcg_rng.random_f32()
    }

    pub fn next_2d(&mut self) -> (f32, f32) {
        (self.next_1d(), self.next_1d())
    }

    pub fn sample_pixel(&mut self, _x: u32, _y: u32, _rendering_pass: u32) -> (f32, f32) {
        self.next_2d()
    }

}

fn laine_karras_permutation(x: u32, seed: u32) -> u32
{
    let x = x.wrapping_add(seed);
    let x = x ^ (x.wrapping_mul(0x6c50b47cu32));
    let x = x ^ (x.wrapping_mul(0xb82f1e52u32));
    let x = x ^ (x.wrapping_mul(0xc7afe638u32));
    let x = x ^ (x.wrapping_mul(0x8d22f6e6u32));
    return x;
}

fn nested_uniform_scramble(x: u32, seed: u32) -> u32
{
    let x = x.reverse_bits();
    let x = laine_karras_permutation(x, seed);
    let x = x.reverse_bits();
    return x;
}

fn hash_combine(seed: u32, v: u32) -> u32 {
    return seed ^ (v.wrapping_add(seed << 6).wrapping_add(seed >> 2));
}

pub struct SobolSampler {
    sobol_sequence: SobolSequences,
    uniform_sampler: UniformSampler,
    dimension: u32,
    index: u64,
    scramble: u32,
    user_seed: u32,
    cur_seed: u32,
}

impl SobolSampler {
    pub fn new(xres: u32, yres: u32, seed: u64) -> SobolSampler {
        let sobol_sequence = SobolSequences::new();
        let uniform_sampler = UniformSampler::new(xres, yres, seed);
        SobolSampler{sobol_sequence, uniform_sampler, dimension: 0, index: 0, scramble: 0, user_seed: 1, cur_seed: 0}
    }

    pub fn next_1d(&mut self) -> f32 {
        if self.dimension > 1024 {
            return self.uniform_sampler.next_1d();
        }

        let index = nested_uniform_scramble(self.index as u32, self.cur_seed);
        let sample = self.sobol_sequence.sample_u32(index as u64, self.dimension, self.scramble);
        let sample = nested_uniform_scramble(sample, hash_combine(self.cur_seed, self.dimension));
        self.dimension += 1;
        sample as f32 * (1.0 / (1u64<<32) as f32)
    }

    pub fn next_2d(&mut self) -> (f32, f32) {
        (self.next_1d(), self.next_1d())
    }

    pub fn sample_pixel(&mut self, x: u32, y: u32, rendering_pass: u32) -> (f32, f32) {
        self.dimension = 0;
        self.cur_seed = x ^ (y << 16) ^ self.user_seed.wrapping_mul(0x736caf6f);
        self.index = rendering_pass as u64;
        self.next_2d()
    }
}

pub enum PathSampler {
    Uniform(UniformSampler),
    Sobol(SobolSampler)
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum SamplerType {
    Uniform,
    Sobol
}

impl PathSampler {
    pub fn new(sampler_type: SamplerType, xres: u32, yres: u32, _n_samples: u32, seed: u64) -> PathSampler {
        match sampler_type {
            SamplerType::Uniform => PathSampler::Uniform(UniformSampler::new(xres, yres, seed)),
            SamplerType::Sobol => PathSampler::Sobol(SobolSampler::new(xres, yres, seed))
        }
    }

    pub fn sample_pixel(&mut self, x: u32, y: u32, rendering_pass: u32) -> (f32, f32) {
        match self {
            PathSampler::Uniform(uniform) => uniform.sample_pixel(x, y, rendering_pass),
            PathSampler::Sobol(sobol) => sobol.sample_pixel(x, y, rendering_pass),
        }
    }

    pub fn next_1d(&mut self) -> f32 {
        match self {
            PathSampler::Uniform(uniform) => uniform.next_1d(),
            PathSampler::Sobol(sobol) => sobol.next_1d(),
        }
    }
}


#[cfg(test)]
mod tests {

    use super::*;
    use crate::buffers::ColorBufferRGB;

    #[test]
    fn sampling_pixels() {
        let mut sampler = PathSampler::new(SamplerType::Sobol, 4, 4, 4, 10_000_000);
        let mut col_buffer = ColorBufferRGB::new(512, 512);
        for pass in 0..16 {
            for j in 0..16 {
                for i in 0..16 {
                    let (x, y) = sampler.sample_pixel(i, j, pass);
                    let xi = i * 32 + (x * 32.0) as u32;
                    let yi = j * 32 + (y * 32.0) as u32;
                    col_buffer.set_color(xi as usize, yi as usize, 255, 0, 0);
                    //println!("x:{} y:{} = {}  {}", i, j, x, y);
                }
            }
        }
        col_buffer.save("samples.png").unwrap();
    }
}
