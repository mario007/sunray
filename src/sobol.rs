use crate::sobol_matrices;


pub struct SobolSequences {
    size: u32,
    n_dimensions: u32,
    matrices: &'static[u32],
}

impl SobolSequences {
    pub fn new() -> SobolSequences {
        SobolSequences {
            size: 52,
            n_dimensions: 1024,
            matrices: sobol_matrices::sobol_matrices()
        }
    }
    
    pub fn sample_f32(&self, index: u64, dimension: u32, scramble: u32) -> f32 {

        let result = self.sample_u32(index, dimension, scramble);
        result as f32 * (1.0 / (1u64<<32) as f32)
    }

    pub fn sample_u32(&self, index: u64, dimension: u32, scramble: u32) -> u32 {
        if dimension > self.n_dimensions {
            panic!("Maximal number of sobol dimensions is {} - got {} ", self.n_dimensions, dimension)
        }

        let mut result = scramble;
        let mut i = dimension * self.size;
        let mut index = index;

        while index != 0 {
            if index & 1 != 0 {
                result ^= self.matrices[i as usize];
            }

            index = index >> 1;
            i+=1;
        };
        result
    }
}


#[cfg(test)]
mod tests {

    use super::*;
    use std::time::Instant;

    #[test]
    fn sobol_test() {
        let sobol = SobolSequences::new();
        let mut samples = vec![];
        for i in 0..16 {
            for j in 0..2 {
                samples.push(sobol.sample_f32(i, j, 0));
            }
        }
        let expected = vec![0.0, 0.0, 0.5, 0.5, 0.25, 0.75, 0.75, 0.25, 0.125, 0.625, 0.625, 0.125, 0.375, 0.375, 0.875, 0.875, 0.0625,
            0.9375, 0.5625, 0.4375, 0.3125, 0.1875, 0.8125, 0.6875, 0.1875, 0.3125, 0.6875, 0.8125, 0.4375, 0.5625, 0.9375, 0.0625];
        
        assert_eq!(samples, expected);
    }

    #[test]
    fn sobol_speed() {
        let mut accum: f32 = 0.0;
        let stopwatch = Instant::now();
        let sobol = SobolSequences::new();
        for i in 0..16_000 {
            //let sobol = SobolSequences::new();
            accum += sobol.sample_f32(i as u64, 5, 262255);
        }
        let t1 = stopwatch.elapsed();
        println!("Accum {} Speed {} ", accum, t1.as_secs_f64());
    }
}
