// PCG random number generator
// See http://www.pcg-random.org/


pub struct PCGRandom {
    state: u64,
    sequence: u64,
}

impl PCGRandom {
    pub fn new(state: u64, sequence: u64) -> PCGRandom {
        PCGRandom{state, sequence}
    }

    pub fn random_u32(&mut self) -> u32 {

        let old_state = self.state;
        self.state = old_state.wrapping_mul(6364136223846793005u64) + (self.sequence | 1);
        let xor_shifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot: u32 = (old_state >> 59) as u32;
        (xor_shifted >> rot) | (xor_shifted << ((-(rot as i32) as u32) & 31))
    }

    pub fn random_f32(&mut self) -> f32 {
        self.random_u32() as f32 * (1.0 / (1u64<<32) as f32)
    }
}


#[cfg(test)]
mod tests {

    use super::*;
    use std::time::Instant;

    #[test]
    fn pcg_test() {
        let mut rng = PCGRandom::new(8000000000000000, 0);
        for _i in 0..32 {
            println!("{}", rng.random_f32());
        }
    }

    #[test]
    fn pcg_speed() {
        let mut accum: f32 = 0.0;
        let stopwatch = Instant::now();
        let mut rng = PCGRandom::new(8000000000000000, 500000);
        for _i in 0..25_000_000 {
            accum += rng.random_f32();
        }
        let t1 = stopwatch.elapsed();
        println!("Accum {} Speed {} ", accum, t1.as_secs_f64());
    }
}