use std::thread;
use std::sync::mpsc;
use std::sync::Arc;
use std::mem::drop;

use crate::buffers::{ColorBuffer, TMOType};
use crate::scene::{Scene, IntegratorType};
use crate::sampler::PathSampler;
use crate::render;


#[derive(Debug, Copy, Clone, PartialEq)]
struct Tile {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl Tile {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {x, y, width, height}
    }
}

struct AdaptiveTiles {
    tiles: Vec<Tile>,
    width: u32,
    height: u32,
    pub tiles_to_render: Arc<Vec<Vec<Tile>>>
}

impl AdaptiveTiles {
    pub fn new() -> Self {
        Self {tiles: Vec::new(), width: 200, height: 200, tiles_to_render: Arc::new(Vec::new())}
    }

    pub fn create_tiles(&mut self, width: u32, height: u32, tile_size: u32) {
        self.width = width;
        self.height = height;
        // TODO Initial block/blocks??
        //self.tiles.push(Tile::new(0, 0, width, height));
        for y in (0..height).step_by(tile_size as usize) {
            for x in (0..width).step_by(tile_size as usize) {
                let mut xsize = tile_size;
                if x + tile_size > width {
                    xsize = width - x;
                }
                let mut ysize = tile_size;
                if y + tile_size > height {
                    ysize = height - y;
                }
                self.tiles.push(Tile::new(x, y, xsize, ysize))
            }
        }
    }

    pub fn update_tiles(&mut self, color_buf_a: &ColorBuffer, color_buf_b: &ColorBuffer) {
        // NOTE: Default values from paper
        //let terminate_error = 0.0002; // user parameter - quality
        //let split_error = 256.0 * terminate_error;

        let terminate_error = 0.0001; // user parameter - quality
        let split_error = 0.005;

        let mut new_tiles = Vec::<Tile>::new();
        for tile in &self.tiles {
            let error = self.tile_error(tile, color_buf_a, color_buf_b);
            if error < terminate_error {
                continue;
            } else if error < split_error && (tile.width > 8 || tile.height > 8) {
                let (t1, t2) = self.split_tile(tile, color_buf_a, color_buf_b);
                new_tiles.push(t1);
                new_tiles.push(t2);
            } else {
                new_tiles.push(*tile);
            }
        }
        self.tiles = new_tiles;
    }

    fn split_tile(&self, tile: &Tile, _color_buf_a: &ColorBuffer, _color_buf_b: &ColorBuffer) -> (Tile, Tile) {
        
        if tile.width > tile.height {
            // TODO similar error for new blocks?

            /*let mut min_width = tile.width / 2;
            let mut min_error = 1.0;

            for i in 1..tile.width - 1 {
                let xwidth = i;
                let t1 = Tile::new(tile.x, tile.y, xwidth, tile.height);
                let t2 = Tile::new(tile.x + xwidth, tile.y, tile.width - xwidth, tile.height);
                let error1 = self.tile_error(&t1, color_buf_a, color_buf_b);
                let error2 = self.tile_error(&t2, color_buf_a, color_buf_b);
                let diff = (error2 - error1).abs();
                if diff < min_error {
                    min_width = i;
                    min_error = diff;
                }
            }*/
            let xwidth = tile.width / 2;
            //let xwidth = min_width;
            let t1 = Tile::new(tile.x, tile.y, xwidth, tile.height);
            let t2 = Tile::new(tile.x + xwidth, tile.y, tile.width - xwidth, tile.height);
            return (t1, t2);
        } else {
            // TODO similar error for new blocks?

            /*let mut min_height = tile.height / 2;
            let mut min_error = 1.0;

            for i in 1..tile.height - 1 {
                let ywidth = i;
                let t1 = Tile::new(tile.x, tile.y, tile.width, ywidth);
                let t2 = Tile::new(tile.x, tile.y + ywidth, tile.width, tile.height - ywidth);
                let error1 = self.tile_error(&t1, color_buf_a, color_buf_b);
                let error2 = self.tile_error(&t2, color_buf_a, color_buf_b);
                let diff = (error2 - error1).abs();
                if diff < min_error {
                    min_height = i;
                    min_error = diff;
                }
            }*/

            let ywidth = tile.height / 2;
            //let ywidth = min_height;
            let t1 = Tile::new(tile.x, tile.y, tile.width, ywidth);
            let t2 = Tile::new(tile.x, tile.y + ywidth, tile.width, tile.height - ywidth);
            return (t1, t2);
        }
    }

    fn tile_error(&self, tile: &Tile, color_buf_a: &ColorBuffer, color_buf_b: &ColorBuffer) -> f32 {
        let mut tile_error = 0.0;
        for y in tile.y..tile.y + tile.height {
            for x in tile.x..tile.x + tile.width {
                let (r1, g1, b1) = color_buf_a.get_color(x as usize, y as usize);
                let (r2, g2, b2) = color_buf_b.get_color(x as usize, y as usize);
                let denom = (r1 + g1 + b1).sqrt();
                let mut pix_err = 0.0;
                if denom > 0.0 {
                    pix_err = (r1 - r2).abs() + (g1 - g2).abs() + (b1 - b2).abs() / denom;
                }
                tile_error += pix_err;
            }
        }
        let r = ((tile.width * tile.height) as f32 / (self.width * self.height) as f32).sqrt();
        tile_error = tile_error * (r / (tile.width * tile.height) as f32);
        return tile_error;
    }

    fn split_to_small_tiles(&self, tile: &Tile) -> Vec<Tile> {
        let mut small_tiles = Vec::<Tile>::new();
        let tile_size = 16u32;
        for y in (tile.y..tile.y + tile.height).step_by(tile_size as usize) {
            for x in (tile.x..tile.x + tile.width).step_by(tile_size as usize) {
                let mut xsize = tile_size;
                if x + tile_size > tile.x + tile.width {
                    xsize = tile.x + tile.width - x;
                }
                let mut ysize = tile_size;
                if y + tile_size > tile.y + tile.height {
                    ysize = tile.y + tile.height - y;
                }
                small_tiles.push(Tile::new(x, y, xsize, ysize));
            }
        }
        small_tiles
    }

    pub fn generate_tiles_to_render(&mut self, nthreads: u32) {

        let mut small_tiles = Vec::<Tile>::new();
        for tile in &self.tiles {
            small_tiles.extend(self.split_to_small_tiles(tile));
        }

        let mut vec: Vec<Vec<Tile>> = Vec::new();
        for i in 0..nthreads as usize {
            let mut thread_tiles = Vec::new();
            for tile_idx in (i..small_tiles.len()).step_by(nthreads as usize) {
                thread_tiles.push(small_tiles[tile_idx]);
            }
            vec.push(thread_tiles);
        }
        self.tiles_to_render = Arc::new(vec);
    }
}


struct Pixel {
    x: u32,
    y: u32,
    r: f32,
    g: f32,
    b: f32,
}

impl Pixel {
    pub fn new(x: u32, y: u32, r: f32, g: f32, b: f32) -> Self {
        Self {x, y, r, g, b}
    }
}


pub struct Renderer {
    pub scene: Scene,
    nthreads: u32,
    adaptive_sampling: bool,
    pub rendering_pass: u32,
    adaptive_color_buffer: ColorBuffer,
    adaptive_tiles: AdaptiveTiles,
}

impl Renderer {

    pub fn new(scene: Scene) -> Self {
        let nthreads = 16; // TODO detect number of threads
        let adaptive_sampling = false;
        let rendering_pass = 0;
        let adaptive_color_buffer = ColorBuffer::new(200, 200);
        let adaptive_tiles = AdaptiveTiles::new();
        Self {scene, nthreads, adaptive_sampling, rendering_pass, adaptive_color_buffer, adaptive_tiles}
    }

    pub fn prepare(&mut self) {
        self.scene.prepare();
        self.rendering_pass = 0;
        let xres = self.scene.options.xres;
        let yres = self.scene.options.yres;
        self.adaptive_color_buffer = ColorBuffer::new(xres as usize, yres as usize);
        self.adaptive_tiles.create_tiles(xres, yres, 64);
    }

    pub fn render(&mut self) -> bool {
        if self.adaptive_sampling && self.rendering_pass > 9 {
            if self.rendering_pass > 2048 {
                return true;
            }
            return self.render_adaptive()
        } else {
            return self.render_regular()
        }
    }

    fn render_regular(&mut self) -> bool {
        if self.rendering_pass >= self.scene.options.n_samples {
            return true;
        }
        self.render_pass();
        self.rendering_pass >= self.scene.options.n_samples
    }

    fn render_adaptive(&mut self) -> bool {
        if self.adaptive_tiles.tiles.len() == 0 {
            return true;
        }

        self.adaptive_tiles.update_tiles(&self.scene.color_buffer, &self.adaptive_color_buffer);
        self.adaptive_tiles.generate_tiles_to_render(self.nthreads);
        for _i in 0..10 {
            self.render_adaptive_pass();
        }
        self.adaptive_tiles.tiles.len() == 0
    }

    fn render_pass(&mut self) {

        let (tx, rx) = mpsc::channel();

        for cur_y_value in 0..self.nthreads {
            let start_y = cur_y_value;
            let t_sender = tx.clone();
            let options = Arc::clone(&self.scene.options);
            let camera = Arc::clone(&self.scene.camera);
            let scene_data = Arc::clone(&self.scene.scene_data);
            
            let step = self.nthreads as usize;
            let rendering_pass = self.rendering_pass;

            thread::spawn(move || {
                let seed = start_y as u64 * 123456789 + 123456 * rendering_pass as u64;
                let mut sampler = PathSampler::new(options.sampler_type, options.xres, options.yres, options.n_samples, seed);
                for y in (start_y..options.yres).step_by(step) {
                    for x in 0..options.xres {
                        let (xp, yp) = sampler.sample_pixel(x, y, rendering_pass);
                        let ray = camera.generate_ray(x as f32 + xp, y as f32 + yp);
                        let rad = match options.integrator_type {
                        IntegratorType::DirectLighting => render::radiance_direct_lgt(&ray, &scene_data, &mut sampler),
                        IntegratorType::PathTracer => render::radiance_path_tracer(&ray, &scene_data, &mut sampler),
                        IntegratorType::Isect => render::radiance_isect(&ray, &scene_data, &mut sampler),
                    };
                    let pixel = Pixel::new(x, y, rad.0, rad.1, rad.2);
                    t_sender.send(pixel).expect("Pixel value not send!");
                    }
                }
            });
        }

        drop(tx);
        for pix in rx {
            self.scene.color_buffer.add_color(pix.x as usize, pix.y as usize, pix.r, pix.g, pix.b, 1.0);
            if self.rendering_pass % 2 == 0 {
                self.adaptive_color_buffer.add_color(pix.x as usize, pix.y as usize, pix.r, pix.g, pix.b, 1.0)
            }
        }
        self.rendering_pass += 1;
    }

    fn render_adaptive_pass(&mut self) {

        let (tx, rx) = mpsc::channel();

        for thread_id in 0..self.nthreads {
            let thread_tiles = Arc::clone(&self.adaptive_tiles.tiles_to_render);
            let t_sender = tx.clone();
            let options = Arc::clone(&self.scene.options);
            let camera = Arc::clone(&self.scene.camera);
            let scene_data = Arc::clone(&self.scene.scene_data);

            let rendering_pass = self.rendering_pass;

            thread::spawn(move || {
                let seed = thread_id as u64 * 123456789 + 123456 * rendering_pass as u64;
                let mut sampler = PathSampler::new(options.sampler_type, options.xres, options.yres, options.n_samples, seed);
                for tile in &thread_tiles[thread_id as usize] {
                    for y in tile.y..tile.y + tile.height {
                        for x in tile.x..tile.x + tile.width {
                            let (xp, yp) = sampler.sample_pixel(x, y, rendering_pass);
                            let ray = camera.generate_ray(x as f32 + xp, y as f32 + yp);
                            let rad = match options.integrator_type {
                                IntegratorType::DirectLighting => render::radiance_direct_lgt(&ray, &scene_data, &mut sampler),
                                IntegratorType::PathTracer => render::radiance_path_tracer(&ray, &scene_data, &mut sampler),
                                IntegratorType::Isect => render::radiance_isect(&ray, &scene_data, &mut sampler),
                            };
                            let pixel = Pixel::new(x, y, rad.0, rad.1, rad.2);
                            t_sender.send(pixel).expect("Pixel value not send!");
                        }
                    }
                }
            });
        }

        drop(tx);
        for pix in rx {
            self.scene.color_buffer.add_color(pix.x as usize, pix.y as usize, pix.r, pix.g, pix.b, 1.0);
            if self.rendering_pass % 2 == 0 {
                self.adaptive_color_buffer.add_color(pix.x as usize, pix.y as usize, pix.r, pix.g, pix.b, 1.0)
            }
        }
        self.rendering_pass += 1;
    }

    pub fn save_image(&self, tmo_type: TMOType) {
        self.scene.color_buffer.save(&self.scene.output_filename, tmo_type).unwrap();
    }
}
