use std::env;
use std::time::Instant;

use sunray::render;
use sunray::buffers::TMOType;


fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Not enough input arguments");
        return;
    }

    println!("Loading of {} ", &args[1]);
    let stop_watch = Instant::now(); 
    let scene = sunray::parse_input_file(&args[1]);
    let mut scene = match scene {
        Err(err) => {
            eprintln!("Problem parsing input file {}: {}", &args[1], err);
            return;
        },
        Ok(scene) => scene,
    };
    let loading_time = stop_watch.elapsed();
    println!("Loading of {} took {} seconds", &args[1], loading_time.as_secs_f64());
    println!("Preparing scene for rendering...");
    let stop_watch = Instant::now(); 
    scene.prepare();
    let prep_time = stop_watch.elapsed();
    println!("Prepare phase is finished - Time: {} seconds", prep_time.as_secs_f64());
    println!("Rendering...");
    let stop_watch = Instant::now();
    for pass in 0..scene.options.n_samples {
        render::render(&mut scene, pass);
        println!("Rendering {} pass: Time: {}", pass, stop_watch.elapsed().as_secs_f64());
    }
    let rendering_time = stop_watch.elapsed();
    println!("Rendering time: {} seconds", rendering_time.as_secs_f64());
    scene.color_buffer.to_rgb(TMOType::Reinhard).save(&scene.output_filename).unwrap();
}
