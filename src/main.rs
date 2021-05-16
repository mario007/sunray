use std::env;
use std::time::Instant;

use sunray::buffers::TMOType;
use sunray::renderer::Renderer;


fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Not enough input arguments");
        return;
    }

    println!("Loading of {} ", &args[1]);
    let stop_watch = Instant::now(); 
    let scene = sunray::parse_input_file(&args[1]);
    let scene = match scene {
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
    let mut ren = Renderer::new(scene);
    ren.prepare();
    let prep_time = stop_watch.elapsed();
    println!("Prepare phase is finished - Time: {} seconds", prep_time.as_secs_f64());
    println!("Rendering...");
    let stop_watch = Instant::now();
    while ren.render() == false {
        println!("Rendering {} pass: Time: {}", ren.rendering_pass, stop_watch.elapsed().as_secs_f64());
    }
    let rendering_time = stop_watch.elapsed();
    println!("Rendering time: {} seconds", rendering_time.as_secs_f64());
    ren.save_image(TMOType::Gamma);
}
