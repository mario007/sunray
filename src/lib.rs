mod vec;
mod scene;
mod matrix;
mod camera;
mod sampler;
mod sobol_matrices;
mod sobol;
mod pcg;
pub mod buffers;
mod ray;
pub mod render;
mod isects;
mod shapes;
mod materials;
mod lights;
mod math;
mod ply_reader;
mod grid;
mod scene_data;
mod sampling;
mod bsdf;
pub mod renderer;
mod spectrum;
mod texture;
mod image_io;

use std::error::Error;
use std::fs;
use std::collections::{HashSet, HashMap};
use std::str::FromStr;
use std::fmt::Display;
use std::path::{Path, PathBuf};

use crate::matrix::Matrix4x4;
use crate::sampler::SamplerType;
use crate::materials::{Material, MatteMaterial, PhongMaterial, WardMaterial, MetalMaterial, ConductorParams, MicrofacetDistType};
use crate::shapes::{Sphere, ShapeInstance, TransformShape, Mesh, ShapeType};
use crate::camera::{Camera, PerspectiveCamera};
use crate::lights::{Light, PointLight, AreaLight, DistantLight};

use crate::vec::f32x3;
use crate::scene::{Scene, IntegratorType};


struct AreaLightInfo {
    radiance: f32x3,
    type_name: String,
}

struct ParseState {
    matrices: Vec<Matrix4x4>,
    general_section: bool,
    materials_ids: Vec<u32>,
    area_lights_infos: Vec<AreaLightInfo>,
    path: PathBuf,
    named_materials: Vec<HashMap<String, u32>>,
}

impl ParseState {
    pub fn new() -> Self {
        let matrices = vec![Matrix4x4::identity()];
        let materials_ids = vec![0];
        let info = AreaLightInfo{type_name: "".to_string(), radiance: f32x3(0.0, 0.0, 0.0)};
        let area_lights_infos = vec![info];
        let named_materials = vec![HashMap::new()];

        Self {
            matrices,
            general_section: true,
            materials_ids,
            area_lights_infos,
            path: PathBuf::new(),
            named_materials,
        }
    }

    pub fn push_state(&mut self) {
        self.matrices.push(self.cur_matrix());
        self.materials_ids.push(self.cur_material());
        let info = AreaLightInfo{type_name: "".to_string(), radiance: f32x3(0.0, 0.0, 0.0)};
        self.area_lights_infos.push(info);

        // TODO - copy on write approach for performances reasons
        let index = self.named_materials.len() - 1;
        let map = self.named_materials[index].clone();
        self.named_materials.push(map);
    }

    pub fn cur_matrix(&self) -> Matrix4x4 {
        self.matrices[self.matrices.len() - 1]
    }

    pub fn cur_material(&self) -> u32 {
        //self.materials_ids[self.materials_ids.len() - 1]
        *self.materials_ids.last().expect("No material exist!")
    }

    pub fn cur_area_light_type(&self) -> &str {
        &self.area_lights_infos[self.area_lights_infos.len() - 1].type_name
    }

    pub fn cur_area_light_radiance(&self) -> f32x3 {
        self.area_lights_infos[self.area_lights_infos.len() - 1].radiance
    }

    pub fn set_matrix(&mut self, matrix: Matrix4x4) {
        let index = self.matrices.len() - 1;
        self.matrices[index] = matrix;
    }

    pub fn set_material(&mut self, material_id: u32) {
        let index = self.materials_ids.len() - 1;
        self.materials_ids[index] = material_id;
    }

    pub fn set_area_light_info(&mut self, type_name: String, radiance: f32x3) {
        let index = self.area_lights_infos.len() - 1;
        self.area_lights_infos[index].type_name = type_name;
        self.area_lights_infos[index].radiance = radiance;
    }

    pub fn pop_state(&mut self) {
        self.matrices.pop();
        self.materials_ids.pop();
        self.area_lights_infos.pop();
        self.named_materials.pop();
    }

    pub fn set_in_general_section(&mut self, value: bool) {
        self.general_section = value;
    }

    pub fn set_path(&mut self, path: PathBuf) {
        self.path = path;
    }

    pub fn add_named_material(&mut self, name: String, material_id: u32) {
        let index = self.named_materials.len() - 1;
        let map = &mut self.named_materials[index];
        // Todo - logger - if material allready exist it will be redefined
        map.insert(name, material_id);
    }

    pub fn get_named_material(&self, name: &str) -> u32 {
        let index = self.named_materials.len() - 1;
        let map = &self.named_materials[index];
        *map.get(name).unwrap_or_else(|| panic!("Material {} doesn't exist!", name))
    }
}

pub fn parse_input_file(filename: &str) -> Result<Scene, Box<dyn Error>> {
    let contents = fs::read_to_string(filename)?;
    let mut scene = Scene::new();
    let mut state = ParseState::new();
    let mut path = PathBuf::new();
    path.push(filename.to_string());
    state.set_path(path);
    parse_input_string(&contents, &mut scene, &mut state)?;
    Ok(scene)
}

fn parse_input_string(text: &str, scene: &mut Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {

    let mut ct = PBRTTokenizer::new(text);
    let mut cur_directive = "";
    let all_directives: HashSet<_> = vec!["LookAt", "Camera", "Sampler", "Integrator", "Film", "PixelFilter",
        "WorldBegin", "WorldEnd", "AttributeBegin", "AttributeEnd", "LightSource", "AreaLightSource", "Texture",
        "Material", "MakeNamedMaterial", "NamedMaterial", "Include", "Accelerator", "Shape",
        "Scale", "Translate", "Rotate", "Identity", "Transform", "ConcatTransform"].into_iter().collect();
    let directives_to_process: HashSet<_> = vec!["LookAt", "Camera", "Sampler", "Integrator", "Film", "PixelFilter", 
        "WorldBegin", "WorldEnd", "AttributeBegin", "AttributeEnd", "LightSource", "AreaLightSource", "Texture",
        "Material", "MakeNamedMaterial", "NamedMaterial", "Accelerator",
        "Scale", "Translate", "Rotate", "Identity", "Transform", "ConcatTransform"].into_iter().collect();

    let mut fetch_token = true;
    loop {
        if fetch_token {
            cur_directive = match ct.next() {
                Some(token) => token.trim(),
                None => break
            };
        }
        fetch_token = true;
        if directives_to_process.contains(cur_directive) {
            let mut tokens = vec![cur_directive.to_string()];
            loop {
                let token = match ct.next() {
                    Some(token) => token.trim(),
                    None => break
                };

                if all_directives.contains(token) {
                    cur_directive = token;
                    fetch_token = false;
                    break;
                }
                tokens.push(token.to_string());
            }
            process_directive(&tokens, scene, state)?;
        } else if cur_directive == "Shape" {
            let token = match ct.next() {
                Some(token) => token,
                None => break
            };

            let mut tokens = vec![token.to_string()];
            if token == "trianglemesh" { 
                let next_directive = process_trianglemesh(&mut ct, scene, state)?;
                if all_directives.contains(next_directive.as_str()) {
                    let next_dir = all_directives.get(next_directive.as_str()).unwrap();
                    cur_directive = next_dir;
                    fetch_token = false;
                } else {
                    return Err(format!("Unknown directive: {}", next_directive).into())
                }
            } else {
                loop {
                    let token = match ct.next() {
                        Some(token) => token.trim(),
                        None => break
                    };
    
                    if all_directives.contains(token) {
                        cur_directive = token;
                        fetch_token = false;
                        break;
                    }
                    tokens.push(token.to_string());
                }
                process_shape(&tokens, scene, state)?;
            }

        } else if cur_directive == "Include" {
            let filename = match ct.next() {
                Some(token) => token,
                None => break
            };
            // TODO - Test this!
            let full_path = create_path(state, &filename.to_string());
            let contents = fs::read_to_string(full_path)?; 
            parse_input_string(&contents, scene, state)?;
        } else {
            return Err(format!("Unknown directive: {}", cur_directive).into())
        }
    }
    Ok(())
}

fn process_directive(tokens: &[String], scene: &mut Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    match &tokens[0].trim() as &str {
        "LookAt" => process_look_at(tokens, scene, state)?,
        "Camera" => process_camera(tokens, scene, state)?,
        "Integrator" => process_integrator(tokens, scene, state)?,
        "Film" => process_film(tokens, scene, state)?,
        "Sampler" => process_sampler(tokens, scene, state)?,
        "PixelFilter" => process_pixel_filter(tokens, scene, state)?,
        "WorldBegin" => process_world_begin(tokens, scene, state)?,
        "WorldEnd" => process_world_end(tokens, scene, state)?,
        "AttributeBegin" => process_attribute_begin(tokens, scene, state)?,
        "AttributeEnd" => process_attribute_end(tokens, scene, state)?,
        "LightSource" => process_light_source(tokens, scene, state)?,
        "AreaLightSource" => process_area_light_source(tokens, scene, state)?,
        "Texture" => process_texture(tokens, scene, state)?,
        "Material" => process_material(tokens, scene, state)?,
        "MakeNamedMaterial" => process_make_named_material(tokens, scene, state)?,
        "NamedMaterial" => process_named_material(tokens, scene, state)?,
        "Accelerator" => process_accelerator(tokens, scene, state)?,
        "Scale" => process_scale_transform(tokens, scene, state)?,
        "Translate" => process_translate_transform(tokens, scene, state)?,
        "Rotate" => process_rotate_transform(tokens, scene, state)?,
        "Identity" => process_identity_transform(tokens, scene, state)?,
        "Transform" => process_transform(tokens, scene, state)?,
        "ConcatTransform" => process_concat_transform(tokens, scene, state)?,
        _=> return Err(format!("Unsupported directive to process: {}", tokens[0]).into())
    }
    Ok(())
}

fn process_shape(tokens: &[String], scene: &mut Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {

    match &tokens[0] as &str {
        "sphere" => process_shape_sphere(tokens, scene, state)?,
        "plymesh" => process_shape_plymesh(tokens, scene, state)?,
        _=> return Err(format!("Unsupported shape to process: {}", tokens[0]).into())
    }
    Ok(())
}

fn process_trianglemesh(tokenizer: &mut PBRTTokenizer, scene: &mut Scene, state: &mut ParseState) -> Result<String, Box<dyn Error>> {
    let mut vertices: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut normals: Vec<f32> = Vec::new();
    let mut uv_coords: Vec<f32> = Vec::new();

    let mut next_directive = String::new();

    loop {
        let token = match tokenizer.next() {
            Some(token) => token,
            None => break,
        };
        if token == "integer indices" {
            indices = process_u32_values(tokenizer, "Triangle mesh: integer indices ")?;
        } else if token == "point P" {
            vertices = process_f32_values(tokenizer, "Triangle mesh: point P ")?;
        } else if token == "normal N" {
            normals = process_f32_values(tokenizer, "Triangle mesh: normal N ")?;
        } else if token == "float uv" {
            uv_coords = process_f32_values(tokenizer, "Triangle mesh: float uv ")?;
        }
        else {
            next_directive = token.trim().to_string();
            break;
        }
    }

    let mut mesh = Mesh::new();

    if indices.len() % 3 != 0 {
        return Err(format!("Number of mesh indices must be divisible by 3: {}", indices.len()).into())
    }

    for i in 0..indices.len() / 3 {
        mesh.add_indices(indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]);
    }

    if vertices.len() % 3 != 0 {
        return Err(format!("Number of mesh vertices must be divisible by 3: {}", vertices.len()).into())
    }

    for i in 0..vertices.len() / 3 {
        mesh.add_vertex(vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]);
    }

    if !normals.is_empty() && normals.len() != vertices.len() {
        return Err(format!("Mesh must have same number of vertices and normals: {} {}", vertices.len(), normals.len()).into())
    }

    for i in 0..normals.len() / 3 {
        mesh.add_vertex_normal(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
    }

    for i in 0..uv_coords.len() / 2 {
        mesh.add_uv(uv_coords[i * 2], uv_coords[i * 2 + 1]);
    }

    let material_id = state.cur_material();

    if state.cur_matrix().is_identity() {
        let mesh_inst = ShapeInstance::new(mesh, material_id);
        let id = scene.add_mesh(mesh_inst);
        add_diffuse_area_light(ShapeType::Mesh, id, scene, state)?;
    } else {
        let tran_mesh = TransformShape::new(mesh, state.cur_matrix());
        let id = scene.add_transformed_mesh(ShapeInstance::new(tran_mesh, material_id));
        add_diffuse_area_light(ShapeType::TransformMesh, id, scene, state)?;
    }
    
    Ok(next_directive)
}

fn process_u32_values(tokenizer: &mut PBRTTokenizer, err_msg: &str) -> Result<Vec<u32>, Box<dyn Error>> {
    let token = match tokenizer.next() {
        Some(token) => token,
        None => return Err(format!("{} '[' is expected", err_msg).into())
    };
    if token != "[" { return Err(format!("{} '[' is expected, got '{}'", err_msg, token).into()); }

    let mut values: Vec<u32> = Vec::new();
    loop {
        let token = match tokenizer.next() {
            Some(token) => token,
            None => return Err(format!("{} ']' is expected", err_msg).into())
        };
        if token == "]" { break; }
        values.push(parse_u32(token, err_msg)?)
    }
    Ok(values)
}

fn process_f32_values(tokenizer: &mut PBRTTokenizer, err_msg: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let token = match tokenizer.next() {
        Some(token) => token,
        None => return Err(format!("{} '[' is expected", err_msg).into())
    };
    if token != "[" { return Err(format!("{} '[' is expected, got '{}'", err_msg, token).into()); }

    let mut values: Vec<f32> = Vec::new();
    loop {
        let token = match tokenizer.next() {
            Some(token) => token,
            None => return Err(format!("{} ']' is expected", err_msg).into())
        };
        if token == "]" { break; }
        values.push(parse_f32(token, err_msg)?)
    }
    Ok(values)
}

fn process_shape_sphere(tokens: &[String], scene: &mut Scene, state: &ParseState) -> Result<(), Box<dyn Error>> {
    let radius = find_value("float radius", &tokens[1..], 1.0, "Sphere::radius - ")?;
    let position = find_f32x3("point position", &tokens[1..], f32x3(0.0, 0.0, 0.0), "Sphere:point:position - ")?;
    let sphere = Sphere::new(position, radius);
    let material_id = state.cur_material() as u32;
    
    if state.cur_matrix().is_identity() {
        let sphere_inst = ShapeInstance::new(sphere, material_id);
        let id = scene.add_sphere(sphere_inst);
        add_diffuse_area_light(ShapeType::Sphere, id, scene, state)?;
    } else {
        let tran_sphere = TransformShape::new(sphere, state.cur_matrix());
        let id = scene.add_transformd_sphere(ShapeInstance::new(tran_sphere, material_id));
        add_diffuse_area_light(ShapeType::TransformSphere, id, scene, state)?;
    }
    Ok(())
}

fn process_shape_plymesh(tokens: &[String], scene: &mut Scene, state: &ParseState) -> Result<(), Box<dyn Error>> {
    let filename = find_value("string filename", &tokens[1..], "".to_string(), "Plymesh::filename - ")?;
    if filename.is_empty() {
        return Err("Plymesh: filename expected!".into())
    }
    let mut mesh = Mesh::new();
    let full_path = create_path(state, &filename);
    let result = ply_reader::read_ply_file(&full_path, &mut mesh);
    if let Err(e) = result {
        return Err(format!("Loading of {} failed! {}", &full_path, e).into());
    }
    let material_id = state.cur_material();

    if state.cur_matrix().is_identity() {
        let mesh_inst = ShapeInstance::new(mesh, material_id);
        let id = scene.add_mesh(mesh_inst);
        add_diffuse_area_light(ShapeType::Mesh, id, scene, state)?;

    } else {
        let tran_mesh = TransformShape::new(mesh, state.cur_matrix());
        let id = scene.add_transformed_mesh(ShapeInstance::new(tran_mesh, material_id));
        add_diffuse_area_light(ShapeType::TransformMesh, id, scene, state)?;
    }
    result
}

fn create_path(state: &ParseState, filename: &str) -> String {
    if Path::new(filename).is_absolute() {
        return filename.to_string();
    }
    let full_path = match state.path.parent() {
        Some(dir) => dir.join(filename),
        None => PathBuf::new(),
    };
    return full_path.to_str().expect("Path conversion faild!").to_string();
}

fn process_look_at(tokens: &[String], _scene: &mut Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    if tokens.len() != 10 {
        return Err("LookAt: Exactly 9 values expected!".into())
    }
    let eye = parse_f32x3(&tokens[1], &tokens[2], &tokens[3], "LookAt:eye ")?;
    let look_at = parse_f32x3(&tokens[4], &tokens[5], &tokens[6], "LookAt:look_at ")?;
    let up = parse_f32x3(&tokens[7], &tokens[8], &tokens[9], "LookAt:up ")?;
    let matrix = state.cur_matrix() * Matrix4x4::look_at(eye, look_at, up);
    state.set_matrix(matrix);
    Ok(())
}

fn process_camera(tokens: &[String], scene: &mut Scene, state: &ParseState) -> Result<(), Box<dyn Error>> {
    if tokens.len() < 2 {
        return Err("Camera: Type of camera not specified!".into())
    }
    // TODO refactor to be like material and lights - each type of camera separate function
    if tokens[1] == "perspective" {
        let fov = find_value("float fov", &tokens[2..], 90.0, "Camera::fov - ")?;
        let (xres, yres) = scene.get_resolution();
        let mut camera = Camera::Perspective(PerspectiveCamera::new(xres, yres, fov));
        camera.set_camera_to_world(state.cur_matrix().inverse());
        scene.set_camera(camera);
        if has_parameter("float frameaspectratio", &tokens[2..]) {
            let aspect_ratio = find_value("float frameaspectratio", &tokens[2..], 1.0, "Camera::aspect_ratio - ")?;
            scene.set_camera_aspect_ratio(Some(aspect_ratio));
        }
    } else {
        return Err(format!("Camera: Unsupported camera type - {}", tokens[1]).into())
    }
    Ok(())
}

fn process_integrator(tokens: &[String], scene: &mut Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    if tokens.len() < 2 {
        return Err("Integrator: Type of integrator not specified!".into())
    }
    match &tokens[1] as &str {
        "directlighting" => process_integrator_direct_lgt(tokens, scene, state)?,
        "intersector" => process_integrator_isect(tokens, scene, state)?,
        "path" => process_integrator_path(tokens, scene, state)?,
        _=> return Err(format!("Unsupported integrator type {}", tokens[1]).into())
    }
    Ok(())
}

fn process_integrator_path(_tokens: &[String], scene: &mut Scene, _state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    scene.set_integrator_type(IntegratorType::PathTracer);
    Ok(())
}

fn process_integrator_direct_lgt(_tokens: &[String], scene: &mut Scene, _state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    scene.set_integrator_type(IntegratorType::DirectLighting);
    Ok(())
}

fn process_integrator_isect(_tokens: &[String], scene: &mut Scene, _state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    scene.set_integrator_type(IntegratorType::Isect);
    Ok(())
}

fn process_film(tokens: &[String], scene: &mut Scene, _state: &ParseState) -> Result<(), Box<dyn Error>> {
    if tokens.len() < 2 {
        return Err("Film: Type of film image not specified!".into())
    }
    if tokens[1] != "image" {
        return Err(format!("Film: Type 'image' expected, got {}", tokens[1]).into())
    }
    let filename = find_value("string filename", &tokens[2..], "output.png".to_string(), "Film::filename - ")?;
    let xres = find_value("integer xresolution", &tokens[2..], 200, "Film::xresolution - ")?;
    let yres = find_value("integer yresolution", &tokens[2..], 200, "Film::yresolution - ")?;
    scene.set_resolution(xres, yres);
    scene.set_output_filename(filename);
    Ok(())
}

fn process_sampler(tokens: &[String], scene: &mut Scene, _state: &ParseState) -> Result<(), Box<dyn Error>> {
    if tokens.len() < 2 {
        return Err("Sampler: Type of sampler not specified!".into())
    }

    let n_samples = find_value("integer pixelsamples", &tokens[2..], 16, "Sampler::pixelsamples - ")?;

    match &tokens[1] as &str {
        "sobol" => scene.set_sampler(SamplerType::Sobol, n_samples),
        "random" => scene.set_sampler(SamplerType::Uniform, n_samples),
        _ => scene.set_sampler(SamplerType::Sobol, n_samples)
    }
    Ok(())
}

fn process_pixel_filter(_tokens: &[String], _scene: &Scene, _state: &ParseState) -> Result<(), Box<dyn Error>> {
    Ok(())
}

fn process_world_begin(_tokens: &[String], _scene: &Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    state.set_matrix(Matrix4x4::identity());
    state.set_in_general_section(false);
    Ok(())
}

fn process_world_end(_tokens: &[String], _scene: &Scene, _state: &ParseState) -> Result<(), Box<dyn Error>> {
    Ok(())
}

fn process_attribute_begin(_tokens: &[String], _scene: &Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    state.push_state();
    Ok(())
}

fn process_attribute_end(_tokens: &[String], _scene: &Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    state.pop_state();
    Ok(())
}

fn process_light_source(tokens: &[String], scene: &mut Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    if tokens.len() < 2 {
        return Err("Light: Type of light not specified!".into())
    }
    match &tokens[1] as &str {
        "point" => process_point_light(tokens, scene, state)?,
        "distant" => process_distant_light(tokens, scene, state)?,
        _=> return Err(format!("Unsupported light type {}", tokens[1]).into())
    }
    Ok(())
}

fn process_area_light_source(tokens: &[String], scene: &mut Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    if tokens.len() < 2 {
        return Err("AreaLight: Type of area light not specified!".into())
    }
    match &tokens[1] as &str {
        "diffuse" => process_diffuse_area_light(tokens, scene, state)?,
        _=> return Err(format!("Unsupported area light type {}", tokens[1]).into())
    }
    Ok(())
}

fn process_texture(_tokens: &[String], _scene: &Scene, _state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    Ok(())
}

fn process_material(tokens: &[String], scene: &mut Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    if tokens.len() < 2 {
        return Err("Material: Type of material not specified!".into())
    }
    process_material_types(tokens, scene, state, &tokens[1] as &str, "")
}

fn process_make_named_material(tokens: &[String], scene: &mut Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    if tokens.len() < 4 {
        return Err("MakeNamedMaterial: At least 4 token required!".into())
    }
    let mat_name = &tokens[1].to_string();
    let mat_type = find_value("string type", tokens, "Unknown".to_string(), "MakeNamedMaterial:type - ")?;
    // Note: Just to be sure we remove first two tokens (MakeNamedMaterial "material_name") because material_name can have same name as some material parameter
    let tokens = Vec::from(&tokens[2..]);
    return process_material_types(&tokens, scene, state, mat_type.as_str(), mat_name);
}

fn process_named_material(tokens: &[String], _scene: &Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    if tokens.len() < 2 {
        return Err("NamedMaterial: At least 2 token required!".into())
    }
    let id = state.get_named_material(&tokens[1]);
    state.set_material(id);
    Ok(())
}

fn process_material_types(tokens: &[String], scene: &mut Scene, state: &mut ParseState, mat_type: &str, mat_name: &str) -> Result<(), Box<dyn Error>> {
    match mat_type {
        "matte" => process_matte_material(tokens, scene, state, mat_name)?,
        "phong" => process_phong_material(tokens, scene, state, mat_name)?,
        "ward" => process_ward_material(tokens, scene, state, mat_name)?,
        "metal" => process_metal_material(tokens, scene, state, mat_name)?,
        _=> return Err(format!("Unsupported material type {}", mat_type).into())
    }
    Ok(())
}

fn process_point_light(tokens: &[String], scene: &mut Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    let intensity = find_spectrum("I", &tokens[2..], f32x3(1.0, 1.0, 1.0), "Light:point:I - ")?;
    let scale = find_spectrum("scale", &tokens[2..], f32x3(1.0, 1.0, 1.0), "Light:point:scale - ")?;
    let intensity = intensity.mul(scale);
    let position = find_f32x3("point from", &tokens[2..], f32x3(0.0, 0.0, 0.0), "Light:point:from - ")?;
    let matrix = Matrix4x4::translate(position.0, position.1, position.2) * state.cur_matrix();
    let light_pos = matrix.transform_point(f32x3(0.0, 0.0, 0.0));
    let light = Light::Point(PointLight::new(light_pos, intensity));
    scene.add_light(light);
    Ok(())
}

fn process_distant_light(tokens: &[String], scene: &mut Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    let intensity = find_spectrum("L", &tokens[2..], f32x3(1.0, 1.0, 1.0), "Light:distant:L - ")?;
    let scale = find_spectrum("scale", &tokens[2..], f32x3(1.0, 1.0, 1.0), "Light:distant:scale - ")?;
    let intensity = intensity.mul(scale);

    let point_from = find_f32x3("point from", &tokens[2..], f32x3(0.0, 0.0, 0.0), "Light:distant:from - ")?;
    let point_to = find_f32x3("point to", &tokens[2..], f32x3(0.0, 0.0, 1.0), "Light:distant:to - ")?;

    let matrix_from = Matrix4x4::translate(point_from.0, point_from.1, point_from.2) * state.cur_matrix();
    let light_from = matrix_from.transform_point(f32x3(0.0, 0.0, 0.0));

    let matrix_to = Matrix4x4::translate(point_to.0, point_to.1, point_to.2) * state.cur_matrix();
    let light_to = matrix_to.transform_point(f32x3(0.0, 0.0, 0.0));

    let wi_light = (light_from - light_to).normalize();
    let light = Light::Distant(DistantLight::new(wi_light, intensity));
    scene.add_light(light);
    Ok(())
}

fn process_diffuse_area_light(tokens: &[String], _scene: &mut Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    let radiance = find_spectrum("L", &tokens[2..], f32x3(1.0, 1.0, 1.0), "AreaLight:diffuse:L - ")?;
    state.set_area_light_info("diffuse".to_string(), radiance);
    Ok(())
}

fn add_diffuse_area_light(shape_type: ShapeType, shape_id: u32, scene: &mut Scene, state: &ParseState) -> Result<(), Box<dyn Error>> {
    if state.cur_area_light_type() == "diffuse" {
        let light = Light::Area(AreaLight::new(shape_type, shape_id, state.cur_area_light_radiance()));
        let light_id = scene.add_light(light);
        scene.set_area_light(&shape_type, shape_id, light_id as i32);
    }
    Ok(())
}

fn process_accelerator(_tokens: &[String], _scene: &Scene, _state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    Ok(())
}

fn process_scale_transform(tokens: &[String], _scene: &Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    if tokens.len() != 4 {
        return Err("Scale Transform: Exactly 3 values expected!".to_string().into())
    }
    let scale = parse_f32x3(&tokens[1], &tokens[2], &tokens[3], "Transform:scale ")?;
    let matrix = state.cur_matrix() * Matrix4x4::scale(scale.0, scale.1, scale.2);
    state.set_matrix(matrix);
    Ok(())
}

fn process_translate_transform(tokens: &[String], _scene: &Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    if tokens.len() != 4 {
        return Err("Translate Transform: Exactly 3 values expected!".to_string().into())
    }
    let delta = parse_f32x3(&tokens[1], &tokens[2], &tokens[3], "Transform:translate ")?;
    let matrix = state.cur_matrix() * Matrix4x4::translate(delta.0, delta.1, delta.2);
    state.set_matrix(matrix);
    Ok(())
}

fn process_rotate_transform(tokens: &[String], _scene: &Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    if tokens.len() != 5 {
        return Err("Rotate Transform: Exactly 4 values expected!".to_string().into())
    }
    let err_msg = "Rotate Transform: ";
    let angle = parse_f32(&tokens[1], err_msg)?;
    let x = parse_f32(&tokens[2], err_msg)?;
    let y = parse_f32(&tokens[3], err_msg)?;
    let z = parse_f32(&tokens[4], err_msg)?;
    let matrix = state.cur_matrix() * Matrix4x4::rotate_around_axis(angle, x, y, z);
    state.set_matrix(matrix);
    Ok(())
}

fn process_identity_transform(_tokens: &[String], _scene: &Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    state.set_matrix(Matrix4x4::identity());
    Ok(())
}

fn process_transform(tokens: &[String], _scene: &Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    let values = extract_f32_values(&tokens[1..], "Transform: ")?;
    if values.len() != 16 {
        return Err("Transform: Exactly 16 values expected!".to_string().into())
    }
    let matrix: [f32; 16] = [values[0], values[4], values[8],  values[12],
                             values[1], values[5], values[9],  values[13],
                             values[2], values[6], values[10], values[14],
                             values[3], values[7], values[11], values[15]];
    state.set_matrix(Matrix4x4::from(matrix));
    Ok(())
}

fn process_concat_transform(tokens: &[String], _scene: &Scene, state: &mut ParseState) -> Result<(), Box<dyn Error>> {
    let values = extract_f32_values(&tokens[1..], "ConcatTransform: ")?;
    if values.len() != 16 {
        return Err("ConcatTransform: Exactly 16 values expected!".to_string().into())
    }
    let matrix: [f32; 16] = [values[0], values[4], values[8],  values[12],
                             values[1], values[5], values[9],  values[13],
                             values[2], values[6], values[10], values[14],
                             values[3], values[7], values[11], values[15]];
    let matrix = state.cur_matrix() * Matrix4x4::from(matrix);
    state.set_matrix(matrix);
    Ok(())
}

fn process_matte_material(tokens: &[String], scene: &mut Scene, state: &mut ParseState, mat_name: &str) -> Result<(), Box<dyn Error>> {
    let spec = find_spectrum("Kd", &tokens[2..], f32x3(0.5, 0.5, 0.5), "Material:mate:Kd - ")?;
    let roughness = find_value("float sigma", &tokens[2..], 0.0, "Material:mate:sigma - ")?;
    let mat = Material::Matte(MatteMaterial::new(spec, roughness));
    let id = scene.add_material(mat);
    scene.add_dyn_material(Box::new(MatteMaterial::new(spec, roughness)));
    add_material_to_state(id, mat_name, state);
    Ok(())
}

fn process_phong_material(tokens: &[String], scene: &mut Scene, state: &mut ParseState, mat_name: &str) -> Result<(), Box<dyn Error>> {
    let kd = find_spectrum("Kd", &tokens[2..], f32x3(0.5, 0.5, 0.5), "Material:phong:Kd - ")?;
    let ks = find_spectrum("Ks", &tokens[2..], f32x3(0.5, 0.5, 0.5), "Material:phong:Ks - ")?;
    let shininess = find_value("float shininess", &tokens[2..], 10.0, "Material:mate:shininess - ")?;
    let mat = Material::Phong(PhongMaterial::new(kd, ks, shininess));
    let id = scene.add_material(mat);
    scene.add_dyn_material(Box::new(PhongMaterial::new(kd, ks, shininess)));
    add_material_to_state(id, mat_name, state);
    Ok(())
}

fn process_ward_material(tokens: &[String], scene: &mut Scene, state: &mut ParseState, mat_name: &str) -> Result<(), Box<dyn Error>> {
    let kd = find_spectrum("Kd", &tokens[2..], f32x3(0.5, 0.5, 0.5), "Material:ward:Kd - ")?;
    let ks = find_spectrum("Ks", &tokens[2..], f32x3(0.5, 0.5, 0.5), "Material:ward:Ks - ")?;
    let ax = find_value("float ax", &tokens[2..], 0.15, "Material:ward:ax - ")?;
    let ay = find_value("float ay", &tokens[2..], 0.15, "Material:ward:ay - ")?;
    let mat = Material::Ward(WardMaterial::new(kd, ks, ax, ay));
    let id = scene.add_material(mat);
    scene.add_dyn_material(Box::new(WardMaterial::new(kd, ks, ax, ay)));
    add_material_to_state(id, mat_name, state);
    Ok(())
}

fn process_metal_material(tokens: &[String], scene: &mut Scene, state: &mut ParseState, mat_name: &str) -> Result<(), Box<dyn Error>> {
    let mut alpha = find_value("float roughness", &tokens[2..], 0.01, "Material:Conductor:roughness - ")?;
    if has_parameter("bool remaproughness", tokens) {
        let value = find_value("bool remaproughness", &tokens[2..], "".to_string(), "Material:Conductor::remaproughness - ")?;
        if value == "true" {
            alpha = bsdf::ggx_and_beckmann_roughness_to_alpha(alpha);
        }
    } else {
        alpha = bsdf::ggx_and_beckmann_roughness_to_alpha(alpha);
    }
    let con_params: ConductorParams;
    let con_params2: ConductorParams;
    // NOTE copper is default metal material
    if has_parameter("rgb F0", tokens) {
        let f0 = find_spectrum("F0", &tokens[2..], f32x3(0.955, 0.638, 0.538), "Material:Metal:F0 - ")?;
        con_params = ConductorParams::F0(f0);
        con_params2 = ConductorParams::F0(f0);
    } else {
        let eta = find_spectrum("eta", &tokens[2..], f32x3(0.21258, 0.8231, 1.2438), "Material:Metal:eta - ")?;
        let k = find_spectrum("k", &tokens[2..], f32x3(4.1003, 2.4763, 2.288), "Material:Metal:k - ")?;
        con_params = ConductorParams::IOR(eta, k);
        con_params2 = ConductorParams::IOR(eta, k);
    }
    let mut dist = MicrofacetDistType::GGX;
    let mut dist2 = MicrofacetDistType::GGX;
    if has_parameter("string distribution", tokens) {
        let value = find_value("string distribution", &tokens[2..], "ggx".to_string(), "Material:Conductor::distribution - ")?;
        if value == "beckmann" {
            dist = MicrofacetDistType::Beckmann;
            dist2 = MicrofacetDistType::Beckmann;
        } 
    }

    let mat = Material::Metal(MetalMaterial::new(con_params, alpha, dist));
    scene.add_dyn_material(Box::new(MetalMaterial::new(con_params2, alpha, dist2)));
    let id = scene.add_material(mat);
    add_material_to_state(id, mat_name, state);
    Ok(())
}

fn add_material_to_state(material_id: u32, mat_name: &str, state: &mut ParseState) {
    if !mat_name.is_empty() {
        state.add_named_material(mat_name.to_string(), material_id);
    } else {
        state.set_material(material_id);
    }
}

fn find_spectrum(pname: &str, tokens: &[String], default: f32x3, err_msg: &str) -> Result<f32x3, Box<dyn Error>> {
    if has_parameter(&format!("rgb {}", pname), tokens) {
        return find_rgb(&format!("rgb {}", pname), tokens, default, err_msg);
    }
    Ok(default)
}

fn find_rgb(pname: &str, tokens: &[String], default: f32x3, err_msg: &str) -> Result<f32x3, Box<dyn Error>> {
    find_f32x3(pname, tokens, default, err_msg)
}

fn find_f32x3(pname: &str, tokens: &[String], default: f32x3, err_msg: &str) -> Result<f32x3, Box<dyn Error>> {
    let offset = match tokens.iter().position(|token| token == pname) {
        None => return Ok(default),
        Some(offset) => offset,
    };

    if tokens[offset..].len() < 6 {
        return Err(format!("{} - {} Insufficient number of tokens!", err_msg, pname).into())
    }

    if tokens[offset + 1] != "[" {
        return Err(format!("{} {} token '[' is expected, got {}", err_msg, pname, tokens[1]).into())
    }

    if tokens[offset + 5] != "]" {
        return Err(format!("{} {} token ']' is expected, got {}", err_msg, pname, tokens[5]).into())
    }

    parse_f32x3(&tokens[offset + 2], &tokens[offset + 3], &tokens[offset + 4], &format!("{} {} ", err_msg, pname))   
}

fn extract_f32_values(tokens: &[String], err_msg: &str) -> Result<Vec<f32>,  Box<dyn Error>> {
    if tokens.len() < 2 {
        return Err(format!("{} Insufficient number of tokens!", err_msg).into())
    }
    if tokens[0] != "[" {
        return Err(format!("{} '[' token expected!", err_msg).into())
    }
    if tokens[tokens.len()-1] != "]" {
        return Err(format!("{} ']' token expected!", err_msg).into())
    }
    
    let mut v: Vec<f32> = Vec::new();
    let end = tokens.len() - 1;
    for tok in &tokens[1..end] {
        let value = parse_f32(tok, err_msg)?;
        v.push(value);
    }
    Ok(v)
}

fn parse_f32x3(v0: &str, v1: &str, v2: &str, err_msg: &str) -> Result<f32x3,  Box<dyn Error>> {
    let v0 = parse_f32(v0, err_msg)?;
    let v1 = parse_f32(v1, err_msg)?;
    let v2 = parse_f32(v2, err_msg)?;
    Ok(f32x3(v0, v1, v2))
}

fn parse_f32(val: &str, err_msg: &str) ->Result<f32,  Box<dyn Error>> {
    let val: f32 = match val.trim().parse() {
        Err(e) => return Err(format!("{} Parsing '{}':{}", err_msg, val, e).into()),
        Ok(val) => val
    };
    Ok(val)   
}

fn parse_u32(val: &str, err_msg: &str) ->Result<u32,  Box<dyn Error>> {
    let val: u32 = match val.trim().parse() {
        Err(e) => return Err(format!("{} Parsing '{}':{}", err_msg, val, e).into()),
        Ok(val) => val
    };
    Ok(val)   
}

fn has_parameter(pname: &str, tokens: &[String]) -> bool {
    tokens.iter().any(|token| token == pname)
}

fn find_value<T>(pname: &str, tokens: &[String], default: T, err_msg: &str) -> Result<T, Box<dyn Error>> 
where T: FromStr, <T as FromStr>::Err: Display
{
    let offset = match tokens.iter().position(|token| token == pname) {
        None => return Ok(default),
        Some(offset) => offset,
    };

    let mut result = match tokens.get(offset + 1) {
        None => return Err(format!("{} Value expected for {}", err_msg, pname).into()),
        Some(result) => result
    };
    if result == "[" {
        result = match tokens.get(offset + 2) {
            None => return Err(format!("{} Value expected for {}", err_msg, pname).into()),
            Some(result) => result
        };

        let _close = match tokens.get(offset + 3) {
            None => return Err(format!("{} ']' expected for '{}'", err_msg, pname).into()),
            Some(val) => val
        };
    }
    let val: T = match result.trim().parse() {
        Err(e) => return Err(format!("{} Parsing:{}", err_msg, e).into()),
        Ok(val) => val
    };
    Ok(val)
}

struct PBRTTokenizer<'a> {
    text: &'a str
}

impl<'a> PBRTTokenizer<'a> {
    pub fn new(text: &'a str) -> Self {
        PBRTTokenizer {text}
    }
}

fn find_offsets(text: &str) -> (usize, usize) {

    let mut chars = text.chars().enumerate();
    let skip = [' ', '\n', '\t', '\r'];
    let mut skip_until_end_of_line = false;
    let mut start_offset = -1;
    let mut inside_p = false;

    loop {
        let (index, c) = match chars.next() {
            Some(c) => c,
            None => break
        };
        if skip_until_end_of_line && c != '\n'{
            continue;
        } else if skip_until_end_of_line && c == '\n' {
            skip_until_end_of_line = false;
            continue;
        }
        if c == '#' {
            skip_until_end_of_line = true;
            continue;
        }
        if c == '[' { return (index, index+1); }
        if c == ']' { return (index, index+1); }
        if skip.contains(&c) { continue; }

        start_offset = index as i32;
        if c == '"' {
            inside_p = true;
        }
        break;
    }

    if start_offset == -1 {
        return (0, 0);
    }
    let mut end_offset = start_offset;

    loop {
        let (index, c) = match chars.next() {
            Some(c) => c,
            None => break
        };
        if inside_p {
            if c == '"' {
                end_offset = index as i32;
                break;
            }

        } else {
            if c == ' ' || c == ']' || c == '\n' || c == '\t' {
                end_offset = index as i32;
                break;
            }
        }              
    }
    (start_offset as usize, end_offset as usize)
}

impl<'a> Iterator for PBRTTokenizer<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        
        let (start, end) = find_offsets(self.text);
        if end > start {
            // NOTE: we exclude quotes, if we have "float t" we return 'float t' without quotes
            if &self.text[start..start+1] == "\"" {
                let result = Some(&self.text[start+1..end]);
                self.text = &self.text[end+1..];
                result
            } else {
                let result = Some(&self.text[start..end]);
                self.text = &self.text[end..];
                result
            }
        }
        else {
            None
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn pbrt_tokenizer() {
        let text = " [ 0 0   1 0 \"pero fov\" 1 1   0 1 2.2\t3.3]        \"5\"";
        let toks = PBRTTokenizer::new(text);
        for tok in toks {
            println!("{}", tok);
        }
    }

}
