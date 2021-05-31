use std::fs::File;
use std::io::{BufReader, BufRead};
use std::error::Error;
use std::path::Path;
use std::io::Read;


pub trait PlyModel {
    fn set_number_of_vertices(&mut self, _n: usize) {}
    fn set_number_of_faces(&mut self, _n: usize) {}
	fn add_vertex(&mut self, _v0: f32, _v1: f32, _v2: f32) {}
	fn add_normal(&mut self, _n0: f32, _n1: f32, _n2: f32) {}
	fn add_uv_coord(&mut self, _u: f32, _v: f32) {}
	fn add_rgb(&mut self, _r: u8, _g: u8, _b: u8) {}
	fn add_face(&mut self, _f0: u32, _f1: u32, _f2: u32) {}
}

pub fn read_ply_file<P: AsRef<Path>>(path: P, model: &mut impl PlyModel)  -> Result<(), Box<dyn Error>> {
    let file = File::open(&path)?;
	let mut buf_reader = BufReader::new(file);
	let file_path = path.as_ref().to_str().unwrap();
    let header = read_header(&mut buf_reader, file_path)?;
	if header.elements.len() < 2 {
		return Err(format!("{} - Vertex or Indices list is missing!", file_path).to_string().into())
	}
	read_vertices(&mut buf_reader, &header, model)?;
	read_faces(&mut buf_reader, &header, model)?;
	if header.elements.len() > 2 {
		// TODO
		//read_rest_of_elements(&header, model);
	}
    Ok(())
}

#[allow(non_camel_case_types)]
#[derive(PartialEq)]
pub enum PlyFormat {
	ascii,
	binary_little_endian,
	binary_big_endian,
}

pub enum DataType {
	CHAR,
	UCHAR,
	SHORT,
	USHORT,
	INT,
	UINT,
	FLOAT,
	DOUBLE,
}

pub struct PlyItem {
	dtype: DataType,
	name: String,
}

pub struct PlyList {
	n_indices_dtype: DataType,
	indices_dtype: DataType,
	name: String,
}

pub enum PlyProperty {
	Item(PlyItem),
	List(PlyList),
}

#[allow(dead_code)]
pub struct PlyElement {
	name: String,
	n: usize,
	properties: Vec<PlyProperty>,
}

impl PlyElement {
	pub fn new(name: String, n: usize) -> Self {
		Self {name, n, properties: Vec::new()}
	}

	pub fn add_property(&mut self, property: PlyProperty) {
		self.properties.push(property);
	}
}

pub struct PlyHeader {
	format: PlyFormat,
	elements: Vec<PlyElement>,
}

impl PlyHeader {
	pub fn new(format: PlyFormat, elements: Vec<PlyElement>) -> Self {
		PlyHeader{format, elements}
	}
}

fn conv_dtype(dtype: &str, err_msg: &str) -> Result<DataType, Box<dyn Error>>{
	match dtype {
		"char" => Ok(DataType::CHAR),
		"char8" => Ok(DataType::CHAR),
		"uchar" => Ok(DataType::UCHAR),
		"uchar8" => Ok(DataType::UCHAR),
		"uint8" => Ok(DataType::UCHAR),
		"short" => Ok(DataType::SHORT),
		"short16" => Ok(DataType::SHORT),
		"ushort" => Ok(DataType::USHORT),
		"ushort16" => Ok(DataType::USHORT),
		"int" => Ok(DataType::INT),
		"int32" => Ok(DataType::INT),
		"uint" => Ok(DataType::UINT),
		"uint32" => Ok(DataType::UINT),
		"float" => Ok(DataType::FLOAT),
		"float32" => Ok(DataType::FLOAT),
		"double" => Ok(DataType::DOUBLE),
		"double64" => Ok(DataType::DOUBLE),
		_ => return Err(format!("{} - Unexpected data type {}", err_msg, dtype).to_string().into()),
	}
}

fn read_header(buf_reader: &mut BufReader<File>, path: &str) -> Result<PlyHeader, Box<dyn Error>> {
	let mut line = String::new();
	let _len = buf_reader.read_line(&mut line);
	if line.trim() != "ply" {
        let err_msg = format!("{} not valid, got magic number {} instead of 'ply'", path, line.trim());
        return Err(err_msg.to_string().into());
	}
	line.clear();

    let mut ply_format = PlyFormat::ascii;
	let mut ply_elements = Vec::<PlyElement>::new();

    loop {
		let len = buf_reader.read_line(&mut line);
        if let Ok(0) = len {
            let err_msg = format!("{} - Unexpected end of header!", path);
            return Err(err_msg.to_string().into());
        } 
		if line.trim() == "" || line.trim_start().starts_with("comment") {
			line.clear();
			continue;
		}

        if line.trim_start().starts_with("format") {
			let format = line.split_whitespace().nth(1);
            let format = match format {
                Some(format) => format.trim(),
                None => return Err(format!("{} - Ply format not specified", path).to_string().into())
            };
			match format {
				"ascii" => ply_format = PlyFormat::ascii,
				"binary_little_endian" => ply_format = PlyFormat::binary_little_endian,
				"binary_big_endian" => ply_format = PlyFormat::binary_big_endian,
				_=> return Err(format!("{} - Unexpected ply format!", path).to_string().into())
			}
			line.clear();
			continue;
		}

        if line.trim_start().starts_with("element") {
			let mut iter = line.split_whitespace();
            let name = match iter.nth(1) {
                Some(name) => name,
                None => return Err(format!("{} - Name of element missing", path).to_string().into())
            };

            let n = match iter.next() {
                Some(n) => match n.parse() {
                    Ok(value) => value,
                    Err(e) => return Err(format!("{} - Element {} Parsing: {}", path, name, e).to_string().into()),
                },
                None => return Err(format!("{} - Number of elements {} expected!", path, name).to_string().into())
            };

			ply_elements.push(PlyElement::new(name.to_string(), n));
			line.clear();
			continue;
		}

        if line.trim_start().starts_with("property") {
			let mut iter = line.split_whitespace();
			let dtype = match iter.nth(1) {
                Some(dtype) => dtype,
                None => return Err(format!("{} - Data type of element is missing", path).to_string().into())
            };

			if dtype == "list" {
				let n_indices_dtype = match iter.next() {
					Some(n_indices_dtype) => n_indices_dtype,
					None => return Err(format!("{} - Missing data type for number of indices in 'list'", path).to_string().into())
				};
				let indices_dtype = match iter.next() {
					Some(indices_dtype) => indices_dtype,
					None => return Err(format!("{} - Missing data type for indices element in 'list'", path).to_string().into())
				};
				let name = match iter.next() {
					Some(name) => name,
					None => return Err(format!("{} - Missing property list name", path).to_string().into())
				};
				
				let n_indices_dtype = conv_dtype(n_indices_dtype, path)?;
				let indices_dtype = conv_dtype(indices_dtype, path)?;
				let property = PlyProperty::List(PlyList{n_indices_dtype, indices_dtype: indices_dtype, name: name.to_string()});
				let n = ply_elements.len();
				if n == 0 {
					return Err(format!("{} - Element list is empty", path).to_string().into())
				}
				&mut ply_elements[n-1].add_property(property);
			} else {
				let name = match iter.next() {
					Some(name) => name,
					None => return Err(format!("{} - Missing property name", path).to_string().into())
				};

				let dtype = conv_dtype(dtype, path)?;
				let property = PlyProperty::Item(PlyItem{dtype, name: name.to_string()});
				let n = ply_elements.len();
				if n == 0 {
					return Err(format!("{} - Element list is empty", path).to_string().into())
				}
				&mut ply_elements[n-1].add_property(property);
			}

			line.clear();
			continue;
		}

        if line.trim() == "end_header" {
			break;
		}
		line.clear();
    }

	let header = PlyHeader::new(ply_format, ply_elements);
    Ok(header)
}

fn read_vertices(buf_reader: &mut BufReader<File>, header: &PlyHeader, model: &mut impl PlyModel) -> Result<(), Box<dyn Error>>  {
	let element = &header.elements[0];
	model.set_number_of_vertices(element.n);
	match header.format {
		PlyFormat::ascii => read_ascii_vertices(buf_reader, header, model)?,
		PlyFormat::binary_little_endian => read_binary_vertices(buf_reader, header, model, true)?,
		PlyFormat::binary_big_endian => read_binary_vertices(buf_reader, header, model, false)?,
	}
	Ok(())
}

fn vertex_element_has_name(header: &PlyHeader, name: &str) -> bool {
	let element = &header.elements[0];
	if element.properties.len() < 1 {
		return false;
	}
	let mut has_name = false; 
	for property in element.properties.iter() {
		let pname = match property {
			PlyProperty::Item(item) => &item.name,
			PlyProperty::List(item) => &item.name, 
		};
		if pname == name {
			has_name = true;
		}
	}
	has_name
}

fn read_ascii_vertices(buf_reader: &mut BufReader<File>, header: &PlyHeader, model: &mut impl PlyModel) -> Result<(), Box<dyn Error>>  {
	let element = &header.elements[0];
	let n_vertices = element.n;
	let mut i: usize = 0;
	let mut line = String::new();

	let has_normals = vertex_element_has_name(header, "nx");
	let has_uv = vertex_element_has_name(header, "u");
	let has_rgb = vertex_element_has_name(header, "red");

	while i < n_vertices {
		let len = buf_reader.read_line(&mut line);
		if let Ok(0) = len {
            let err_msg = format!("Unexpected end of file!");
            return Err(err_msg.to_string().into());
        }
		if line.trim() == "" || line.trim_start().starts_with("comment") {
			line.clear();
			continue;
		}
		let mut iter = line.split_whitespace();
		let v0 = parse_f32(iter.next())?;
		let v1 = parse_f32(iter.next())?;
		let v2 = parse_f32(iter.next())?;
		model.add_vertex(v0, v1, v2);
		if has_normals {
			let n0 = parse_f32(iter.next())?;
			let n1 = parse_f32(iter.next())?;
			let n2 = parse_f32(iter.next())?;
			model.add_normal(n0, n1, n2);
		}
		if has_uv {
			let u = parse_f32(iter.next())?;
			let v = parse_f32(iter.next())?;
			model.add_uv_coord(u, v);
		}
		if has_rgb {
			let r = parse_u8(iter.next())?;
			let g = parse_u8(iter.next())?;
			let b = parse_u8(iter.next())?;
			model.add_rgb(r, g, b);
		}

		line.clear();
		i += 1;
	}
	Ok(())
}

fn parse_f32(token: Option<&str>) -> Result<f32, Box<dyn Error>>  {
	match token {
		Some(v0) => match v0.parse() {
			Ok(value) => Ok(value),
			Err(e) => return Err(format!("Parsing vertex: {}", e).to_string().into()),
		},
		None => return Err(format!("Vertex missing!").to_string().into())
	}
}

fn parse_u8(token: Option<&str>) -> Result<u8, Box<dyn Error>>  {
	match token {
		Some(v0) => match v0.parse() {
			Ok(value) => Ok(value),
			Err(e) => return Err(format!("Parsing vertex color: {}", e).to_string().into()),
		},
		None => return Err(format!("Vertex color missing!").to_string().into())
	}
}

fn byte_size(dtype: &DataType) -> usize {
	match dtype {
		DataType::CHAR => 1,
		DataType::UCHAR => 1,
		DataType::SHORT => 2,
		DataType::USHORT => 2,
		DataType::INT => 4,
		DataType::UINT => 4,
		DataType::FLOAT => 4,
		DataType::DOUBLE => 8,
	}
}

fn vertex_component_size(header: &PlyHeader) -> usize {
	let element = &header.elements[0];
	if element.properties.len() < 1 {
		return 0;
	}
	match &element.properties[0] {
		PlyProperty::Item(item) => byte_size(&item.dtype),
		PlyProperty::List(_) => 0, 
	}
}

fn face_list_n_type_size(header: &PlyHeader) -> usize {
	let element = &header.elements[1];
	if element.properties.len() < 1 {
		return 0;
	}
	match &element.properties[0] {
		PlyProperty::Item(_) => 0,
		PlyProperty::List(item) => byte_size(&item.n_indices_dtype), 
	}
}

fn face_list_item_type_size(header: &PlyHeader) -> usize {
	let element = &header.elements[1];
	if element.properties.len() < 1 {
		return 0;
	}
	match &element.properties[0] {
		PlyProperty::Item(_) => 0,
		PlyProperty::List(item) => byte_size(&item.indices_dtype), 
	}
}

fn read_binary_vertices(buf_reader: &mut BufReader<File>, header: &PlyHeader,
						model: &mut impl PlyModel, little_endian: bool) -> Result<(), Box<dyn Error>>  {

	let element = &header.elements[0];
	let n_vertices = element.n;
	let mut i: usize = 0;

	let v_size = vertex_component_size(header);
	if v_size != 4 && v_size != 8 {
		return Err("Vertex component(x, y, z) data type must be float or double".to_string().into());
	}
	let has_normals = vertex_element_has_name(header, "nx");
	let has_uv = vertex_element_has_name(header, "u");
	let has_rgb = vertex_element_has_name(header, "red");

	let mut buffer1: [u8; 1] = [0; 1];
	let mut buffer4_0: [u8; 4] = [0; 4];
	let mut buffer4_1: [u8; 4] = [0; 4];
	let mut buffer4_2: [u8; 4] = [0; 4];

	let mut _buffer8: [u8; 8] = [0; 8];
	// Supported: (x, y, z), (nx, ny, nz), (u, v), (red, green, blue)
	// TODO (x, y, z), (red, green, blue), (nx, ny, nz), (u, v)
	// TODO  - skip_bytes if some unknown property()
	while i < n_vertices {
		let _len = buf_reader.read_exact(&mut buffer4_0);
		let _len = buf_reader.read_exact(&mut buffer4_1);
		let len = buf_reader.read_exact(&mut buffer4_2);
		if little_endian {
			let v0 = f32::from_le_bytes(buffer4_0);
			let v1 = f32::from_le_bytes(buffer4_1);
			let v2 = f32::from_le_bytes(buffer4_2);
			model.add_vertex(v0, v1, v2);
		} else {
			let v0 = f32::from_be_bytes(buffer4_0);
			let v1 = f32::from_be_bytes(buffer4_1);
			let v2 = f32::from_be_bytes(buffer4_2);
			model.add_vertex(v0, v1, v2);
		}
		if let Err(e) = len {
            return Err(format!("Error: {}", e).to_string().into());
        }

		if has_normals {
			let _len = buf_reader.read_exact(&mut buffer4_0);
			let _len = buf_reader.read_exact(&mut buffer4_1);
			let _len = buf_reader.read_exact(&mut buffer4_2);
			if little_endian {
				let n0 = f32::from_le_bytes(buffer4_0);
				let n1 = f32::from_le_bytes(buffer4_1);
				let n2 = f32::from_le_bytes(buffer4_2);
				model.add_normal(n0, n1, n2);
			} else {
				let n0 = f32::from_be_bytes(buffer4_0);
				let n1 = f32::from_be_bytes(buffer4_1);
				let n2 = f32::from_be_bytes(buffer4_2);
				model.add_normal(n0, n1, n2);
			}
		}
		if has_uv {
			let _len = buf_reader.read_exact(&mut buffer4_0);
			let _len = buf_reader.read_exact(&mut buffer4_1);
			if little_endian {
				let u = f32::from_le_bytes(buffer4_0);
				let v = f32::from_le_bytes(buffer4_1);
				model.add_uv_coord(u, v);
			} else {
				let u = f32::from_be_bytes(buffer4_0);
				let v = f32::from_be_bytes(buffer4_1);
				model.add_uv_coord(u, v);
			}
		}
		if has_rgb {
			let _len = buf_reader.read_exact(&mut buffer1);
			let r = buffer1[0];
			let _len = buf_reader.read_exact(&mut buffer1);
			let g = buffer1[0];
			let _len = buf_reader.read_exact(&mut buffer1);
			let b = buffer1[0];
			model.add_rgb(r, g, b);
		}
		// TODO skip bytes if necessary
		i += 1;
	}

	Ok(())
}

fn read_faces(buf_reader: &mut BufReader<File>, header: &PlyHeader, model: &mut impl PlyModel) -> Result<(), Box<dyn Error>>  {
	let element = &header.elements[0];
	model.set_number_of_vertices(element.n);
	match header.format {
		PlyFormat::ascii => read_ascii_faces(buf_reader, header, model)?,
		PlyFormat::binary_little_endian => read_binary_faces(buf_reader, header, model, true)?,
		PlyFormat::binary_big_endian => read_binary_faces(buf_reader, header, model, false)?,
	}
	Ok(())
}

fn read_ascii_faces(buf_reader: &mut BufReader<File>, header: &PlyHeader, model: &mut impl PlyModel) -> Result<(), Box<dyn Error>>  {
	let element = &header.elements[1];
	let n_faces = element.n;
	let mut i: usize = 0;
	let mut line = String::new();

	while i < n_faces {
		let len = buf_reader.read_line(&mut line);
		if let Ok(0) = len {
            let err_msg = format!("Unexpected end of file!");
            return Err(err_msg.to_string().into());
        }
		if line.trim() == "" || line.trim_start().starts_with("comment") {
			line.clear();
			continue;
		}

		let mut iter = line.split_whitespace();
		let n = parse_u32(iter.next())?;

		let idx0 = parse_u32(iter.next())?;
		let mut idx_n_1 = parse_u32(iter.next())?;
		for _ in 2..n {
			let  idx_n = parse_u32(iter.next())?;
			model.add_face(idx0, idx_n_1, idx_n);
			idx_n_1 = idx_n;
		}

		line.clear();
		i += 1;
	}

	Ok(())
}

fn parse_u32(token: Option<&str>) -> Result<u32, Box<dyn Error>>  {
	match token {
		Some(v0) => match v0.parse() {
			Ok(value) => Ok(value),
			Err(e) => return Err(format!("Parsing vertex: {}", e).to_string().into()),
		},
		None => return Err(format!("Vertex missing!").to_string().into())
	}
}

fn read_binary_faces(buf_reader: &mut BufReader<File>, header: &PlyHeader, model: &mut impl PlyModel, little_endian: bool) -> Result<(), Box<dyn Error>>  {

	let element = &header.elements[1];
	let n_faces = element.n;
	let mut i: usize = 0;

	let n_lst_size = face_list_n_type_size(header);
	let lst_item_size = face_list_item_type_size(header);

	while i < n_faces {
		let (end, nitems) = read_binary_value(buf_reader, n_lst_size, little_endian);
		
		if end {
			return Ok(())
		}
		let (_, idx0) = read_binary_value(buf_reader, lst_item_size, little_endian);
		let (_, mut idx_n_1) = read_binary_value(buf_reader, lst_item_size, little_endian);
		for _ in 2..nitems {
			let (_, idx_n) = read_binary_value(buf_reader, lst_item_size, little_endian);
			model.add_face(idx0, idx_n_1, idx_n);
			idx_n_1 = idx_n;
		}
		// TODO skip bytes
		i += 1;
	}

	Ok(())
}

fn read_binary_value(buf_reader: &mut BufReader<File>, size: usize, little_endian: bool) -> (bool, u32) {

	let mut value = 0u32;
	if size == 1 {
		let mut buffer1: [u8; 1] = [0; 1];
		let len = buf_reader.read_exact(&mut buffer1);
		if let Err(_) = len {
            return (true, 0);
        }
		value = buffer1[0] as u32;
	} else if size == 2 {
		let mut buffer2: [u8; 2] = [0; 2];
		let len = buf_reader.read_exact(&mut buffer2);
		if let Err(_) = len {
            return (true, 0);
        }
		if little_endian {
			value = u16::from_le_bytes(buffer2) as u32;
		} else {
			value = u16::from_be_bytes(buffer2) as u32;
		}
	} else if size == 4 {
		let mut buffer4: [u8; 4] = [0; 4];
		let len = buf_reader.read_exact(&mut buffer4);
		if let Err(_) = len {
            return (true, 0);
        }
		if little_endian {
			value = u32::from_le_bytes(buffer4);
		} else {
			value = u32::from_be_bytes(buffer4);
		}
	}
	return (false, value);
}
