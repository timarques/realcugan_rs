use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let source_dir = manifest_dir.join("src");
    
    let shaders_dir = source_dir.join("shaders");
    let cpp_dir = source_dir.join("cpp");
    let output_shaders_dir = out_dir.join("shaders");
    
    fs::create_dir_all(&output_shaders_dir).unwrap();

    println!("cargo:rerun-if-changed={}", shaders_dir.display());
    println!("cargo:rerun-if-changed={}", cpp_dir.display());
    
    generate_shader_headers(&shaders_dir, &output_shaders_dir);
    build(&cpp_dir, &output_shaders_dir);
}

fn generate_shader_headers(shaders_dir: &Path, output_dir: &Path) {
    let shader_files = [
        "realcugan_preproc.comp",
        "realcugan_postproc.comp", 
        "realcugan_4x_postproc.comp",
        "realcugan_preproc_tta.comp",
        "realcugan_postproc_tta.comp",
        "realcugan_4x_postproc_tta.comp",
    ];
    
    for shader in &shader_files {
        let shader_path = shaders_dir.join(shader);
        let shader_stem = Path::new(shader).file_stem().unwrap().to_str().unwrap();
        let header_path = output_dir.join(format!("{shader_stem}.comp.hex.h"));
        
        if !shader_path.exists() {
            panic!("Shader file not found: {}", shader_path.display());
        }
        
        let comp_data = fs::read_to_string(&shader_path).unwrap();
        
        let version_start = comp_data.find("#version").unwrap_or(0);
        let comp_data = &comp_data[version_start..];
        let comp_data = comp_data.replace("\n ", "\n");
        
        let comp_data_hex: Vec<String> = comp_data
            .bytes()
            .map(|byte| format!("0x{byte:02x}"))
            .collect();
        
        let header_content = format!(
            "static const char {}_comp_data[] = {{{}}};\n",
            shader_stem,
            comp_data_hex.join(",")
        );
        
        fs::write(&header_path, header_content).unwrap();
    }
}

fn build(cpp_dir: &Path, shaders_dir: &Path) {
    println!("cargo:rustc-link-lib=stdc++");
    
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=pthread");
    }

    let mut build = cc::Build::new();
    
    build
        .cpp(true)
        .std("c++11")
        .opt_level(3)
        .warnings(false)
        .flag_if_supported("-O3")
        .flag_if_supported("-pthread")
        .flag_if_supported("-fopenmp")
        .include(cpp_dir)
        .include(shaders_dir)
        .file(cpp_dir.join("wrapper.cpp"))
        .file(cpp_dir.join("realcugan.cpp"));

    pkg_config::probe_library("vulkan").unwrap();
    pkg_config::probe_library("ncnn").unwrap();

    build.compile("realcugan-wrapper");
    
}