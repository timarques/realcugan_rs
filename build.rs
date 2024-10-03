use cmake::Config;
use std::process::Command;
use std::io::BufRead;

const NCNN_REPO_URL: &str = "https://github.com/Tencent/ncnn";
const NCNN_COMMIT_HASH: &str = "066614351391d309c96ae1e00c6fb1bd873b4949";

fn execute_command(command: &mut Command) -> Result<(), String> {
    let status = command.status().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!("Command failed with exit code: {}", status));
    }
    Ok(())
}

fn clone_ncnn(target_dir: &str) -> Result<(), String> {
    if std::fs::exists(target_dir).unwrap() {
        return Ok(())
    }
    execute_command(
        Command::new("git")
            .args(&["clone", "--recursive", NCNN_REPO_URL])
            .arg(target_dir)
    )?;

    execute_command(
        Command::new("git")
            .current_dir(target_dir)
            .args(&["checkout", NCNN_COMMIT_HASH])
    )?;
    Ok(())
}

fn configure_ncnn_build(target_dir: &str) -> Config {
    let mut config = Config::new(target_dir);
    config.define("NCNN_BUILD_TOOLS", "OFF")
          .define("NCNN_BUILD_EXAMPLES", "OFF")
          .define("NCNN_BUILD_BENCHMARK", "OFF")
          .define("NCNN_ENABLE_LTO", "ON")
          .define("NCNN_SHARED_LIB", "OFF")
          .define("NCNN_VULKAN", "ON")
          .define("NCNN_SYSTEM_GLSLANG", "OFF")
          .define("CMAKE_BUILD_TYPE", "Release");
    config
}

fn disable_logs(target_dir: &str) -> Result<(), std::io::Error> {
    let platform_file = format!("{}/src/platform.h.in", target_dir);
    let file = std::fs::File::open(&platform_file)?;
    let reader = std::io::BufReader::new(file);

    let mut text = String::new();
    let mut skip_lines = false;

    for line_result in reader.lines() {
        let line = line_result?;

        if line.contains("#define NCNN_LOGE(...) do {") {
            skip_lines = true;
            text += &format!("#define NCNN_LOGE(...)\n");
            continue
        } else if (line.contains("#endif") || line.contains("#else")) && skip_lines {
            skip_lines = false;
        } else if skip_lines {
            continue
        }
        text += &format!("{}\n", line);
    }

    std::fs::write(platform_file, text)?;

    Ok(())
}

fn build_ncnn(output: &str) -> Result<(), String> {
    let target_dir = format!("{}/ncnn", output);

    println!("cargo:rustc-link-lib={}", "stdc++");
    println!("cargo:rustc-link-lib={}", "pthread");
    println!("cargo:rustc-link-lib={}", "omp");
    println!("cargo:rustc-link-lib={}", "vulkan");

    clone_ncnn(&target_dir)?;
    disable_logs(&target_dir).map_err(|r| r.to_string())?;
    configure_ncnn_build(&target_dir)
        .cflag("-O3")
        .cxxflag("-O3")
        .build();

    println!("cargo:rustc-link-search=native={}/lib64", output);
    println!("cargo:rustc-link-lib=static={}", "MachineIndependent");
    println!("cargo:rustc-link-lib=static={}", "SPIRV");
    println!("cargo:rustc-link-lib=static={}", "GenericCodeGen");
    println!("cargo:rustc-link-lib=static={}", "OSDependent");
    println!("cargo:rustc-link-lib=static={}", "OGLCompiler");
    println!("cargo:rustc-link-lib=static={}", "glslang");
    println!("cargo:rustc-link-lib=static={}", "ncnn");
    Ok(())
}

fn main() {
    let output = std::env::var("OUT_DIR").unwrap();
    if cfg!(feature = "system-ncnn") {
        println!("cargo:rustc-link-lib=dylib={}", "ncnn");
    } else {
        if let Err(e) = build_ncnn(&output) {
            panic!("Failed to build ncnn: {}", e);
        }
    }
    Config::new("src").build();
    println!("cargo:rustc-link-search=native={}/lib", &output);
    println!("cargo:rustc-link-lib=static={}", "realcugan-wrapper");
}