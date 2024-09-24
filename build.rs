use std::env;
use std::fs::create_dir;
use std::path::PathBuf;

use cmake::Config;

fn main() {
    println!("cargo::rerun-if-changed=src");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let realcugan_dir = out_dir.join("realcugan");
    create_dir(&realcugan_dir).unwrap_or_default();
    let dst = Config::new("src").out_dir(realcugan_dir).build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static:-bundle={}", "realcugan-wrapper");
    println!("cargo:rustc-link-lib=dylib={}", "ncnn");
    if cfg!(unix) {
        println!("cargo:rustc-link-lib=dylib={}", "stdc++");
    }
}