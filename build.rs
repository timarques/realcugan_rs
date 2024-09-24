use std::env;
use std::fs::create_dir;
use std::path::PathBuf;

use cmake::Config;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let realcugan_dir = out_dir.join("realcugan");
    create_dir(&realcugan_dir).unwrap_or_default();
    let realcugan = {
        let mut config = Config::new("src");
        config.out_dir(realcugan_dir);
        config.build()
    };
    println!("cargo:rustc-link-search=native={}", realcugan.join("lib").display());
    println!("cargo:rustc-link-lib=static:-bundle={}", "realcugan-rs");
    println!("cargo:rustc-link-lib=dylib={}", "ncnn");
    if cfg!(unix) {
        println!("cargo:rustc-link-lib=dylib={}", "stdc++");
    }
}