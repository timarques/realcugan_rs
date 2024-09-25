fn main() {
    println!("cargo::rerun-if-changed=src");
    //let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // let realcugan_dir = out_dir.join("realcugan");
    // create_dir(&realcugan_dir).unwrap_or_default();
    let dst = cmake::Config::new("src").build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static:-bundle={}", "realcugan-wrapper");
}