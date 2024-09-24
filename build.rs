
use cmake::Config;

fn main() {
    let realcugan = Config::new("src").build();
    println!("cargo:rustc-link-search=native={}", realcugan.display());
    println!("cargo:rustc-link-lib=static:-bundle={}", "realcugan-wrapper");
    println!("cargo:rustc-link-lib=dylib={}", "ncnn");
}