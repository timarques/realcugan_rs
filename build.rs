fn main() {
    let dst = cmake::Config::new("src").build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static:-bundle={}", "realcugan-wrapper");
    println!("cargo:rustc-link-lib=dylib={}", "stdc++");
    println!("cargo:rustc-link-lib=dylib={}", "ncnn");
}