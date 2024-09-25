fn main() {
    println!("cargo::rerun-if-changed=src");
    let dst = cmake::Config::new("src")
        .cflag("-w")
        .build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static:-bundle={}", "realcugan-wrapper");
    println!("cargo:rustc-link-lib=dylib={}", "stdc++");
}