use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, AtomicU8, Ordering};

use image::{DynamicImage, GrayAlphaImage, GrayImage, RgbaImage, RgbImage};
use libc::{c_char, c_int, c_uchar, c_uint, c_void};

#[repr(C)]
#[derive(Debug)]
pub struct Image {
    pub data: *const c_uchar,
    pub w: c_int,
    pub h: c_int,
    pub c: c_int,
}

extern "C" {
    fn realcugan_init(
        gpuid: c_int,
        tta_mode: bool,
        num_threads: c_int,
    ) -> *mut c_void;

    fn realcugan_set_parameters(
        realcugan: *mut c_void,
        scale: c_int,
        noise: c_int,
        prepadding: c_int,
        sync_gap: c_int,
        tilesize: c_int,
    );

    fn realcugan_get_gpu_count() -> c_int;

    fn realcugan_destroy_gpu_instance();

    fn realcugan_load(realcugan: *mut c_void, param_path: *const c_char, model_path: *const c_char);

    fn realcugan_process(
        realcugan: *mut c_void,
        in_image: *const Image,
        out_image: *const Image,
        mat_ptr: *mut *mut c_void,
    ) -> c_int;

    fn realcugan_process_cpu(
        realcugan: *mut c_void,
        in_image: &Image,
        out_image: &Image,
        mat_ptr: *mut *mut c_void,
    ) -> c_int;

    fn realcugan_get_heap_budget(gpuid: c_int) -> c_uint;

    fn realcugan_free_image(mat_ptr: *mut c_void);

    fn realcugan_free(realcugan: *mut c_void);
}

#[derive(Debug)]
pub struct RealCugan {
    pointer: Arc<AtomicPtr<c_void>>,
    scale: i32,
    use_cpu: bool,
    clones: Arc<AtomicU8>
}

unsafe impl Send for RealCugan {}

impl RealCugan {

    fn get_prepadding(scale: i32) -> Result<i32, String> {
        match scale {
            2 => Ok(18),
            3 => Ok(14),
            4 => Ok(19),
            _ => Err(format!("Invalid scale value: {}. Expected 2, 3, or 4.", scale)),
        }
    }

    fn get_sync_gap(sync_gap: i32, model_str: &str) -> i32 {
        if model_str.contains("models-se") {
            0
        } else {
            sync_gap
        }
    }

    fn get_tile_size(tile_size: i32, scale: i32, gpu: i32) -> i32 {
        if tile_size != 0 {
            return tile_size
        }
    
        if gpu == -1 {
            return 400
        }
    
        let heap_budget = unsafe { realcugan_get_heap_budget(gpu) };
        match scale {
            2 => {
                if heap_budget > 1300 {
                    400
                } else if heap_budget > 800 {
                    300
                } else if heap_budget > 200 {
                    100
                } else {
                    32
                }
            },
            3 => {
                if heap_budget > 330 {
                    400
                } else if heap_budget > 1900 {
                    300
                } else if heap_budget > 950 {
                    200
                } else if heap_budget > 320 {
                    100
                } else {
                    32
                }
            },
            4 => {
                if heap_budget > 1690 {
                    400
                } else if heap_budget > 980 {
                    300
                } else if heap_budget > 530 {
                    200
                } else if heap_budget > 240 {
                    100
                } else {
                    32
                }
            },
            _ => {
                32
            }
        }
    
    }

    fn validate_gpu(gpu: i32) -> Result<(), String> {
        if gpu == -1 {
            return Ok(())
        }
        let count = unsafe { realcugan_get_gpu_count() };
        if gpu >= count {
            unsafe { realcugan_destroy_gpu_instance() }
            return Err(format!("gpu {} not found", gpu))
        }
        Ok(())
    }

    fn validate_model_paths(param: &str, bin: &str) -> Result<(), String> {
        if !std::fs::exists(&param).unwrap_or(false) {
            return Err(format!("model file {} does not exist", &param));
        } else if !std::fs::exists(&bin).unwrap_or(false) {
            return Err(format!("model file {} does not exist", &bin));
        }
        Ok(())
    }

    fn init_model(realcugan: *mut c_void, param: &str, bin: &str) -> Result<(), String> {
        let bin_path_cstr = CString::new(bin)
            .map_err(|_| format!("Failed to create CString for bin path"))?;
        let param_path_cstr = CString::new(param)
            .map_err(|_| format!("Failed to create CString for param path"))?;
        unsafe { realcugan_load(realcugan, param_path_cstr.as_ptr(), bin_path_cstr.as_ptr()) }
        Ok(())
    }

    fn create_realcugan(gpu: i32, threads: i32, tta: bool) -> *mut c_void {
        unsafe {
            realcugan_init(
                gpu,
                tta,
                threads
            )
        }
    }

    pub fn new(
        gpu: i32,
        threads: i32,
        tta: bool,

        scale: i32,
        noise: i32,
        sync_gap: i32,
        tile_size: i32,
        
        model_param: &str,
        model_bin: &str,
    ) -> Result<Self, String> {
        Self::validate_gpu(gpu)?;
        Self::validate_model_paths(model_param, model_bin)?;
        let pointer = Self::create_realcugan(gpu, threads, tta);
        Self::init_model(pointer, model_param, model_bin)?;

        unsafe {
            realcugan_set_parameters(
                pointer,
                scale,
                noise,
                Self::get_prepadding(scale)?,
                Self::get_sync_gap(sync_gap, model_bin),
                Self::get_tile_size(tile_size, scale, gpu)
            );
        }

        Ok(Self {
            pointer: Arc::new(AtomicPtr::new(pointer)),
            scale: scale,
            use_cpu: gpu == -1,
            clones: Arc::new(AtomicU8::new(0)),
        })
    }

    pub fn from_files(param: &str, bin: &str) -> Result<Self, String> {
        RealCuganBuilder::new().files(param, bin).build()
    }

    fn convert_image(width: u32, height: u32, channels: u8, bytes: Vec<u8>) -> Result<DynamicImage, String> {
        match channels {
            4 => RgbaImage::from_raw(width, height, bytes).map(DynamicImage::from),
            3 => RgbImage::from_raw(width, height, bytes).map(DynamicImage::from),
            2 => GrayAlphaImage::from_raw(width, height, bytes).map(DynamicImage::from),
            1 => GrayImage::from_raw(width, height, bytes).map(DynamicImage::from),
            _ => None
        }.ok_or("invalid image".to_string())
    }

    fn prepare_image(&self, image: DynamicImage) -> (DynamicImage, u8) {
        let bytes_per_pixel = image.color().bytes_per_pixel();
        match bytes_per_pixel {
            1 => (DynamicImage::from(image.to_rgb8()), 3),
            2 => (DynamicImage::from(image.to_rgba8()), 4),
            _ => (image, bytes_per_pixel),
        }
    }

    fn create_input_buffer(&self, image: &DynamicImage, channels: u8) -> Result<Image, String> {
        Ok(Image {
            data: image.as_bytes().as_ptr(),
            w: i32::try_from(image.width()).map_err(|e| format!("Invalid width: {}", e))?,
            h: i32::try_from(image.height()).map_err(|e| format!("Invalid height: {}", e))?,
            c: i32::from(channels),
        })
    }

    fn create_output_buffer(&self, in_buffer: &Image, channels: u8) -> Image {
        Image {
            data: std::ptr::null_mut(),
            w: in_buffer.w * self.scale,
            h: in_buffer.h * self.scale,
            c: i32::from(channels),
        }
    }

    fn process(&self, in_buffer: Image, out_buffer: Image, channels: u8) -> Result<DynamicImage, String> {
        let mut mat_ptr = std::ptr::null_mut();
        let ptr = self.pointer.load(Ordering::Acquire);

        if self.use_cpu {
            unsafe {
                realcugan_process_cpu(
                    ptr,
                    &in_buffer,
                    &out_buffer,
                    &mut mat_ptr,
                );
            }
        } else {
            unsafe {
                realcugan_process(
                    ptr,
                    &in_buffer,
                    &out_buffer,
                    &mut mat_ptr,
                );
            }
        }

        let length = usize::try_from(out_buffer.h * out_buffer.w * out_buffer.c)
            .map_err(|e| format!("Invalid buffer length: {}", e))?;

        let copied_bytes = unsafe { std::slice::from_raw_parts(out_buffer.data as *const u8, length).to_vec() };
        unsafe { realcugan_free_image(mat_ptr) }

        Self::convert_image(
            out_buffer.w as u32,
            out_buffer.h as u32,
            channels,
            copied_bytes,
        )
    }

    pub fn process_image(&self, image: DynamicImage) -> Result<DynamicImage, String> {
        let (image, channels) = self.prepare_image(image);
        let in_buffer = self.create_input_buffer(&image, channels)?;
        let out_buffer = self.create_output_buffer(&in_buffer, channels);

        self.process(in_buffer, out_buffer, channels)
    }

    pub fn process_raw_image(&self, image: &[u8]) -> Result<DynamicImage, String> {
        image::load_from_memory(image)
            .map_err(|x| x.to_string())
            .and_then(|i| self.process_image(i))
    }

    pub fn process_image_from_path<P: AsRef<Path>>(&self, path: &P) -> Result<DynamicImage, String> {
        let image = image::open(path)
            .map_err(|x| x.to_string())?;
        self.process_image(image)
    }

}

impl Clone for RealCugan {

    fn clone(&self) -> Self {
        self.clones.fetch_add(1, Ordering::Relaxed);
        RealCugan {
            pointer: self.pointer.clone(),
            scale: self.scale,
            use_cpu: self.use_cpu,
            clones: self.clones.clone(),
        }
    }

}

impl Default for RealCugan {

    fn default() -> Self {
        RealCuganBuilder::new().build().unwrap()
    }

}

impl Drop for RealCugan {
    fn drop(&mut self) {
        let clones = self.clones.fetch_sub(1, Ordering::Relaxed);
        if clones == 1 {
            let ptr = self.pointer.load(Ordering::Acquire);
            unsafe { realcugan_free(ptr) }
        }
    }
}


#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Model {
    Nose,
    Pro,
    Se,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Noise {
    Default,        // -1
    Off,            // 0
    Low,            // 1
    Medium,         // 2
    High,           // 3
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Scale {
    Double,   // 2x
    Triple,   // 3x
    Quadruple // 4x
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum SyncGap {
    Disabled,      // 0
    Loose,         // 1
    Moderate,      // 2
    Strict,        // 3 (default)
}

#[derive(Debug, Clone)]
pub struct RealCuganBuilder {
    gpu: i32,
    noise: i32,
    scale: i32,
    tile_size: i32,
    sync_gap: i32,
    threads: i32,
    tta: bool,
    model: String,
    directory: PathBuf,
    files: Option<(String, String)>,
}

impl Default for RealCuganBuilder {
    fn default() -> Self {
        Self {
            gpu: 0,
            noise: -1,
            scale: 2,
            model: format!("models-se"),
            tile_size: 0,
            sync_gap: 3,
            tta: false,
            threads: 0,
            directory: std::env::current_dir().unwrap_or_default(),
            files: None,
        }
    }
}

impl RealCuganBuilder {

    pub fn new() -> Self {
        Self::default()
    }

    pub fn files(mut self, param: &str, bin: &str) -> Self {
        self.noise = if param.contains("no-denoise") {
            0
        } else if param.contains("conservative") {
            1
        } else if param.contains("denoise2x") {
            2
        } else if param.contains("denoise3x") {
            3
        } else {
            self.noise
        };

        self.scale = if param.contains("up2x") {
            2
        } else if param.contains("up3x") {
            3
        } else if param.contains("up4x") {
            4
        } else {
            self.scale
        };

        self.files = Some((param.to_string(), bin.to_string()));
        self
    }

    pub fn gpu(mut self, gpu: u32) -> Self {
        self.gpu = gpu as i32;
        self
    }

    pub fn noise(mut self, noise: Noise) -> Self {
        self.noise = match noise {
            Noise::Default => -1,
            Noise::Off => 0,
            Noise::Low => 1,
            Noise::Medium => 2,
            Noise::High => 3
        };
        self
    }

    pub fn scale(mut self, scale: Scale) -> Self {
        self.scale = match scale {
            Scale::Double => 2,
            Scale::Triple => 3,
            Scale::Quadruple => 4
        };
        self
    }

    pub fn model(mut self, model: Model) -> Self {
        self.model = match model {
            Model::Se => "models-se",
            Model::Nose => "models-nose",
            Model::Pro => "models-pro",
        }.to_string();
        self
    }

    pub fn tile_size(mut self, tile_size: u32) -> Self {
        self.tile_size = tile_size as i32;
        self
    }

    pub fn sync_gap(mut self, sync_gap: SyncGap) -> Self {
        self.sync_gap = match sync_gap {
            SyncGap::Disabled => 0,
            SyncGap::Loose => 1,
            SyncGap::Moderate => 2,
            SyncGap::Strict => 3,
        };
        self
    }

    pub fn tta(mut self, tta: bool) -> Self {
        self.tta = tta;
        self
    }

    pub fn threads(mut self, threads: i32) -> Self {
        self.threads = threads;
        self
    }

    pub fn directory<P: AsRef<Path>>(mut self, directory: P) -> Self {
        self.directory = directory.as_ref().to_path_buf();
        self
    }

    pub fn cpu(mut self) -> Self {
        self.gpu = -1;
        self
    }

    fn get_model_paths(&self) -> Result<(String, String), String> {
        if let Some(files) = self.files.clone() {
            let param = std::fs::canonicalize(&files.0)
                .map_err(|_| format!("Failed to canonicalize file path {}", &files.0))
                .map(|p| p.to_string_lossy().to_string())?;
            let bin = std::fs::canonicalize(&files.1)
                .map_err(|_| format!("Failed to canonicalize file path {}", &files.1))
                .map(|p| p.to_string_lossy().to_string())?;
            return Ok((param, bin))
        }

        let directory = self.directory.display();
        if !std::fs::exists(&self.directory).unwrap_or(false) {
            return Err(format!("models directory {} does not exist", directory));
        }

        let (param, bin) = match self.noise {
            -1 => (
                format!("up{}x-conservative.param", self.scale),
                format!("up{}x-conservative.bin", self.scale),
            ),
            0 => (
                format!("up{}x-no-denoise.param", self.scale),
                format!("up{}x-no-denoise.bin", self.scale),
            ),
            _ => (
                format!("up{}x-denoise{}x.param", self.scale, self.noise),
                format!("up{}x-denoise{}x.bin", self.scale, self.noise),
            )
        };

        Ok((
            self.directory.join(&self.model).join(param).to_str().unwrap().to_string(),
            self.directory.join(&self.model).join(bin).to_str().unwrap().to_string(),
        ))
    }

    pub fn build(&self) -> Result<RealCugan, String> {
        let (param, bin) = self.get_model_paths()?;
        RealCugan::new(
            self.gpu,
            self.threads,
            self.tta,
            self.scale,
            self.noise,
            self.sync_gap,
            self.tile_size,
            &param,
            &bin
        )
    }

}