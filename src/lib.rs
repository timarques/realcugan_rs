use std::convert::TryFrom;
use std::ffi::CString;

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
        noise: c_int,
        scale: c_int,
        tilesize: c_int,
        prepadding: c_int,
        sync_gap: c_int,
    ) -> *mut c_void;

    fn realcugan_init_gpu_instance();

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

pub struct RealCugan {
    pointer: *mut c_void,
    scale: i32,
    use_cpu: bool,
}

impl RealCugan {

    fn new(pointer: *mut c_void, scale: i32, use_cpu: bool) -> Self {
        Self {
            pointer,
            scale,
            use_cpu,
        }
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

        if self.use_cpu {
            unsafe {
                realcugan_process_cpu(
                    self.pointer,
                    &in_buffer,
                    &out_buffer,
                    &mut mat_ptr,
                );
            }
        } else {
            unsafe {
                realcugan_process(
                    self.pointer,
                    &in_buffer,
                    &out_buffer,
                    &mut mat_ptr,
                );
            }
        }

        let length = usize::try_from(out_buffer.h * out_buffer.w * out_buffer.c)
            .map_err(|e| format!("Invalid buffer length: {}", e))?;

        let copied_bytes = unsafe {std::slice::from_raw_parts(out_buffer.data as *const u8, length).to_vec()};
        unsafe { realcugan_free_image(mat_ptr); }

        Self::convert_image(
            out_buffer.w as u32,
            out_buffer.h as u32,
            channels,
            copied_bytes,
        )
    }

    pub fn upscale_raw_image(&self, image: Vec<u8>) -> Result<DynamicImage, String> {
        image::load_from_memory(&image)
            .map_err(|x| x.to_string())
            .and_then(|i| self.upscale_image(i))
    }

    pub fn upscale_image(&self, image: DynamicImage) -> Result<DynamicImage, String> {
        let (image, channels) = self.prepare_image(image);
        let in_buffer = self.create_input_buffer(&image, channels)?;
        let out_buffer = self.create_output_buffer(&in_buffer, channels);
        self.process(in_buffer, out_buffer, channels)
    }

    pub fn upscale_image_path(&self, path: &str) -> Result<DynamicImage, String> {
        let image = image::open(path)
            .map_err(|x| x.to_string())?;
        self.upscale_image(image)
    }

}

impl Default for RealCugan {

    fn default() -> Self {
        RealCuganBuilder::new().build().unwrap()
    }

}

impl Drop for RealCugan {
    fn drop(&mut self) {
        unsafe {
            realcugan_free(self.pointer);
        }
    }
}


#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Model {
    Nose,
    Pro,
    Se,
}

pub struct RealCuganBuilder {
    gpu: i32,
    noise: i32,
    scale: i32,
    tile_size: i32,
    sync_gap: i32,
    threads: i32,
    tta_mode: bool,
    model: Model,
    directory: String,
    model_paths: (String, String),
}

impl RealCuganBuilder {

    pub fn new() -> Self {
        Self {
            gpu: 0,
            noise: -1,
            scale: 2,
            model: Model::Se,
            tile_size: 0,
            sync_gap: 0,
            tta_mode: false,
            threads: 0,
            model_paths: (String::new(), String::new()),
            directory: {
                std::env::current_dir()
                    .map(|x| x.to_str().unwrap().to_string())
                    .unwrap_or(String::new())
            },
        }
    }

    pub fn gpu(mut self, gpu: u32) -> Self {
        self.gpu = gpu as i32;
        self
    }

    pub fn noise(mut self, noise: i32) -> Self {
        if noise < -1 {
            self.noise = -1;
            println!("Warning: noise must be >= -1, setting noise to -1");
        } else if noise > 3 {
            self.noise = 3;
            println!("Warning: noise must be <= 3, setting noise to 3");
        } else {
            self.noise = noise;
        }
        self.noise = noise;
        self
    }

    pub fn scale(mut self, scale: u32) -> Self {
        if scale < 2 {
            self.scale = 2;
            println!("Warning: scale must be >= 2, setting scale to 2");
        } else if scale > 4 {
            self.scale = 4;
            println!("Warning: scale must be <= 4, setting scale to 4");
        } else {
            self.scale = scale as i32;
        }
        self
    }

    pub fn model(mut self, model: Model) -> Self {
        self.model = model;
        self
    }

    pub fn tile_size(mut self, tile_size: u32) -> Self {
        if tile_size > 32 {
            println!("Warning: tile_size must be <= 32, setting tile_size to 32");
            self.tile_size = 32;
        } else {
            self.tile_size = tile_size as i32;
        }
        self
    }

    pub fn sync_gap(mut self, sync_gap: u32) -> Self {
        if sync_gap > 3 {
            println!("Warning: sync_gap must be <= 3, setting sync_gap to 3");
            self.sync_gap = 3;
        } else {
            self.sync_gap = sync_gap as i32;
        }
        self
    }

    pub fn tta_mode(mut self, tta_mode: bool) -> Self {
        self.tta_mode = tta_mode;
        self
    }

    pub fn threads(mut self, threads: i32) -> Self {
        self.threads = threads;
        self
    }

    pub fn directory(mut self, directory: &str) -> Self {
        self.directory = directory.to_string();
        self
    }

    pub fn cpu(mut self) -> Self {
        self.gpu = -1;
        self
    }

    fn get_prepadding(&self) -> i32 {
        match self.scale {
            2 => 18,
            3 => 14,
            4 => 19,
            _ => panic!("invalid scale")
        }
    }

    fn get_sync_gap(&self) -> i32 {
        match self.model {
            Model::Se => 0,
            _ => self.sync_gap
        }
    }

    fn create_model_paths(&mut self) -> Result<(), String> {
        if !std::fs::exists(&self.directory).unwrap_or(false) {
            return Err(format!("models directory {} does not exist", self.directory));
        }

        let model = match self.model {
            Model::Se => "models-se",
            Model::Nose => "models-nose",
            Model::Pro => "models-pro",
        };

        let paths = match self.noise {
            -1 => (
                format!("{}/{}/up{}x-conservative.bin", self.directory, model, self.scale),
                format!("{}/{}/up{}x-conservative.param", self.directory, model, self.scale)
            ),
            0 => (
                format!("{}/{}/up{}x-no-denoise.bin", self.directory, model, self.scale),
                format!("{}/{}/up{}x-no-denoise.param", self.directory, model, self.scale)
            ),
            _ => (
                format!("{}/{}/up{}x-denoise{}x.bin", self.directory, model, self.scale, self.noise),
                format!("{}/{}/up{}x-denoise{}x.param", self.directory, model, self.scale, self.noise)
            )
        };

        if !std::fs::exists(&paths.0).unwrap_or(false) {
            return Err(format!("model file {} does not exist", &paths.0));
        } else if !std::fs::exists(&paths.1).unwrap_or(false) {
            return Err(format!("model file {} does not exist", &paths.1));
        }

        self.model_paths = paths;
        Ok(())
    }
    
    fn get_tile_size(&self) -> i32 {
        if self.tile_size != 0 {
            return self.tile_size
        }
    
        if self.gpu == -1 {
            return 400
        }
    
        let heap_budget = unsafe { realcugan_get_heap_budget(self.gpu) };
        match self.scale {
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

    fn init_gpu(&self) {
        if self.gpu == -1 {
            return
        }
        unsafe { realcugan_init_gpu_instance() }
        let count = unsafe { realcugan_get_gpu_count() } as i32;
        if self.gpu >= count {
            unsafe { realcugan_destroy_gpu_instance() }
            panic!("gpu {} not found", self.gpu)
        }
    }

    fn init_model(&self, realcugan: *mut c_void) {
        let bin_path_cstr = CString::new(self.model_paths.0.clone()).unwrap();
        let param_path_cstr = CString::new(self.model_paths.1.clone()).unwrap();
        unsafe {
            realcugan_load(
                realcugan, 
                param_path_cstr.as_ptr(), 
                bin_path_cstr.as_ptr()
            )
        }
    }

    fn init(&self) -> *mut c_void {
        self.init_gpu();
        let cugan = unsafe {
            realcugan_init(
                self.gpu,
                self.tta_mode,
                self.threads,
                self.noise,
                self.scale,
                self.get_tile_size() as i32,
                self.get_prepadding(),
                self.get_sync_gap(),
            )
        };
        self.init_model(cugan);
        cugan
    }

    pub fn build(mut self) -> Result<RealCugan, String> {
        self.create_model_paths()?;
        let cugan = self.init();
        let realcugan = RealCugan::new(cugan, self.scale, self.gpu == -1);
        Ok(realcugan)
    }

}

pub fn new() -> RealCuganBuilder {
    RealCuganBuilder::new()
}
