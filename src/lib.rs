use std::ffi::CString;
use std::path::{Path, PathBuf};

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
    pointer: *mut c_void,
    scale: i32,
    use_cpu: bool,
    builder: RealCuganBuilder
}

unsafe impl Send for RealCugan {}

impl RealCugan {

    fn new(
        pointer: *mut c_void,
        scale: i32,
        use_cpu: bool,
        builder: RealCuganBuilder
    ) -> Self {
        Self {
            pointer,
            scale,
            use_cpu,
            builder
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

impl Default for RealCugan {

    fn default() -> Self {
        RealCuganBuilder::new().build().unwrap()
    }

}

impl Clone for RealCugan {

    fn clone(&self) -> Self {
        self.builder.clone().build().unwrap()
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

#[derive(Debug, Clone)]
pub struct RealCuganBuilder {
    gpu: i32,
    noise: i32,
    scale: i32,
    tile_size: i32,
    sync_gap: i32,
    threads: i32,
    tta_mode: bool,
    model: Model,
    directory: PathBuf,
    model_paths: Option<(PathBuf, PathBuf)>,
}

impl Default for RealCuganBuilder {
    fn default() -> Self {
        Self {
            gpu: 0,
            noise: -1,
            scale: 2,
            model: Model::Se,
            tile_size: 0,
            sync_gap: 0,
            tta_mode: false,
            threads: 0,
            model_paths: None,
            directory: std::env::current_dir().unwrap_or_default(),
        }
    }
}

impl RealCuganBuilder {

    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_from_files(bin: String, param: String) -> Self {
        let scale = match() {
            _ if bin.contains("2x") => 2,
            _ if bin.contains("3x") => 3,
            _ if bin.contains("4x") => 4,
            _ => 1,
        };
        let mut this = Self::new().scale(scale);
        this.model_paths = Some((PathBuf::from(bin), PathBuf::from(param)));
        this
    }

    pub fn gpu(mut self, gpu: u32) -> Self {
        self.gpu = gpu as i32;
        self
    }

    pub fn noise(mut self, noise: i32) -> Self {
        const MIN_NOISE: i32 = -1;
        const MAX_NOISE: i32 = 3;
        self.noise = noise.clamp(MIN_NOISE, MAX_NOISE);
        if noise < MIN_NOISE || noise > MAX_NOISE {
            println!("Noise value {} is out of range [-1, 3], clamping to {}", noise, self.noise);
        }
        self
    }

    pub fn scale(mut self, scale: u32) -> Self {
        const MIN_SCALE: u32 = 2;
        const MAX_SCALE: u32 = 4;
        self.scale = scale.clamp(MIN_SCALE, MAX_SCALE) as i32;
        if scale < MIN_SCALE || scale > MAX_SCALE {
            println!("Scale {} is out of range [{}, {}], clamping to {}", scale, MIN_SCALE, MAX_SCALE, self.scale);
        }
        self
    }

    pub fn model(mut self, model: Model) -> Self {
        self.model = model;
        self
    }

    pub fn tile_size(mut self, tile_size: u32) -> Self {
        const MAX_TILE_SIZE: u32 = 32;
        self.tile_size = tile_size.min(MAX_TILE_SIZE) as i32;
        if tile_size > MAX_TILE_SIZE {
            println!("Warning: tile_size must be <= 32, setting tile_size to 32");
        }
        self
    }

    pub fn sync_gap(mut self, sync_gap: u32) -> Self {
        const MAX_SYNC_GAP: u32 = 3;
        self.sync_gap = sync_gap.min(MAX_SYNC_GAP) as i32;
        if sync_gap > MAX_SYNC_GAP {
            println!("Sync gap {} is greater than maximum {}, clamping to {}", sync_gap, MAX_SYNC_GAP, self.sync_gap);
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

    pub fn directory<P: AsRef<Path>>(mut self, directory: P) -> Self {
        self.directory = directory.as_ref().to_path_buf();
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
            _ => unreachable!("Invalid scale: {}", self.scale)
        }
    }

    fn get_sync_gap(&self) -> i32 {
        match self.model {
            Model::Se => 0,
            _ => self.sync_gap
        }
    }

    fn create_model_paths(&mut self) -> Result<(), String> {
        let directory = self.directory.display();
        if !std::fs::exists(&self.directory).unwrap_or(false) {
            return Err(format!("models directory {} does not exist", directory));
        }

        let model = match self.model {
            Model::Se => "models-se",
            Model::Nose => "models-nose",
            Model::Pro => "models-pro",
        };

        let (bin, param) = match self.noise {
            -1 => (
                format!("up{}x-conservative.bin", self.scale),
                format!("up{}x-conservative.param", self.scale)
            ),
            0 => (
                format!("up{}x-no-denoise.bin", self.scale),
                format!("up{}x-no-denoise.param", self.scale)
            ),
            _ => (
                format!("up{}x-denoise{}x.bin", self.scale, self.noise),
                format!("up{}x-denoise{}x.param", self.scale, self.noise)
            )
        };

        self.model_paths = Some((
            self.directory.join(model).join(bin),
            self.directory.join(model).join(param),
        ));
        Ok(())
    }

    fn validate_model_paths(&self) -> Result<(), String> {
        if let Some(paths) = &self.model_paths {
            if !std::fs::exists(&paths.0).unwrap_or(false) {
                return Err(format!("model file {} does not exist", &paths.0.display()));
            } else if !std::fs::exists(&paths.1).unwrap_or(false) {
                return Err(format!("model file {} does not exist", &paths.1.display()));
            }
            Ok(())
        } else {
            Err(format!("empty model paths")) 
        }
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

    fn init_gpu(&self) -> Result<(), String> {
        if self.gpu == -1 {
            return Ok(())
        }
        let count = unsafe { realcugan_get_gpu_count() } as i32;
        if self.gpu >= count {
            unsafe { realcugan_destroy_gpu_instance() }
            return Err(format!("gpu {} not found", self.gpu))
        }
        Ok(())
    }

    fn init_model(&self, realcugan: *mut c_void) -> Result<(), String> {
        let (bin_path, param_path) = self.model_paths.as_ref().ok_or_else(|| format!("teste"))?;
        let bin_path_cstr = CString::new(bin_path.to_str().unwrap())
            .map_err(|_| format!("Failed to create CString for bin path"))?;
        let param_path_cstr = CString::new(param_path.to_str().unwrap())
            .map_err(|_| format!("Failed to create CString for param path"))?;
        unsafe {
            realcugan_load(
                realcugan, 
                param_path_cstr.as_ptr(), 
                bin_path_cstr.as_ptr()
            )
        }
        Ok(())
    }

    fn init(&self) -> Result<*mut c_void, String> {
        self.validate_model_paths()?;
        self.init_gpu()?;
        let cugan = unsafe {
            realcugan_init(
                self.gpu,
                self.tta_mode,
                self.threads,
                self.noise,
                self.scale,
                self.get_tile_size(),
                self.get_prepadding(),
                self.get_sync_gap(),
            )
        };
        self.init_model(cugan)?;
        Ok(cugan)
    }

    pub fn build(mut self) -> Result<RealCugan, String> {
        if self.model_paths.is_none() {
            self.create_model_paths()?;
        }
        let cugan = self.init()?;
        let realcugan = RealCugan::new(cugan, self.scale, self.gpu == -1, self);
        Ok(realcugan)
    }

}