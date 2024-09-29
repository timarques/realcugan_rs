use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, AtomicU8, Ordering};

use image::{DynamicImage, GrayAlphaImage, GrayImage, RgbImage, RgbaImage};
use libc::{c_char, c_int, c_uchar, c_uint, c_void, FILE};

#[cfg(any(feature = "models-nose", feature = "models-pro", feature = "models-se"))]
pub use build::Model;
pub use build::SyncGap;

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

    fn realcugan_get_heap_budget(gpuid: c_int) -> c_uint;

    fn realcugan_free_image(mat_ptr: *mut c_void);

    fn realcugan_free(realcugan: *mut c_void);

    fn realcugan_load_files(
        realcugan: *mut c_void, 
        param_path: *mut FILE,
        model_path: *mut FILE
    ) -> c_int;

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
}

#[derive(Debug)]
pub struct RealCugan {
    pointer: Arc<AtomicPtr<c_void>>,
    scale_factor: i32,
    use_cpu: bool,
    ref_count: Arc<AtomicU8>
}

unsafe impl Send for RealCugan {}

impl RealCugan {

    fn calculate_prepadding(scale: i32) -> Result<i32, String> {
        match scale {
            2 => Ok(18),
            3 => Ok(14),
            4 => Ok(19),
            _ => Err(format!("invalid scale value: {}. expected 2, 3, or 4", scale))
        }
    }

    fn calculate_tile_size(tile_size: i32, scale: i32, gpu: i32) -> i32 {
        const DEFAULT_CPU_TILE_SIZE: i32 = 400;
        const MIN_TILE_SIZE: i32 = 32;
        
        if tile_size != 0 {
            return tile_size;
        }
    
        if gpu == -1 {
            return DEFAULT_CPU_TILE_SIZE;
        }
    
        let heap_budget = unsafe { realcugan_get_heap_budget(gpu) } as i32;
        
        let thresholds: Vec<(i32, i32)> = match scale {
            2 => vec![(1300, 400), (800, 300), (200, 100)],
            3 => vec![(3300, 400), (1900, 300), (950, 200), (320, 100)],
            4 => vec![(1690, 400), (980, 300), (530, 200), (240, 100)],
            _ => return MIN_TILE_SIZE,
        };
    
        thresholds
            .iter()
            .find(|(threshold, _)| heap_budget > *threshold)
            .map(|&(_, size)| size)
            .unwrap_or(MIN_TILE_SIZE)
    }

    fn validate_gpu(gpu: i32) -> Result<(), String> {
        if gpu == -1 {
            return Ok(())
        }
        let count = unsafe { realcugan_get_gpu_count() };
        if gpu >= count {
            unsafe { realcugan_destroy_gpu_instance() }
            return Err(format!("gpu {} not found. available gpus: {}", gpu, count))
        }
        Ok(())
    }

    fn create_file_pointer(contents: &[u8]) -> *mut FILE {
        let buffer = contents.as_ptr() as *mut c_void;
        let size = contents.len();
        
        unsafe { libc::fmemopen(buffer, size, "rb\0".as_ptr() as *const c_char) }
    }

    fn load_model(realcugan: *mut c_void, param: &[u8], bin: &[u8]) -> Result<(), String> {
        let file_bin_pointer = Self::create_file_pointer(bin);
        let file_param_pointer = Self::create_file_pointer(param);
        let result = unsafe { realcugan_load_files(realcugan, file_param_pointer, file_bin_pointer) };

        if result != 0 {
            Err(format!("failed to load model files. error code: {}", result))
        } else {
            Ok(())
        }
    }

    pub fn new(
        gpu: i32,
        threads: i32,
        tta: bool,
        sync_gap: i32,
        tile_size: i32,
        scale: i32,
        noise: i32,
        param: &[u8],
        bin: &[u8],
    ) -> Result<Self, String> {
        Self::validate_gpu(gpu)?;
        let prepading = Self::calculate_prepadding(scale)?;
        let tile_size = Self::calculate_tile_size(tile_size, scale, gpu);
        let pointer = unsafe { realcugan_init(gpu,tta, threads) };
        Self::load_model(pointer, param, bin)?;

        unsafe {
            realcugan_set_parameters(
                pointer,
                scale,
                noise,
                prepading,
                sync_gap,
                tile_size
            );
        }

        Ok(Self {
            pointer: Arc::new(AtomicPtr::new(pointer)),
            scale_factor: scale,
            use_cpu: gpu == -1,
            ref_count: Arc::new(AtomicU8::new(0)),
        })
    }

    #[cfg(any(feature = "models-nose", feature = "models-pro", feature = "models-se"))]
    pub fn from_model(model: Model) -> Self {
        build::Builder::new().model(model).unwrap()
    }

    pub fn build<'a>() -> build::Builder<'a> {
        build::Builder::new()
    }

    fn convert_image(width: u32, height: u32, channels: u8, bytes: Vec<u8>) -> Result<DynamicImage, String> {
        match channels {
            4 => RgbaImage::from_raw(width, height, bytes).map(DynamicImage::from),
            3 => RgbImage::from_raw(width, height, bytes).map(DynamicImage::from),
            2 => GrayAlphaImage::from_raw(width, height, bytes).map(DynamicImage::from),
            1 => GrayImage::from_raw(width, height, bytes).map(DynamicImage::from),
            _ => None
        }.ok_or(format!("invalid number of channels: {}. expected 1, 2, 3, or 4", channels))
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
            w: i32::try_from(image.width()).map_err(|e| format!("invalid width: {}", e))?,
            h: i32::try_from(image.height()).map_err(|e| format!("invalid height: {}", e))?,
            c: i32::from(channels),
        })
    }

    fn create_output_buffer(&self, in_buffer: &Image, channels: u8) -> Image {
        Image {
            data: std::ptr::null_mut(),
            w: in_buffer.w * self.scale_factor,
            h: in_buffer.h * self.scale_factor,
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
            .map_err(|e| format!("invalid buffer length: {}", e))?;

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
        let input_buffer = self.create_input_buffer(&image, channels)?;
        let output_buffer = self.create_output_buffer(&input_buffer, channels);

        self.process(input_buffer, output_buffer, channels)
    }

    pub fn process_raw_image(&self, image: &[u8]) -> Result<DynamicImage, String> {
        image::load_from_memory(image)
            .map_err(|x| format!("failed to load raw image: {}", x))
            .and_then(|i| self.process_image(i))
    }

    pub fn process_image_from_path<P: AsRef<Path>>(&self, path: &P) -> Result<DynamicImage, String> {
        let image = image::open(path)
            .map_err(|x| format!("failed to open image from path: {}", x))?;
        self.process_image(image)
    }

}

impl Clone for RealCugan {

    fn clone(&self) -> Self {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
        RealCugan {
            pointer: self.pointer.clone(),
            scale_factor: self.scale_factor,
            use_cpu: self.use_cpu,
            ref_count: self.ref_count.clone(),
        }
    }

}

impl Drop for RealCugan {
    fn drop(&mut self) {
        let clones = self.ref_count.fetch_sub(1, Ordering::Relaxed);
        if clones == 1 {
            let ptr = self.pointer.load(Ordering::Acquire);
            unsafe { realcugan_free(ptr) }
        }
    }
}

mod build {

    use super::RealCugan;

    #[cfg(any(feature = "models-nose", feature = "models-pro", feature = "models-se"))]
    #[derive(Debug, Copy, Clone, PartialEq)]
    pub enum Model {
        #[cfg(feature = "models-se")]
        Se2xNoDenoise,
        #[cfg(feature = "models-se")]
        Se2xConservative,
        #[cfg(feature = "models-se")]
        Se2xLowDenoise,
        #[cfg(feature = "models-se")]
        Se2xMediumDenoise,
        #[cfg(feature = "models-se")]
        Se2xHighDenoise,
        #[cfg(feature = "models-se")]
        Se3xNoDenoise,
        #[cfg(feature = "models-se")]
        Se3xConservative,
        #[cfg(feature = "models-se")]
        Se3xHighDenoise,
        #[cfg(feature = "models-se")]
        Se4xNoDenoise,
        #[cfg(feature = "models-se")]
        Se4xConservative,
        #[cfg(feature = "models-se")]
        Se4xHighDenoise,
        #[cfg(feature = "models-pro")]
        Pro2xNoDenoise,
        #[cfg(feature = "models-pro")]
        Pro2XConservative,
        #[cfg(feature = "models-pro")]
        Pro2XHighDenoise,
        #[cfg(feature = "models-pro")]
        Pro3xNoDenoise,
        #[cfg(feature = "models-pro")]
        Pro3XConservative,
        #[cfg(feature = "models-pro")]
        Pro3XHighDenoise,
        #[cfg(feature = "models-nose")]
        Nose2xNoDenoise
    }

    #[derive(Debug, Copy, Clone, PartialEq)]
    pub enum SyncGap {
        Disabled,      // 0
        Loose,         // 1
        Moderate,      // 2
        Strict,        // 3 (default)
    }

    #[derive(Debug, Clone)]
    struct GeneralParameters {
        gpu: i32,
        tile_size: i32,
        sync_gap: i32,
        threads: i32,
        tta: bool,
    }

    #[derive(Debug, Clone)]
    struct ModelParameters<'a> {
        param: &'a [u8],
        bin: &'a [u8],
        scale: i32,
        noise: i32,
        allow_sync_gap: bool,
    }

    #[derive(Debug, Clone)]
    pub struct Builder<'a> {
        files: Option<(&'a str, &'a str)>,
        parameters: GeneralParameters,
        model_parameters: ModelParameters<'a>
    }

    impl <'a>Default for Builder<'a> {
        fn default() -> Self {
            Self {
                files: None,
                parameters: GeneralParameters{
                    gpu: 0,
                    tile_size: 0,
                    sync_gap: 3,
                    tta: false,
                    threads: 1,
                },
                model_parameters: ModelParameters {
                    param: &[],
                    bin: &[],
                    scale: 2,
                    noise: -1,
                    allow_sync_gap: true,
                }
            }
        }
    }

    impl <'a>Builder<'a> {

        pub fn new() -> Self {
            Self::default()
        }

        pub fn gpu(mut self, gpu: u32) -> Self {
            self.parameters.gpu = gpu as i32;
            self
        }

        pub fn cpu(mut self) -> Self {
            self.parameters.gpu = -1;
            self
        }

        pub fn tta(mut self) -> Self {
            self.parameters.tta = true;
            self
        }

        pub fn tile_size(mut self, tile_size: u32) -> Self {
            self.parameters.tile_size = tile_size as i32;
            self
        }

        pub fn sync_gap(mut self, sync_gap: SyncGap) -> Self {
            self.parameters.sync_gap = match sync_gap {
                SyncGap::Disabled => 0,
                SyncGap::Loose => 1,
                SyncGap::Moderate => 2,
                SyncGap::Strict => 3,
            };
            self
        }

        pub fn threads(mut self, threads: i32) -> Self {
            self.parameters.threads = threads;
            self
        }

        pub fn scale(mut self, scale: i32) -> Self {
            self.model_parameters.scale = scale;
            self
        }

        pub fn noise(mut self, noise: i32) -> Self {
            self.model_parameters.noise = noise;
            self
        }

        pub fn model_files(mut self, param_file: &'a str, bin_file: &'a str) -> Self {
            self.files = Some((param_file, bin_file));
            self
        }

        pub fn model_bytes(mut self, param: &'a [u8], bin: &'a [u8]) -> Self {
            self.model_parameters.param = param;
            self.model_parameters.bin = bin;
            self.files = None;
            self
        }

        #[cfg(any(feature = "models-nose", feature = "models-pro", feature = "models-se"))]
        pub fn model(mut self, model: Model) -> Self {
            let model = match model {
                #[cfg(feature = "models-nose")]
                Model::Nose2xNoDenoise => MODEL_NOSE_2X_NO_DENOISE,
                #[cfg(feature = "models-pro")]
                Model::Pro2xNoDenoise => MODEL_PRO_2X_NO_DENOISE,
                #[cfg(feature = "models-pro")]
                Model::Pro2XConservative => MODEL_PRO_2X_CONSERVATIVE,
                #[cfg(feature = "models-pro")]
                Model::Pro2XHighDenoise => MODEL_PRO_2X_DENOISE_X3,
                #[cfg(feature = "models-pro")]
                Model::Pro3xNoDenoise => MODEL_PRO_3X_NO_DENOISE,
                #[cfg(feature = "models-pro")]
                Model::Pro3XConservative => MODEL_PRO_3X_CONSERVATIVE,
                #[cfg(feature = "models-pro")]
                Model::Pro3XHighDenoise => MODEL_PRO_3X_DENOISE_X3,
                #[cfg(feature = "models-se")]
                Model::Se2xNoDenoise => MODEL_SE_2X_NO_DENOISE,
                #[cfg(feature = "models-se")]
                Model::Se2xConservative => MODEL_SE_2X_CONSERVATIVE,
                #[cfg(feature = "models-se")]
                Model::Se2xLowDenoise => MODEL_SE_2X_DENOISE_X1,
                #[cfg(feature = "models-se")]
                Model::Se2xMediumDenoise => MODEL_SE_2X_DENOISE_X2,
                #[cfg(feature = "models-se")]
                Model::Se2xHighDenoise => MODEL_SE_2X_DENOISE_X3,
                #[cfg(feature = "models-se")]
                Model::Se3xNoDenoise => MODEL_SE_3X_NO_DENOISE,
                #[cfg(feature = "models-se")]
                Model::Se3xConservative => MODEL_SE_3X_CONSERVATIVE,
                #[cfg(feature = "models-se")]
                Model::Se3xHighDenoise => MODEL_SE_3X_DENOISE_X3,
                #[cfg(feature = "models-se")]
                Model::Se4xNoDenoise => MODEL_SE_4X_NO_DENOISE,
                #[cfg(feature = "models-se")]
                Model::Se4xConservative => MODEL_SE_4X_CONSERVATIVE,
                #[cfg(feature = "models-se")]
                Model::Se4xHighDenoise => MODEL_SE_4X_DENOISE_X3,
            };
            self.files = None;
            self.model_parameters = model;
            self
        }

        fn get_bytes(&self) -> Result<(Vec<u8>, Vec<u8>), String> {
            if let Some((param_file, bin_file)) = &self.files {
                let param = std::fs::read(param_file)
                    .map_err(|e| format!("failed to read param file: {}", e))?;
                let bin = std::fs::read(bin_file)
                    .map_err(|e| format!("failed to read bin file: {}", e))?;
                Ok((param, bin))
            } else {
                Ok((self.model_parameters.param.to_vec(), self.model_parameters.bin.to_vec()))
            }
        }

        pub fn build(&self) -> Result<RealCugan, String> {

            let (param, bin) = self.get_bytes()?;

            let sync_gap = if self.model_parameters.allow_sync_gap { 
                self.parameters.sync_gap
            } else {
                0
            };
            RealCugan::new(
                self.parameters.gpu,
                self.parameters.threads,
                self.parameters.tta,
                sync_gap,
                self.parameters.tile_size,
                self.model_parameters.scale,
                self.model_parameters.noise,
                &param,
                &bin
            )
        }

        pub fn unwrap(&self) -> RealCugan {
            self.build().unwrap()
        }

    }

    #[cfg(feature = "models-nose")]
    const MODEL_NOSE_2X_NO_DENOISE: ModelParameters = ModelParameters {
        param: include_bytes!("../models/models-nose/up2x-no-denoise.param"),
        bin: include_bytes!("../models/models-nose/up2x-no-denoise.bin"),
        scale: 2,
        noise: 0,
        allow_sync_gap: true,
    };

    #[cfg(feature = "models-pro")]
    const MODEL_PRO_2X_NO_DENOISE: ModelParameters = ModelParameters {
        param: include_bytes!("../models/models-pro/up2x-no-denoise.param"),
        bin: include_bytes!("../models/models-pro/up2x-no-denoise.bin"),
        scale: 2,
        noise: 0,
        allow_sync_gap: true,
    };

    #[cfg(feature = "models-pro")]
    const MODEL_PRO_2X_CONSERVATIVE: ModelParameters = ModelParameters {
        param: include_bytes!("../models/models-pro/up2x-conservative.param"),
        bin: include_bytes!("../models/models-pro/up2x-conservative.bin"),
        scale: 2,
        noise: -1,
        allow_sync_gap: true,
    };

    #[cfg(feature = "models-pro")]
    const MODEL_PRO_2X_DENOISE_X3: ModelParameters = ModelParameters {
        param: include_bytes!("../models/models-pro/up2x-denoise3x.param"),
        bin: include_bytes!("../models/models-pro/up2x-denoise3x.bin"),
        scale: 2,
        noise: 3,
        allow_sync_gap: true,
    };

    #[cfg(feature = "models-pro")]
    const MODEL_PRO_3X_NO_DENOISE: ModelParameters = ModelParameters {
        param: include_bytes!("../models/models-pro/up3x-no-denoise.param"),
        bin: include_bytes!("../models/models-pro/up3x-no-denoise.bin"),
        scale: 3,
        noise: 0,
        allow_sync_gap: true,
    };

    #[cfg(feature = "models-pro")]
    const MODEL_PRO_3X_CONSERVATIVE: ModelParameters = ModelParameters {
        param: include_bytes!("../models/models-pro/up3x-conservative.param"),
        bin: include_bytes!("../models/models-pro/up3x-conservative.bin"),
        scale: 3,
        noise: -1,
        allow_sync_gap: true,
    };

    #[cfg(feature = "models-pro")]
    const MODEL_PRO_3X_DENOISE_X3: ModelParameters = ModelParameters {
        param: include_bytes!("../models/models-pro/up3x-denoise3x.param"),
        bin: include_bytes!("../models/models-pro/up3x-denoise3x.bin"),
        scale: 3,
        noise: 3,
        allow_sync_gap: true,
    };

    #[cfg(feature = "models-se")]
    const MODEL_SE_2X_NO_DENOISE: ModelParameters = ModelParameters {
        param: include_bytes!("../models/models-se/up2x-no-denoise.param"),
        bin: include_bytes!("../models/models-se/up2x-no-denoise.bin"),
        scale: 2,
        noise: 0,
        allow_sync_gap: false,
    };

    #[cfg(feature = "models-se")]
    const MODEL_SE_2X_CONSERVATIVE: ModelParameters = ModelParameters {
        param: include_bytes!("../models/models-se/up2x-conservative.param"),
        bin: include_bytes!("../models/models-se/up2x-conservative.bin"),
        scale: 2,
        noise: -1,
        allow_sync_gap: false,
    };

    #[cfg(feature = "models-se")]
    const MODEL_SE_2X_DENOISE_X1: ModelParameters = ModelParameters {
        param: include_bytes!("../models/models-se/up2x-denoise1x.param"),
        bin: include_bytes!("../models/models-se/up2x-denoise1x.bin"),
        scale: 2,
        noise: 1,
        allow_sync_gap: false,
    };

    #[cfg(feature = "models-se")]
    const MODEL_SE_2X_DENOISE_X2: ModelParameters = ModelParameters {
        param: include_bytes!("../models/models-se/up2x-denoise2x.param"),
        bin: include_bytes!("../models/models-se/up2x-denoise2x.bin"),
        scale: 2,
        noise: 2,
        allow_sync_gap: false,
    };

    #[cfg(feature = "models-se")]
    const MODEL_SE_2X_DENOISE_X3: ModelParameters = ModelParameters {
        param: include_bytes!("../models/models-se/up2x-denoise3x.param"),
        bin: include_bytes!("../models/models-se/up2x-denoise3x.bin"),
        scale: 2,
        noise: 3,
        allow_sync_gap: false,
    };

    #[cfg(feature = "models-se")]
    const MODEL_SE_3X_NO_DENOISE: ModelParameters = ModelParameters {
        param: include_bytes!("../models/models-se/up3x-no-denoise.param"),
        bin: include_bytes!("../models/models-se/up3x-no-denoise.bin"),
        scale: 3,
        noise: 0,
        allow_sync_gap: false,
    };

    #[cfg(feature = "models-se")]
    const MODEL_SE_3X_CONSERVATIVE: ModelParameters = ModelParameters {
        param: include_bytes!("../models/models-se/up3x-conservative.param"),
        bin: include_bytes!("../models/models-se/up3x-conservative.bin"),
        scale: 3,
        noise: -1,
        allow_sync_gap: false,
    };

    #[cfg(feature = "models-se")]
    const MODEL_SE_3X_DENOISE_X3: ModelParameters = ModelParameters {
        param: include_bytes!("../models/models-se/up3x-denoise3x.param"),
        bin: include_bytes!("../models/models-se/up3x-denoise3x.bin"),
        scale: 3,
        noise: 3,
        allow_sync_gap: false,
    };

    #[cfg(feature = "models-se")]
    const MODEL_SE_4X_NO_DENOISE: ModelParameters = ModelParameters {
        param: include_bytes!("../models/models-se/up4x-no-denoise.param"),
        bin: include_bytes!("../models/models-se/up4x-no-denoise.bin"),
        scale: 4,
        noise: 0,
        allow_sync_gap: false,
    };

    #[cfg(feature = "models-se")]
    const MODEL_SE_4X_CONSERVATIVE: ModelParameters = ModelParameters {
        param: include_bytes!("../models/models-se/up4x-conservative.param"),
        bin: include_bytes!("../models/models-se/up4x-conservative.bin"),
        scale: 4,
        noise: -1,
        allow_sync_gap: false,
    };

    #[cfg(feature = "models-se")]
    const MODEL_SE_4X_DENOISE_X3: ModelParameters = ModelParameters {
        param: include_bytes!("../models/models-se/up4x-denoise3x.param"),
        bin: include_bytes!("../models/models-se/up4x-denoise3x.bin"),
        scale: 4,
        noise: 3,
        allow_sync_gap: false,
    };

}