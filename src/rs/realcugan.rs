use crate::builder::Builder;
#[cfg(any(feature = "models-nose", feature = "models-pro", feature = "models-se"))]
use crate::builder::Model;

use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, AtomicU8, Ordering};

use image::{DynamicImage, GrayAlphaImage, GrayImage, RgbImage, RgbaImage};
use libc::{c_char, c_int, c_uchar, c_uint, c_void, FILE};

static INSTANCES: AtomicU8 = AtomicU8::new(0);

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
    use_cpu: bool
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
            if INSTANCES.load(Ordering::Relaxed) == 0 {
                unsafe { realcugan_destroy_gpu_instance() }
            }
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
        if file_bin_pointer.is_null() || file_param_pointer.is_null() {
            return Err(format!("failed to create file pointers"));
        }
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

        INSTANCES.fetch_add(1, Ordering::Relaxed);

        Ok(Self {
            pointer: Arc::new(AtomicPtr::new(pointer)),
            scale_factor: scale,
            use_cpu: gpu == -1,
        })
    }

    #[cfg(any(feature = "models-nose", feature = "models-pro", feature = "models-se"))]
    pub fn from_model(model: Model) -> Self {
        Builder::new().model(model).unwrap()
    }

    pub fn build<'a>() -> Builder<'a> {
        Builder::new()
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
        if ptr.is_null() {
            return Err(format!("invalid pointer"))
        }

        let result = if self.use_cpu {
            unsafe {
                realcugan_process_cpu(
                    ptr,
                    &in_buffer,
                    &out_buffer,
                    &mut mat_ptr,
                )
            }
        } else {
            unsafe {
                realcugan_process(
                    ptr,
                    &in_buffer,
                    &out_buffer,
                    &mut mat_ptr,
                )
            }
        };

        if result != 0 {
            unsafe { realcugan_free_image(mat_ptr) };
            return Err(format!("failed to process image"))
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

    pub fn process_raw_image(&self, image: &[u8]) -> Result<Vec<u8>, String> {
        let format = image::guess_format(image).unwrap_or(image::ImageFormat::Png);
        image::load_from_memory(image)
            .map_err(|x| format!("failed to load raw image: {}", x))
            .and_then(|i| self.process_image(i))
            .and_then(|i| {
                let mut bytes = Cursor::new(Vec::new());
                i.write_to(&mut bytes, format)
                    .map_err(|e| format!("Failed to write to buffer: {}", e))
                    .map(|_| bytes.into_inner())
            })
    }

    pub fn process_image_from_path<P: AsRef<Path>>(&self, path: &P) -> Result<DynamicImage, String> {
        let image = image::open(path)
            .map_err(|x| format!("failed to open image from path: {}", x))?;
        self.process_image(image)
    }

}

impl Clone for RealCugan {

    fn clone(&self) -> Self {
        RealCugan {
            pointer: self.pointer.clone(),
            scale_factor: self.scale_factor,
            use_cpu: self.use_cpu,
        }
    }

}

impl Drop for RealCugan {
    fn drop(&mut self) {
        if Arc::strong_count(&self.pointer) == 1 {
            let ptr = self.pointer.load(Ordering::Acquire);
            if !ptr.is_null() {
                unsafe { realcugan_free(ptr) }
            }

            if INSTANCES.fetch_sub(1, Ordering::AcqRel) == 1 {
                unsafe { realcugan_destroy_gpu_instance() }
            }
        }
    }
}