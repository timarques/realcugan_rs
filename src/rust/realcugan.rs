use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Once;

use libc::{c_int, c_uchar, FILE};

use crate::error::Error;
use crate::options::Options;

extern "C" {
    fn realcugan_init(
        gpuid: c_int,
        tta_mode: bool,
        num_threads: c_int,
        scale: c_int,
        noise: c_int,
        sync_gap: c_int,
        tilesize: c_int,
    ) -> *mut c_void;

    fn realcugan_get_gpu_count() -> c_int;
    fn realcugan_destroy_gpu_instance();
    fn realcugan_free(realcugan: *mut c_void);

    fn realcugan_load_files(
        realcugan: *mut c_void,
        param_path: *mut FILE,
        model_path: *mut FILE
    ) -> c_int;

    fn realcugan_process(
        realesrgan: *mut c_void,
        in_image: *const c_uchar,
        out_image: *mut c_uchar,
        width: c_int,
        height: c_int,
        channels: c_int,
    ) -> c_int;

    fn realcugan_process_cpu(
        realesrgan: *mut c_void,
        in_image: *const c_uchar,
        out_image: *mut c_uchar,
        width: c_int,
        height: c_int,
        channels: c_int,
    ) -> c_int;
}


#[derive(Debug)]
pub struct RealCugan<'a> {
    pointer: *mut c_void,
    options: Options<'a>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> RealCugan<'a> {

    pub fn new(options: Options<'a>) -> Result<Self, Error> {
        Self::setup_cleanup();
        Self::validate_gpu(options.gpuid)?;

        let pointer = unsafe {
            realcugan_init(
                options.gpuid,
                options.tta_mode,
                options.threads,
                options.scale_factor,
                options.noise_level,
                options.sync_gap,
                options.tile_size,
            )
        };

        if pointer.is_null() {
            unsafe { realcugan_destroy_gpu_instance() };
            return Err(Error::InvalidPointer);
        }

        Self::load_model(pointer, options.param, options.bin)?;

        Ok(Self {
            pointer,
            options,
            _marker: PhantomData,
        })
    }
    
    pub fn options(&self) -> &Options<'a> {
        &self.options
    }
    
    fn validate_gpu(gpu: i32) -> Result<(), Error> {
        if gpu == -1 {
            return Ok(());
        }
        let count = unsafe { realcugan_get_gpu_count() };
        if gpu >= count {
            return Err(Error::GpuNotFound { requested: gpu, available: count });
        }
        Ok(())
    }

    fn create_file_pointer(contents: &[u8]) -> *mut FILE {
        unsafe { 
            libc::fmemopen(
                contents.as_ptr() as *mut c_void,
                contents.len(),
                c"rb".as_ptr()
            )
        }
    }

    fn load_model(realcugan: *mut c_void, param: &[u8], bin: &[u8]) -> Result<(), Error> {
        if param.is_empty() || bin.is_empty() {
            return Err(Error::InvalidModel);
        }

        let file_param_pointer = Self::create_file_pointer(param);
        let file_bin_pointer = Self::create_file_pointer(bin);

        if file_bin_pointer.is_null() || file_param_pointer.is_null() {
            if !file_param_pointer.is_null() {
                unsafe {
                    libc::fclose(file_param_pointer)
                };
            }

            if !file_bin_pointer.is_null() { 
                unsafe {
                    libc::fclose(file_bin_pointer)
                };
            }

            return Err(Error::FilePointerCreationFailed);
        }

        let result = unsafe {
            realcugan_load_files(
                realcugan,
                file_param_pointer,
                file_bin_pointer
            )
        };

        unsafe {
            libc::fclose(file_param_pointer);
            libc::fclose(file_bin_pointer);
        }

        if result != 0 {
            Err(Error::ModelLoadFailed { code: result })
        } else {
            Ok(())
        }
    }
    
    fn setup_cleanup() {
        static CLEANUP: Once = Once::new();
        CLEANUP.call_once(|| {
            extern "C" fn cleanup() {
                unsafe { realcugan_destroy_gpu_instance() };
            }
            unsafe { libc::atexit(cleanup) };
        });
    }

    pub fn process(&self, input: &[u8], width: usize, height: usize) -> Result<Vec<u8>, Error> {
        if self.pointer.is_null() {
            return Err(Error::InvalidPointer);
        }

        let expected_length = width * height;
        if input.len() % expected_length != 0 {
            return Err(Error::InvalidInput {
                expected_length,
                actual_length: input.len()
            });
        }
        
        let process_fn = if self.options.gpuid >= 0 {
            realcugan_process
        } else {
            realcugan_process_cpu
        };

        let channels = input.len() / expected_length;
        let output_width = width * self.options.scale_factor as usize;
        let output_height = height * self.options.scale_factor as usize;
        let mut output = vec![0u8; output_width * output_height * channels];
        let code = unsafe {
            process_fn(
                self.pointer,
                input.as_ptr(),
                output.as_mut_ptr(),
                width as c_int,
                height as c_int,
                channels as c_int
            )
        };

        if code == 0 {
            Ok(output)
        } else {
            Err(Error::ProcessingFailed { code })
        }
    }


    pub fn process_batch<I, B>(
        &self,
        inputs: I,
        width: usize,
        height: usize,
    ) -> Result<Vec<Vec<u8>>, Error>
    where 
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        inputs
            .into_iter()
            .map(|input_chunk| self.process(input_chunk.as_ref(), width, height))
            .collect()
    }
    
    #[cfg(feature = "image")]
    pub fn process_file<P>(&self, path: P) -> Result<crate::Image, Error>
    where
        P: AsRef<std::path::Path>,
    {
        let img = image::open(path).map_err(|e| Error::ImageOpenFailed(e.to_string()))?;
        self.process_image(img)
    }

    #[cfg(feature = "image")]
    pub fn process_image(&self, image: crate::Image) -> Result<crate::Image, Error> {
        use image::{ColorType, ImageBuffer, DynamicImage};
        
        let color_type = image.color();
        let input = image.to_rgb8().into_raw();
        let width = image.width();
        let height = image.height();
        let output = self.process(&input, width as usize, height as usize)?;
        let new_width = width * self.options.scale_factor as u32;
        let new_height = height * self.options.scale_factor as u32;
    
        let dynamic_image = match color_type {
            ColorType::Rgb8 => ImageBuffer::from_raw(new_width, new_height, output).map(DynamicImage::ImageRgb8),
            ColorType::Rgba8 => ImageBuffer::from_raw(new_width, new_height, output).map(DynamicImage::ImageRgba8),
            ColorType::L8 => ImageBuffer::from_raw(new_width, new_height, output).map(DynamicImage::ImageLuma8),
            ColorType::La8 => ImageBuffer::from_raw(new_width, new_height, output).map(DynamicImage::ImageLumaA8),
            _ => ImageBuffer::from_raw(new_width, new_height, output).map(DynamicImage::ImageRgb8),
        };
    
        dynamic_image.ok_or(Error::ColorConversionFailed)
    }
}


impl Drop for RealCugan<'_> {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            unsafe { realcugan_free(self.pointer) };
        }
    }
}