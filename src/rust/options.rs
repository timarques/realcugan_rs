use std::path::Path;

#[cfg(feature = "models-nose")]
const MODEL_NOSE_2X_NO_DENOISE: (&'static [u8], &'static [u8]) = (
    include_bytes!("../../models/models-nose/up2x-no-denoise.param"),
    include_bytes!("../../models/models-nose/up2x-no-denoise.bin"),
);

#[cfg(feature = "models-pro")]
const MODEL_PRO_2X_NO_DENOISE: (&'static [u8], &'static [u8]) = (
    include_bytes!("../../models/models-pro/up2x-no-denoise.param"),
    include_bytes!("../../models/models-pro/up2x-no-denoise.bin"),
);

#[cfg(feature = "models-pro")]
const MODEL_PRO_2X_CONSERVATIVE: (&'static [u8], &'static [u8]) = (
    include_bytes!("../../models/models-pro/up2x-conservative.param"),
    include_bytes!("../../models/models-pro/up2x-conservative.bin"),
);

#[cfg(feature = "models-pro")]
const MODEL_PRO_2X_DENOISE_X3: (&'static [u8], &'static [u8]) = (
    include_bytes!("../../models/models-pro/up2x-denoise3x.param"),
    include_bytes!("../../models/models-pro/up2x-denoise3x.bin"),
);

#[cfg(feature = "models-pro")]
const MODEL_PRO_3X_NO_DENOISE: (&'static [u8], &'static [u8]) = (
    include_bytes!("../../models/models-pro/up3x-no-denoise.param"),
    include_bytes!("../../models/models-pro/up3x-no-denoise.bin"),
);

#[cfg(feature = "models-pro")]
const MODEL_PRO_3X_CONSERVATIVE: (&'static [u8], &'static [u8]) = (
    include_bytes!("../../models/models-pro/up3x-conservative.param"),
    include_bytes!("../../models/models-pro/up3x-conservative.bin"),
);

#[cfg(feature = "models-pro")]
const MODEL_PRO_3X_DENOISE_X3: (&'static [u8], &'static [u8]) = (
    include_bytes!("../../models/models-pro/up3x-denoise3x.param"),
    include_bytes!("../../models/models-pro/up3x-denoise3x.bin"),
);

#[cfg(feature = "models-se")]
const MODEL_SE_2X_NO_DENOISE: (&'static [u8], &'static [u8]) = (
    include_bytes!("../../models/models-se/up2x-no-denoise.param"),
    include_bytes!("../../models/models-se/up2x-no-denoise.bin"),
);

#[cfg(feature = "models-se")]
const MODEL_SE_2X_CONSERVATIVE: (&'static [u8], &'static [u8]) = (
    include_bytes!("../../models/models-se/up2x-conservative.param"),
    include_bytes!("../../models/models-se/up2x-conservative.bin"),
);

#[cfg(feature = "models-se")]
const MODEL_SE_2X_DENOISE_X1: (&'static [u8], &'static [u8]) = (
    include_bytes!("../../models/models-se/up2x-denoise1x.param"),
    include_bytes!("../../models/models-se/up2x-denoise1x.bin"),
);

#[cfg(feature = "models-se")]
const MODEL_SE_2X_DENOISE_X2: (&'static [u8], &'static [u8]) = (
    include_bytes!("../../models/models-se/up2x-denoise2x.param"),
    include_bytes!("../../models/models-se/up2x-denoise2x.bin"),
);

#[cfg(feature = "models-se")]
const MODEL_SE_2X_DENOISE_X3: (&'static [u8], &'static [u8]) = (
    include_bytes!("../../models/models-se/up2x-denoise3x.param"),
    include_bytes!("../../models/models-se/up2x-denoise3x.bin"),
);

#[cfg(feature = "models-se")]
const MODEL_SE_3X_NO_DENOISE: (&'static [u8], &'static [u8]) = (
    include_bytes!("../../models/models-se/up3x-no-denoise.param"),
    include_bytes!("../../models/models-se/up3x-no-denoise.bin"),
);

#[cfg(feature = "models-se")]
const MODEL_SE_3X_CONSERVATIVE: (&'static [u8], &'static [u8]) = (
    include_bytes!("../../models/models-se/up3x-conservative.param"),
    include_bytes!("../../models/models-se/up3x-conservative.bin"),
);

#[cfg(feature = "models-se")]
const MODEL_SE_3X_DENOISE_X3: (&'static [u8], &'static [u8]) = (
    include_bytes!("../../models/models-se/up3x-denoise3x.param"),
    include_bytes!("../../models/models-se/up3x-denoise3x.bin"),
);

#[cfg(feature = "models-se")]
const MODEL_SE_4X_NO_DENOISE: (&'static [u8], &'static [u8]) = (
    include_bytes!("../../models/models-se/up4x-no-denoise.param"),
    include_bytes!("../../models/models-se/up4x-no-denoise.bin"),
);

#[cfg(feature = "models-se")]
const MODEL_SE_4X_CONSERVATIVE: (&'static [u8], &'static [u8]) = (
    include_bytes!("../../models/models-se/up4x-conservative.param"),
    include_bytes!("../../models/models-se/up4x-conservative.bin"),
);

#[cfg(feature = "models-se")]
const MODEL_SE_4X_DENOISE_X3: (&'static [u8], &'static [u8]) = (
    include_bytes!("../../models/models-se/up4x-denoise3x.param"),
    include_bytes!("../../models/models-se/up4x-denoise3x.bin"),
);

#[cfg(any(feature = "models-nose", feature = "models-pro", feature = "models-se"))]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum OptionsModel {
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

#[cfg(any(feature = "models-nose", feature = "models-pro", feature = "models-se"))]
impl OptionsModel {

    const fn get_bytes(&self) -> (&'static [u8], &'static [u8]) {
        match self {
            #[cfg(feature = "models-se")]
            Self::Se2xNoDenoise => MODEL_SE_2X_NO_DENOISE,
            #[cfg(feature = "models-se")]
            Self::Se2xConservative => MODEL_SE_2X_CONSERVATIVE,
            #[cfg(feature = "models-se")]
            Self::Se2xLowDenoise => MODEL_SE_2X_DENOISE_X1,
            #[cfg(feature = "models-se")]
            Self::Se2xMediumDenoise => MODEL_SE_2X_DENOISE_X2,
            #[cfg(feature = "models-se")]
            Self::Se2xHighDenoise => MODEL_SE_2X_DENOISE_X3,
            #[cfg(feature = "models-se")]
            Self::Se3xNoDenoise => MODEL_SE_3X_NO_DENOISE,
            #[cfg(feature = "models-se")]
            Self::Se3xConservative => MODEL_SE_3X_CONSERVATIVE,
            #[cfg(feature = "models-se")]
            Self::Se3xHighDenoise => MODEL_SE_3X_DENOISE_X3,
            #[cfg(feature = "models-se")]
            Self::Se4xNoDenoise => MODEL_SE_4X_NO_DENOISE,
            #[cfg(feature = "models-se")]
            Self::Se4xConservative => MODEL_SE_4X_CONSERVATIVE,
            #[cfg(feature = "models-se")]
            Self::Se4xHighDenoise => MODEL_SE_4X_DENOISE_X3,
            #[cfg(feature = "models-pro")]
            Self::Pro2xNoDenoise => MODEL_PRO_2X_NO_DENOISE,
            #[cfg(feature = "models-pro")]
            Self::Pro2XConservative => MODEL_PRO_2X_CONSERVATIVE,
            #[cfg(feature = "models-pro")]
            Self::Pro2XHighDenoise => MODEL_PRO_2X_DENOISE_X3,
            #[cfg(feature = "models-pro")]
            Self::Pro3xNoDenoise => MODEL_PRO_3X_NO_DENOISE,
            #[cfg(feature = "models-pro")]
            Self::Pro3XConservative => MODEL_PRO_3X_CONSERVATIVE,
            #[cfg(feature = "models-pro")]
            Self::Pro3XHighDenoise => MODEL_PRO_3X_DENOISE_X3,
            #[cfg(feature = "models-nose")]
            Self::Nose2xNoDenoise => MODEL_NOSE_2X_NO_DENOISE
        }
    }

    const fn get_scale_factor(&self) -> i32 {
        match self {
            #[cfg(feature = "models-se")]
            Self::Se2xNoDenoise => 2,
            #[cfg(feature = "models-se")]
            Self::Se2xConservative => 2,
            #[cfg(feature = "models-se")]
            Self::Se2xLowDenoise => 2,
            #[cfg(feature = "models-se")]
            Self::Se2xMediumDenoise => 2,
            #[cfg(feature = "models-se")]
            Self::Se2xHighDenoise => 2,
            #[cfg(feature = "models-se")]
            Self::Se3xNoDenoise => 3,
            #[cfg(feature = "models-se")]
            Self::Se3xConservative => 3,
            #[cfg(feature = "models-se")]
            Self::Se3xHighDenoise => 3,
            #[cfg(feature = "models-se")]
            Self::Se4xNoDenoise => 4,
            #[cfg(feature = "models-se")]
            Self::Se4xConservative => 4,
            #[cfg(feature = "models-se")]
            Self::Se4xHighDenoise => 4,
            #[cfg(feature = "models-pro")]
            Self::Pro2xNoDenoise => 2,
            #[cfg(feature = "models-pro")]
            Self::Pro2XConservative => 2,
            #[cfg(feature = "models-pro")]
            Self::Pro2XHighDenoise => 2,
            #[cfg(feature = "models-pro")]
            Self::Pro3xNoDenoise => 3,
            #[cfg(feature = "models-pro")]
            Self::Pro3XConservative => 3,
            #[cfg(feature = "models-pro")]
            Self::Pro3XHighDenoise => 3,
            #[cfg(feature = "models-nose")]
            Self::Nose2xNoDenoise => 2
        }
    }

    const fn allow_sync_gap(&self) -> bool {
        match self {
            #[cfg(feature = "models-se")]
            Self::Se2xNoDenoise => false,
            #[cfg(feature = "models-se")]
            Self::Se2xConservative => false,
            #[cfg(feature = "models-se")]
            Self::Se2xLowDenoise => false,
            #[cfg(feature = "models-se")]
            Self::Se2xMediumDenoise => false,
            #[cfg(feature = "models-se")]
            Self::Se2xHighDenoise => false,
            #[cfg(feature = "models-se")]
            Self::Se3xNoDenoise => false,
            #[cfg(feature = "models-se")]
            Self::Se3xConservative => false,
            #[cfg(feature = "models-se")]
            Self::Se3xHighDenoise => false,
            #[cfg(feature = "models-se")]
            Self::Se4xNoDenoise => false,
            #[cfg(feature = "models-se")]
            Self::Se4xConservative => false,
            #[cfg(feature = "models-se")]
            Self::Se4xHighDenoise => false,
            #[cfg(feature = "models-pro")]
            Self::Pro2xNoDenoise => true,
            #[cfg(feature = "models-pro")]
            Self::Pro2XConservative => true,
            #[cfg(feature = "models-pro")]
            Self::Pro2XHighDenoise => true,
            #[cfg(feature = "models-pro")]
            Self::Pro3xNoDenoise => true,
            #[cfg(feature = "models-pro")]
            Self::Pro3XConservative => true,
            #[cfg(feature = "models-pro")]
            Self::Pro3XHighDenoise => true,
            #[cfg(feature = "models-nose")]
            Self::Nose2xNoDenoise => true
        }
    }

    const fn get_noise_level(&self) -> i32 {
        match self {
            #[cfg(feature = "models-se")]
            Self::Se2xNoDenoise => 0,
            #[cfg(feature = "models-se")]
            Self::Se2xConservative => -1,
            #[cfg(feature = "models-se")]
            Self::Se2xLowDenoise => 1,
            #[cfg(feature = "models-se")]
            Self::Se2xMediumDenoise => 2,
            #[cfg(feature = "models-se")]
            Self::Se2xHighDenoise => 3,
            #[cfg(feature = "models-se")]
            Self::Se3xNoDenoise => 0,
            #[cfg(feature = "models-se")]
            Self::Se3xConservative => -1,
            #[cfg(feature = "models-se")]
            Self::Se3xHighDenoise => 1,
            #[cfg(feature = "models-se")]
            Self::Se4xNoDenoise => 0,
            #[cfg(feature = "models-se")]
            Self::Se4xConservative => -1,
            #[cfg(feature = "models-se")]
            Self::Se4xHighDenoise => 1,
            #[cfg(feature = "models-pro")]
            Self::Pro2xNoDenoise => 0,
            #[cfg(feature = "models-pro")]
            Self::Pro2XConservative => -1,
            #[cfg(feature = "models-pro")]
            Self::Pro2XHighDenoise => 1,
            #[cfg(feature = "models-pro")]
            Self::Pro3xNoDenoise => 0,
            #[cfg(feature = "models-pro")]
            Self::Pro3XConservative => -1,
            #[cfg(feature = "models-pro")]
            Self::Pro3XHighDenoise => 1,
            #[cfg(feature = "models-nose")]
            Self::Nose2xNoDenoise => 0
        }
    }

}

pub enum OptionsSyncGap {
    Disabled = 0,
    Loose = 1,
    Moderate = 2,
    Strict = 3,
}

pub enum OptionsScaleFactor {
    Double = 2,
    Triple = 3,
    Quadruple = 4,
}

pub enum OptionsNoiseLevel {
    None = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Conservative = -1,
}

pub struct Options<'a> {
    pub (super) gpuid: i32,
    pub (super) threads: i32,
    pub (super) tta_mode: bool,
    pub (super) sync_gap: i32,
    pub (super) tile_size: i32,
    pub (super) scale_factor: i32,
    pub (super) noise_level: i32,
    pub (super) param: &'a [u8],
    pub (super) bin: &'a [u8],
}

impl <'a>Default for Options<'a> {
    fn default() -> Self {

        #[allow(unreachable_patterns)]
        let bytes: (&[u8], &[u8]) = match () {
            #[cfg(feature = "models-se")]
            () => MODEL_SE_2X_CONSERVATIVE,
            #[cfg(feature = "models-pro")]
            () => MODEL_PRO_2X_CONSERVATIVE,
            #[cfg(feature = "models-nose")]
            () => MODEL_NOSE_2X_NO_DENOISE,
            _ => (&[], &[]),
        };

        Self {
            gpuid: 0,
            threads: 1,
            tta_mode: false,
            sync_gap: 3,
            tile_size: 0,
            scale_factor: 2,
            noise_level: 0,
            param: bytes.0,
            bin: bytes.1,
        }
    }
}

impl <'a>Options<'a> {

    #[cfg(any(feature = "models-nose", feature = "models-pro", feature = "models-se"))]
    pub fn model(mut self, model: OptionsModel) -> Self {
        let model_bytes = model.get_bytes();
        self.param = model_bytes.0;
        self.bin = model_bytes.1;
        self.scale_factor = model.get_scale_factor();
        self.sync_gap = if model.allow_sync_gap() { self.sync_gap } else { 0 };
        self.noise_level = model.get_noise_level();
        self
    }

    pub fn model_bytes(mut self, param: &'a [u8], bin: &'a [u8]) -> Self {
        self.param = param;
        self.bin = bin;
        self
    }

    pub fn model_files<P: AsRef<Path>>(mut self, param_file: P, bin_file: P) -> Result<Self, std::io::Error> {
        let param_file = std::fs::read(param_file)?;
        let bin_file = std::fs::read(bin_file)?;
        self.param = Box::leak(param_file.into_boxed_slice());
        self.bin = Box::leak(bin_file.into_boxed_slice());
        Ok(self)
    }

    pub fn gpuid(mut self, gpuid: i32) -> Self {
        self.gpuid = gpuid;
        self
    }

    pub fn cpu(mut self) -> Self {
        self.gpuid = -1;
        self
    }

    pub fn threads(mut self, threads: i32) -> Self {
        self.threads = threads;
        self
    }

    pub fn tta_mode(mut self, tta_mode: bool) -> Self {
        self.tta_mode = tta_mode;
        self
    }

    pub fn tile_size(mut self, tile_size: i32) -> Self {
        self.tile_size = tile_size;
        self
    }

    pub fn sync_gap(mut self, sync_gap: OptionsSyncGap) -> Self {
        self.sync_gap = sync_gap as i32;
        self
    }

    pub fn scale_factor(mut self, scale: OptionsScaleFactor) -> Self {
        self.scale_factor = scale as i32;
        self
    }

    pub fn noise_level(mut self, noise: OptionsNoiseLevel) -> Self {
        self.noise_level = noise as i32;
        self
    }

}