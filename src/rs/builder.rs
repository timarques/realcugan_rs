use crate::realcugan::RealCugan;

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
    param: include_bytes!("../../models/models-nose/up2x-no-denoise.param"),
    bin: include_bytes!("../../models/models-nose/up2x-no-denoise.bin"),
    scale: 2,
    noise: 0,
    allow_sync_gap: true,
};

#[cfg(feature = "models-pro")]
const MODEL_PRO_2X_NO_DENOISE: ModelParameters = ModelParameters {
    param: include_bytes!("../../models/models-pro/up2x-no-denoise.param"),
    bin: include_bytes!("../../models/models-pro/up2x-no-denoise.bin"),
    scale: 2,
    noise: 0,
    allow_sync_gap: true,
};

#[cfg(feature = "models-pro")]
const MODEL_PRO_2X_CONSERVATIVE: ModelParameters = ModelParameters {
    param: include_bytes!("../../models/models-pro/up2x-conservative.param"),
    bin: include_bytes!("../../models/models-pro/up2x-conservative.bin"),
    scale: 2,
    noise: -1,
    allow_sync_gap: true,
};

#[cfg(feature = "models-pro")]
const MODEL_PRO_2X_DENOISE_X3: ModelParameters = ModelParameters {
    param: include_bytes!("../../models/models-pro/up2x-denoise3x.param"),
    bin: include_bytes!("../../models/models-pro/up2x-denoise3x.bin"),
    scale: 2,
    noise: 3,
    allow_sync_gap: true,
};

#[cfg(feature = "models-pro")]
const MODEL_PRO_3X_NO_DENOISE: ModelParameters = ModelParameters {
    param: include_bytes!("../../models/models-pro/up3x-no-denoise.param"),
    bin: include_bytes!("../../models/models-pro/up3x-no-denoise.bin"),
    scale: 3,
    noise: 0,
    allow_sync_gap: true,
};

#[cfg(feature = "models-pro")]
const MODEL_PRO_3X_CONSERVATIVE: ModelParameters = ModelParameters {
    param: include_bytes!("../../models/models-pro/up3x-conservative.param"),
    bin: include_bytes!("../../models/models-pro/up3x-conservative.bin"),
    scale: 3,
    noise: -1,
    allow_sync_gap: true,
};

#[cfg(feature = "models-pro")]
const MODEL_PRO_3X_DENOISE_X3: ModelParameters = ModelParameters {
    param: include_bytes!("../../models/models-pro/up3x-denoise3x.param"),
    bin: include_bytes!("../../models/models-pro/up3x-denoise3x.bin"),
    scale: 3,
    noise: 3,
    allow_sync_gap: true,
};

#[cfg(feature = "models-se")]
const MODEL_SE_2X_NO_DENOISE: ModelParameters = ModelParameters {
    param: include_bytes!("../../models/models-se/up2x-no-denoise.param"),
    bin: include_bytes!("../../models/models-se/up2x-no-denoise.bin"),
    scale: 2,
    noise: 0,
    allow_sync_gap: false,
};

#[cfg(feature = "models-se")]
const MODEL_SE_2X_CONSERVATIVE: ModelParameters = ModelParameters {
    param: include_bytes!("../../models/models-se/up2x-conservative.param"),
    bin: include_bytes!("../../models/models-se/up2x-conservative.bin"),
    scale: 2,
    noise: -1,
    allow_sync_gap: false,
};

#[cfg(feature = "models-se")]
const MODEL_SE_2X_DENOISE_X1: ModelParameters = ModelParameters {
    param: include_bytes!("../../models/models-se/up2x-denoise1x.param"),
    bin: include_bytes!("../../models/models-se/up2x-denoise1x.bin"),
    scale: 2,
    noise: 1,
    allow_sync_gap: false,
};

#[cfg(feature = "models-se")]
const MODEL_SE_2X_DENOISE_X2: ModelParameters = ModelParameters {
    param: include_bytes!("../../models/models-se/up2x-denoise2x.param"),
    bin: include_bytes!("../../models/models-se/up2x-denoise2x.bin"),
    scale: 2,
    noise: 2,
    allow_sync_gap: false,
};

#[cfg(feature = "models-se")]
const MODEL_SE_2X_DENOISE_X3: ModelParameters = ModelParameters {
    param: include_bytes!("../../models/models-se/up2x-denoise3x.param"),
    bin: include_bytes!("../../models/models-se/up2x-denoise3x.bin"),
    scale: 2,
    noise: 3,
    allow_sync_gap: false,
};

#[cfg(feature = "models-se")]
const MODEL_SE_3X_NO_DENOISE: ModelParameters = ModelParameters {
    param: include_bytes!("../../models/models-se/up3x-no-denoise.param"),
    bin: include_bytes!("../../models/models-se/up3x-no-denoise.bin"),
    scale: 3,
    noise: 0,
    allow_sync_gap: false,
};

#[cfg(feature = "models-se")]
const MODEL_SE_3X_CONSERVATIVE: ModelParameters = ModelParameters {
    param: include_bytes!("../../models/models-se/up3x-conservative.param"),
    bin: include_bytes!("../../models/models-se/up3x-conservative.bin"),
    scale: 3,
    noise: -1,
    allow_sync_gap: false,
};

#[cfg(feature = "models-se")]
const MODEL_SE_3X_DENOISE_X3: ModelParameters = ModelParameters {
    param: include_bytes!("../../models/models-se/up3x-denoise3x.param"),
    bin: include_bytes!("../../models/models-se/up3x-denoise3x.bin"),
    scale: 3,
    noise: 3,
    allow_sync_gap: false,
};

#[cfg(feature = "models-se")]
const MODEL_SE_4X_NO_DENOISE: ModelParameters = ModelParameters {
    param: include_bytes!("../../models/models-se/up4x-no-denoise.param"),
    bin: include_bytes!("../../models/models-se/up4x-no-denoise.bin"),
    scale: 4,
    noise: 0,
    allow_sync_gap: false,
};

#[cfg(feature = "models-se")]
const MODEL_SE_4X_CONSERVATIVE: ModelParameters = ModelParameters {
    param: include_bytes!("../../models/models-se/up4x-conservative.param"),
    bin: include_bytes!("../../models/models-se/up4x-conservative.bin"),
    scale: 4,
    noise: -1,
    allow_sync_gap: false,
};

#[cfg(feature = "models-se")]
const MODEL_SE_4X_DENOISE_X3: ModelParameters = ModelParameters {
    param: include_bytes!("../../models/models-se/up4x-denoise3x.param"),
    bin: include_bytes!("../../models/models-se/up4x-denoise3x.bin"),
    scale: 4,
    noise: 3,
    allow_sync_gap: false,
};