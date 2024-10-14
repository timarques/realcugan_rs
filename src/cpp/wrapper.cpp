
#ifndef REALCUGAN_WRAPPER_H
#define REALCUGAN_WRAPPER_H

#include "realcugan.h"

int realcugan_get_default_tile_size(int gpuid, int scale) {
	int tilesize = 0;
	uint32_t heap_budget = ncnn::get_gpu_device(gpuid)->get_heap_budget();
	if (scale == 2) {
		if (heap_budget > 1300)
			tilesize = 400;
		else if (heap_budget > 800)
			tilesize = 300;
		else if (heap_budget > 400)
			tilesize = 200;
		else if (heap_budget > 200)
			tilesize = 100;
		else
			tilesize = 32;
	}
	if (scale == 3) {
		if (heap_budget > 3300)
			tilesize = 400;
		else if (heap_budget > 1900)
			tilesize = 300;
		else if (heap_budget > 950)
			tilesize = 200;
		else if (heap_budget > 320)
			tilesize = 100;
		else
			tilesize = 32;
	}
	if (scale == 4) {
		if (heap_budget > 1690)
			tilesize = 400;
		else if (heap_budget > 980)
			tilesize = 300;
		else if (heap_budget > 530)
			tilesize = 200;
		else if (heap_budget > 240)
			tilesize = 100;
		else
			tilesize = 32;
	}
	return tilesize;
}

extern "C" RealCUGAN *realcugan_init(
	int gpuid, 
	bool tta_mode, 
	int num_threads,
	int scale,
	int noise,
	int syncgap,
	int tilesize
) {
	RealCUGAN* realcugan = new RealCUGAN(gpuid, tta_mode, num_threads);
	realcugan->noise = noise;
	realcugan->scale = scale;
	realcugan->syncgap = syncgap;
	realcugan->tilesize = tilesize == 0 ? realcugan_get_default_tile_size(gpuid, scale) : tilesize;
	if (scale == 2) {
		realcugan->prepadding = 18;
	} else if (scale == 3) {
		realcugan->prepadding = 14;
	} else if (scale == 4) {
		realcugan->prepadding = 19;
	};
	return realcugan;
}

extern "C" int realcugan_get_gpu_count() {
  return ncnn::get_gpu_count();
}

extern "C" void realcugan_destroy_gpu_instance() {
  ncnn::destroy_gpu_instance();
}

extern "C" int realcugan_load_files(
  RealCUGAN *realcugan,
  FILE* param,
  FILE* bin
) {
  return realcugan->load_files(param, bin);
}

extern "C" int realcugan_process(
    RealCUGAN *realcugan,
    unsigned char *input_data,
    unsigned char *output_data,
    int width,
    int height,
    int channels
) {
    ncnn::Mat in_image_mat = ncnn::Mat(width, height, (void *)input_data, channels, channels);
    ncnn::Mat out_image_mat = ncnn::Mat(width * realcugan->scale, height * realcugan->scale, (void *)output_data, channels, channels);
    return realcugan->process(in_image_mat, out_image_mat);
}

extern "C" int realcugan_process_cpu(
    RealCUGAN *realcugan,
    unsigned char *input_data,
    unsigned char *output_data,
    int width,
    int height,
    int channels
) {
    ncnn::Mat in_image_mat = ncnn::Mat(width, height, (void *)input_data, channels, channels);
    ncnn::Mat out_image_mat = ncnn::Mat(width * realcugan->scale, height * realcugan->scale, (void *)output_data, channels, channels);
    return realcugan->process_cpu(in_image_mat, out_image_mat);
}

extern "C" void realcugan_free(RealCUGAN *realcugan) {
	delete realcugan;
}

#endif // REALCUGAN_WRAPPER_H