#if _WIN32
#include <codecvt>
#include <locale>
#include <string>
#endif

#include "realcugan.h"
#include "cpu.h"
#include <algorithm>
#include <vector>
#include <map>

typedef struct Image {
  unsigned char *data;
  int w;
  int h;
  int c;
} Image;

extern "C" RealCUGAN *realcugan_init(int gpuid, bool tta_mode, int num_threads) {
  return new RealCUGAN(gpuid, tta_mode, num_threads);
}

extern "C" int realcugan_get_gpu_count() { return ncnn::get_gpu_count(); }

extern "C" void realcugan_destroy_gpu_instance() {
  ncnn::destroy_gpu_instance();
}

extern "C" int realcugan_load(RealCUGAN *realcugan, const char *param_path,
                              const char *model_path) {
#if _WIN32
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  return realcugan->load(converter.from_bytes(param_path),
                         converter.from_bytes(model_path));
#else
  return realcugan->load(param_path, model_path);
#endif
}

extern "C" void realcugan_set_parameters(RealCUGAN *realcugan,
                                        int scale, int noise,
                                        int prepadding, int syncgap,
                                        int tilesize) {
  realcugan->noise = noise;
  realcugan->scale = scale;
  realcugan->prepadding = prepadding;
  realcugan->syncgap = syncgap;
  realcugan->tilesize = tilesize;
}

extern "C" int realcugan_process(RealCUGAN *realcugan, const Image *in_image,
                                 Image *out_image, void **mat_ptr) {
  int c = in_image->c;
  ncnn::Mat in_image_mat =
      ncnn::Mat(in_image->w, in_image->h, (void *)in_image->data, (size_t)c, c);

  auto *out_image_mat = new ncnn::Mat(out_image->w, out_image->h, (size_t)c, c);

  int result = realcugan->process(in_image_mat, *out_image_mat);
  out_image->data = static_cast<unsigned char *>(out_image_mat->data);
  *mat_ptr = out_image_mat;
  return result;
}

extern "C" int realcugan_process_cpu(RealCUGAN *realcugan,
                                     const Image *in_image, Image *out_image,
                                     void **mat_ptr) {
  int c = in_image->c;
  ncnn::Mat in_image_mat =
      ncnn::Mat(in_image->w, in_image->h, (void *)in_image->data, (size_t)c, c);
  auto *out_image_mat = new ncnn::Mat(out_image->w, out_image->h, (size_t)c, c);

  int result = realcugan->process_cpu(in_image_mat, *out_image_mat);
  out_image->data = static_cast<unsigned char *>(out_image_mat->data);
  *mat_ptr = out_image_mat;
  return result;
}

extern "C" uint32_t realcugan_get_heap_budget(int gpuid) {
  return ncnn::get_gpu_device(gpuid)->get_heap_budget();
}

extern "C" void realcugan_free_image(ncnn::Mat *mat_ptr) { delete mat_ptr; }

extern "C" void realcugan_free(RealCUGAN *realcugan) {
  delete realcugan;
  ncnn::destroy_gpu_instance();
}