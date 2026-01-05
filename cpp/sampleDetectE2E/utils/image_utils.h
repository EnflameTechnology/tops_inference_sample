#ifndef __IMAGE_UTILS_H__
#define __IMAGE_UTILS_H__

#include <string>

namespace e2e_utils {

void save_gcu_image_by_rgb(const std::string& file_name, void* gcu_rgb, size_t width, size_t height);

void save_cpu_image_by_rgb(const std::string& file_name, void* cpu_rgb, size_t width, size_t height);

void save_cpu_image_by_yuv(const std::string& file_name, void* cpu_yuv, size_t width, size_t height);

}

#endif