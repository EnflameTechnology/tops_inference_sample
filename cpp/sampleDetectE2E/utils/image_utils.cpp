/*
image utils in Enflame E2E Sample
Copyright (C) <2024> <frank.wang>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <fstream>

#include <stdlib.h>
#include <signal.h>
#include <sys/types.h>

//! MACRO should be defined before include
#define cimg_display 0
#ifndef NOT_USE_PNG
#define cimg_use_png
#endif
#ifndef NOT_USE_JPEG
#define cimg_use_jpeg
#endif

#include "image_utils.h"
#include "utils/CImg.h"

#include "tops/tops_runtime_api.h"

#define ENFLAME_RUNTIME_CHECK(_expr)                                           \
  do {                                                                         \
    if (topsError_t::topsSuccess != _expr) {                                   \
      fprintf(stderr, "EnFlame Runtime ERROR : %s @ %s:%d\n", topsGetErrorString(_expr), __FILE__, __LINE__);\
      raise(SIGUSR1);                                                                \
    }                                                                          \
  } while (0)

namespace e2e_utils {

void save_cpu_image_by_rgb(const std::string& file_name, void* cpu_rgb, size_t width, size_t height) {
    using CImage = cimg_library::CImg<unsigned char>;
    auto fout = fopen("tmp.rgb", "w");
    auto total = fwrite(cpu_rgb, 1, width * height * 3, fout);
    fprintf(stderr, "%zu Bytes Write to : tmp.rgb\n", total);
    fclose(fout);
    CImage rgb;
    rgb.load_rgb("tmp.rgb", width, height);
    rgb.save(file_name.c_str());
    fprintf(stderr, "save image : %s\n", file_name.c_str());
}

void save_gcu_image_by_rgb(const std::string& file_name, void* gcu_rgb, size_t width, size_t height) {
    size_t size_in_bytes = width * height * 3;
    uint8_t *dst = new uint8_t[size_in_bytes];
    ENFLAME_RUNTIME_CHECK(topsMemcpyDtoH(dst, gcu_rgb, size_in_bytes));
    save_cpu_image_by_rgb(file_name, dst, width, height);
    delete []dst;
}

void save_cpu_image_by_yuv(const std::string& file_name, void* cpu_yuv, size_t width, size_t height) {
    using CImage = cimg_library::CImg<unsigned char>;
    auto fout = fopen("tmp.yuv", "w");
    auto total = fwrite(cpu_yuv, 1, width * height * 3 / 2, fout);
    fprintf(stderr, "%zu Bytes Write to : tmp.yuv\n", total);
    fclose(fout);
    CImage rgb;
    rgb.load_yuv("tmp.yuv", width, height, 420);
    rgb.save(file_name.c_str());
    fprintf(stderr, "save image : %s\n", file_name.c_str());
}

}
