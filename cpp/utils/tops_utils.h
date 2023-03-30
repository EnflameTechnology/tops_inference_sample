/*=======================================================================
 * Copyright 2020-2023 Enflame. All Rights Reserved.
 *
 *Licensed under the Apache License, Version 2.0 (the "License");
 *you may not use this file except in compliance with the License.
 *You may obtain a copy of the License at
 *
 *http://www.apache.org/licenses/LICENSE-2.0
 *
 *Unless required by applicable law or agreed to in writing, software
 *distributed under the License is distributed on an "AS IS" BASIS,
 *WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *See the License for the specific language governing permissions and
 *limitations under the License.
 *=======================================================================
 */

#pragma once
#include "TopsInference/TopsInferRuntime.h"
#include <cstdio>
#include <vector>

struct ShapeInfo {
  std::vector<int> dims;
  int dtype_size;
  int mem_size;
  ShapeInfo() {}
  ShapeInfo(std::vector<int> &_dims, int _dtype_size, int _mem_size)
      : dims(_dims), dtype_size(_dtype_size), mem_size(_mem_size) {}

  void dump() {
    printf("dims:");
    for (auto d : dims) {
      printf("%d,", d);
    }
    printf("\n");

    printf("dtype_size:%d\n", dtype_size);
    printf("mes_size:%d\n", mem_size);
  }
};

const char *get_precision_str(int precision_type);
std::string engine_name_construct(const char *onnx_path,
                                  const char *engine_folder, int batchsize,
                                  const char *precision);
std::vector<ShapeInfo> get_inputs_shape(TopsInference::IEngine *engine);
std::vector<ShapeInfo> get_outputs_shape(TopsInference::IEngine *engine);
std::vector<void *> alloc_host_memory(std::vector<ShapeInfo> &shapes_info,
                                      int times = 1, bool verbose = false);
void free_host_memory(std::vector<void *> &datum);
bool getEngineIOInfo(std::string &exec_path,
                     std::vector<ShapeInfo> &inputs_shape_info,
                     std::vector<ShapeInfo> &outputs_shape_info);
TopsInference::IEngine *
loadOrCreateEngine(const char *exec_path, const char *onnx_path,
                   TopsInference::BuildFlag precision_flag,
                   const char *input_names = NULL,
                   const char *input_shapes = NULL);
