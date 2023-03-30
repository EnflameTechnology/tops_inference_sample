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

#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "../utils/tops_utils.h"
#include "TopsInference/TopsInferRuntime.h"

char *getCmdOption(char **begin, char **end, const std::string &option) {
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return 0;
}

bool cmdOptionExists(char **begin, char **end, const std::string &option) {
  return std::find(begin, end, option) != end;
}

template <typename T, typename A> int arg_max(std::vector<T, A> const &vec) {
  return static_cast<int>(
      std::distance(vec.begin(), std::max_element(vec.begin(), vec.end())));
}

int main(int argc, char **argv) {
  if (cmdOptionExists(argv, argv + argc, "-h")) {
    std::cout << "usage: resnet [-h]" << std::endl;
    std::cout << "               [-o ONNX MODEL FILE PATH]" << std::endl;
    std::cout << "               [-i PROCESSED IMAGE(224*224*3) FILE PATH]"
              << std::endl;
    std::cout << "               [-n NAME OF MODEL INPUT, DEFAULT IS: input]"
              << std::endl;
    std::cout << "               [-s INPUT SHAPE, DEFAULT IS: (1,3,224,224)]"
              << std::endl;
    return 0;
  }

  const char *onnx_path;
  if (cmdOptionExists(argv, argv + argc, "-o")) {
    onnx_path = getCmdOption(argv, argv + argc, "-o");
  } else {
    std::cout << "[ERROR] Must specify ONNX model path" << std::endl;
    exit(-1);
  }

  const char *img_path;
  if (cmdOptionExists(argv, argv + argc, "-i")) {
    img_path = getCmdOption(argv, argv + argc, "-i");
  } else {
    img_path = "../../data/resnet/ILSVRC2012_val_00006740.JPEG.dat";
  }

  const char *input_names;
  if (cmdOptionExists(argv, argv + argc, "-n")) {
    input_names = getCmdOption(argv, argv + argc, "-n");
  } else {
    input_names = "input";
  }

  const char *input_shapes;
  if (cmdOptionExists(argv, argv + argc, "-s")) {
    input_shapes = getCmdOption(argv, argv + argc, "-s");
  } else {
    input_shapes = "1,3,224,224";
  }

  int precision_type = 2; // 0: default, 1: fp16, 2: mix_fp16
  std::string exec_path =
      engine_name_construct(onnx_path, "../../engines", atoi(input_shapes),
                            get_precision_str(precision_type));

  int card_id = 0;
  // uint32_t cluster_ids[] = {0, 1, 2, 3, 4, 5};  // use all pg to compute
  // uint32_t cluster_num = 6;                     // count of cluster_ids
  uint32_t cluster_ids[] = {0}; // use one pg to compute
  uint32_t cluster_num = 1;     // count of cluster_ids

  int batch_size = 1;

  // 1. init
  TopsInference::topsInference_init();
  void *tops_handler_ =
      TopsInference::set_device(card_id, cluster_ids, cluster_num);

  // 2. load engine file or build it from onnx file
  TopsInference::IEngine *engine = loadOrCreateEngine(
      exec_path.c_str(), onnx_path,
      TopsInference::BuildFlag::TIF_KTYPE_MIX_FP16, input_names, input_shapes);

  // 3. get input&output info from engine
  std::vector<ShapeInfo> inputs_shape_info = get_inputs_shape(engine);
  std::vector<ShapeInfo> outputs_shape_info = get_outputs_shape(engine);

  //  Debug only
  for (auto &s : inputs_shape_info) {
    s.dump();
  }

  for (auto &s : outputs_shape_info) {
    s.dump();
  }

  // 4. alloc host memory for outputs
  std::vector<void *> inputs = alloc_host_memory(inputs_shape_info, batch_size);
  std::vector<void *> outputs =
      alloc_host_memory(outputs_shape_info, batch_size);

  // 5. prepare data
  std::ifstream in_data_stream;
  int data_size = sizeof(float) * 224 * 224 * 3;
  in_data_stream.open(img_path);
  in_data_stream.read(reinterpret_cast<char *>(inputs[0]), data_size);
  in_data_stream.close();
  // int ind = 0;
  // std::ifstream data_list_file("../../data/resnet/test_file.txt");
  // std::string fname, label;
  // while (data_list_file >> fname >> label) {
  //   in_data_stream.open(data_dir + fname);
  //   in_data_stream.read(reinterpret_cast<char *>(inputs[0] + data_size *
  //   ind),
  //                       data_size);
  //   in_data_stream.close();
  //   ind++;
  //   if (ind >= batch_size) break;
  // }
  // data_list_file.close();

  // warmup
  for (int i = 0; i < 3; ++i) {
    engine->run_with_batch(
        batch_size, inputs.data(), outputs.data(),
        TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST);
  }

  // 6. run
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  auto ret = engine->run_with_batch(
      batch_size, inputs.data(), outputs.data(),
      TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST);
  if (!ret) {
    std::cout << "engine run_with_batch failed." << std::endl;
    exit(-1);
  }
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  auto time_diff =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  std::cout << "-------------------- " << std::endl;
  std::cout << "gcu inference cost time:" << (int)time_diff << " ms"
            << std::endl;
  std::cout << "-------------------- " << std::endl;

  std::cout << "result:" << std::endl;
  for (int i = 0; i < batch_size; i++) {
    float *start = (float *)outputs[0] + 1000 * i;
    auto v = std::vector<float>(start, start + 1000);
    int max_idx = arg_max(v);
    std::cout << "classification idx=" << max_idx << std::endl;
  }
  std::cout << "-------------------- " << std::endl;

  // 7. free host outputs memory
  free_host_memory(inputs);
  free_host_memory(outputs);

  // 8. release
  TopsInference::release_engine(engine);
  TopsInference::release_device(tops_handler_);
  TopsInference::topsInference_finish();

  return 0;
}
