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
#include "../utils/tops_utils.h"
#include "TopsInference/TopsInferRuntime.h"

#include <iostream>
#include <vector>

class TopsStreamStruct {
public:
  TopsStreamStruct(std::vector<ShapeInfo> &_inputs_info,
                   std::vector<ShapeInfo> &_outputs_info) {
    inputs_info = _inputs_info;
    outputs_info = _outputs_info;
    int input_num = inputs_info.size();
    int output_num = outputs_info.size();
    device_inputs = new void *[input_num];
    device_outputs = new void *[output_num];
    // allocate buffer on device
    for (int i = 0; i < input_num; i++) {
      std::cout << "[INFO] mem_alloc input, size: " << inputs_info[i].mem_size
                << std::endl;
      TopsInference::mem_alloc(&device_inputs[i], inputs_info[i].mem_size);
    }
    for (int i = 0; i < output_num; i++) {
      std::cout << "[INFO] mem_alloc output, size: " << outputs_info[i].mem_size
                << std::endl;
      TopsInference::mem_alloc(&device_outputs[i], outputs_info[i].mem_size);
    }
    TopsInference::create_stream(&stream);
  }

  ~TopsStreamStruct() {
    int input_num = inputs_info.size();
    int output_num = outputs_info.size();
    // free memory on device
    for (int i = 0; i < input_num; i++) {
      TopsInference::mem_free(device_inputs[i]);
    }
    for (int i = 0; i < output_num; i++) {
      TopsInference::mem_free(device_outputs[i]);
    }

    TopsInference::destroy_stream(stream);
    delete[] device_inputs;
    delete[] device_outputs;
  }

public:
  void **device_inputs;
  void **device_outputs;
  std::vector<ShapeInfo> inputs_info;
  std::vector<ShapeInfo> outputs_info;
  // create stream to bond with async actions
  TopsInference::topsInferStream_t stream;
};