#pragma once
#include "TopsInference/TopsInferRuntime.h"
#include <vector>
#include <string>
#include <cstdio>

struct ShapeInfo {
  std::vector<int> dims;
  int dtype_size;
  int mem_size;
  ShapeInfo() {}
  ShapeInfo(std::vector<int> &_dims, int _dtype_size, int _mem_size)
      : dims(_dims), dtype_size(_dtype_size), mem_size(_mem_size) {}

  void dump() {
      printf("dims:");
      for(auto d : dims) {
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

std::vector<TopsInference::TensorPtr_t> get_input_tensor_list(
    TopsInference::IEngine *engine, float **inputs, int batch_size);
  
std::vector<TopsInference::TensorPtr_t> get_output_tensor_list(
    TopsInference::IEngine *engine, void **outputs, int batch_size);
