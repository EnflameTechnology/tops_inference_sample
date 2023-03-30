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
#include <map>
#include <vector>

#include "../utils/tops_utils.h"
#include "TopsInference/TopsInferRuntime.h"

std::map<int, std::string> label_map = {
    {0, "person"},         {1, "bicycle"},       {2, "car"},
    {3, "motorbike"},      {4, "aeroplane"},     {5, "bus"},
    {6, "train"},          {7, "truck"},         {8, "boat"},
    {9, "traffic_light"},  {10, "fire_hydrant"}, {11, "stop_sign"},
    {12, "parking_meter"}, {13, "bench"},        {14, "bird"},
    {15, "cat"},           {16, "dog"},          {17, "horse"},
    {18, "sheep"},         {19, "cow"},          {20, "elephant"},
    {21, "bear"},          {22, "zebra"},        {23, "giraffe"},
    {24, "backpack"},      {25, "umbrella"},     {26, "handbag"},
    {27, "tie"},           {28, "suitcase"},     {29, "frisbee"},
    {30, "skis"},          {31, "snowboard"},    {32, "sports_ball"},
    {33, "kite"},          {34, "baseball_bat"}, {35, "baseball_glove"},
    {36, "skateboard"},    {37, "surfboard"},    {38, "tennis_racket"},
    {39, "bottle"},        {40, "wine_glass"},   {41, "cup"},
    {42, "fork"},          {43, "knife"},        {44, "spoon"},
    {45, "bowl"},          {46, "banana"},       {47, "apple"},
    {48, "sandwich"},      {49, "orange"},       {50, "broccoli"},
    {51, "carrot"},        {52, "hot_dog"},      {53, "pizza"},
    {54, "donut"},         {55, "cake"},         {56, "chair"},
    {57, "sofa"},          {58, "pottedplant"},  {59, "bed"},
    {60, "diningtable"},   {61, "toilet"},       {62, "tvmonitor"},
    {63, "laptop"},        {64, "mouse"},        {65, "remote"},
    {66, "keyboard"},      {67, "cell_phone"},   {68, "microwave"},
    {69, "oven"},          {70, "toaster"},      {71, "sink"},
    {72, "refrigerator"},  {73, "book"},         {74, "clock"},
    {75, "vase"},          {76, "scissors"},     {77, "teddy_bear"},
    {78, "hair_drier"},    {79, "toothbrush"},
};

typedef struct Point {
  float x;
  float y;
} Point;

typedef struct Rect {
  float left;
  float top;
  float width;
  float height;

  Rect(float l, float t, float w, float h)
      : left(l), top(t), width(w), height(h) {}

  void dump() {
    std::cout << "(" << left << "," << top << "," << width << "," << height
              << ")" << std::endl;
  }
} Rect;

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

void clamp(float &val, const float low, const float high) {
  if (val > high) {
    val = high;
  } else if (val < low) {
    val = low;
  }
}

static void sort(int n, const std::vector<float> x, std::vector<int> indices) {
  int i, j;
  for (i = 0; i < n; i++)
    for (j = i + 1; j < n; j++) {
      if (x[indices[j]] > x[indices[i]]) {
        // float x_tmp = x[i];
        int index_tmp = indices[i];
        // x[i] = x[j];
        indices[i] = indices[j];
        // x[j] = x_tmp;
        indices[j] = index_tmp;
      }
    }
}

bool nonMaximumSuppression(const std::vector<Rect> rects,
                           const std::vector<float> score,
                           float overlap_threshold,
                           std::vector<int> &index_out) {
  int num_boxes = rects.size();
  int i, j;
  std::vector<float> box_area(num_boxes);
  std::vector<int> indices(num_boxes);
  std::vector<int> is_suppressed(num_boxes);

  for (i = 0; i < num_boxes; i++) {
    indices[i] = i;
    is_suppressed[i] = 0;
    box_area[i] = (float)((rects[i].width + 1) * (rects[i].height + 1));
  }

  sort(num_boxes, score, indices);

  for (i = 0; i < num_boxes; i++) {
    if (!is_suppressed[indices[i]]) {
      for (j = i + 1; j < num_boxes; j++) {
        if (!is_suppressed[indices[j]]) {
          int x1max = std::max(rects[indices[i]].left, rects[indices[j]].left);
          int x2min =
              std::min(rects[indices[i]].left + rects[indices[i]].width,
                       rects[indices[j]].left + rects[indices[j]].width);
          int y1max = std::max(rects[indices[i]].top, rects[indices[j]].top);
          int y2min =
              std::min(rects[indices[i]].top + rects[indices[i]].height,
                       rects[indices[j]].top + rects[indices[j]].height);
          int overlap_w = x2min - x1max + 1;
          int overlap_h = y2min - y1max + 1;
          if (overlap_w > 0 && overlap_h > 0) {
            float iou = (overlap_w * overlap_h) /
                        (box_area[indices[j]] + box_area[indices[i]] -
                         overlap_w * overlap_h);
            if (iou > overlap_threshold) {
              is_suppressed[indices[j]] = 1;
            }
          }
        }
      }
    }
  }

  for (i = 0; i < num_boxes; i++) {
    if (!is_suppressed[i])
      index_out.push_back(i);
  }

  return true;
}

std::vector<Rect> postProcess(std::vector<float> &pred, const float img_height,
                              const float img_width, size_t bbox_num,
                              size_t bbox_len, const float conf_thres,
                              const float iou_thres) {
  std::vector<std::vector<float>> bboxes;
  for (size_t i = 0; i < bbox_num; i++) {
    std::vector<float> det;
    for (size_t j = 0; j < bbox_len; j++) {
      det.push_back(pred[i * 85 + j]);
    }
    bboxes.push_back(det);
  }

  // FIXME: scale is always 1.0 in this ppm case
  float scale = std::min(640.0 / img_width, 640.0 / img_height);
  std::vector<Rect> selected_boxes;
  std::vector<float> confidence;
  std::vector<float> class_id;

  // conf filter, calculate cls conf
  for (std::vector<float> box : bboxes) {
    float conf = box[4];
    if (conf > conf_thres) {
      confidence.push_back(conf);

      float max_conf = 0.0;
      int id = 0;

      for (size_t i = 5; i < box.size(); i++) {
        box[i] *= conf;
        if (box[i] > max_conf) {
          max_conf = box[i];
          id = i - 5;
        }
      }

      class_id.push_back(id);
      float centerX = box[0];
      float centerY = box[1];
      float width = box[2];
      float height = box[3];
      float left = centerX - width / 2.0;
      float top = centerY - height / 2.0;
      selected_boxes.push_back(Rect(left, top, width, height));
    }
  }

  // no object at all
  if (selected_boxes.size() == 0) {
    std::cout << "no bbox over score threshold detected." << std::endl;
    return {};
  }

  std::vector<Rect> det_result;
  std::vector<Rect> nms_out_rects;
  std::vector<int> indexes;
  nonMaximumSuppression(selected_boxes, confidence, iou_thres, indexes);

  for (int id : indexes) {
    det_result.push_back(selected_boxes[id]);
    std::cout << "cls: " << label_map[class_id[id]]
              << " conf: " << confidence[id] << " (" << selected_boxes[id].left
              << "," << selected_boxes[id].top << ","
              << selected_boxes[id].width << "," << selected_boxes[id].height
              << ")" << std::endl;
  }

  for (Rect &result_box : det_result) {
    result_box.left = (result_box.left) / scale;
    result_box.top = (result_box.top) / scale;
    result_box.width = (result_box.width) / scale;
    result_box.height = (result_box.height) / scale;

    clamp(result_box.left, 0, img_width);
    clamp(result_box.top, 0, img_height);
    clamp(result_box.width, 0, img_width);
    clamp(result_box.height, 0, img_height);
  }

  return det_result;
}

void writePPMFileWithBBox(const std::string &filename, uint8_t *ppm,
                          const std::vector<Rect> &bboxs, int ppm_h, int ppm_w,
                          int ppm_max) {
  std::ofstream outfile("./" + filename, std::ofstream::binary);
  assert(!outfile.fail());

  outfile << "P6"
          << "\n"
          << ppm_w << " " << ppm_h << "\n"
          << ppm_max << "\n";

  auto round = [](float x) -> int { return int(std::floor(x + 0.5f)); };

  for (auto &bbox : bboxs) {
    const int x1 = std::min(std::max(0, round(int(bbox.left))), ppm_w - 1);
    const int x2 =
        std::min(std::max(0, round(int(bbox.left + bbox.width))), ppm_w - 1);
    const int y1 = std::min(std::max(0, round(int(bbox.top))), ppm_h - 1);
    const int y2 =
        std::min(std::max(0, round(int(bbox.top + bbox.height))), ppm_h - 1);

    for (int x = x1; x <= x2; ++x) {
      // bbox top border
      ppm[(y1 * ppm_w + x) * 3] = 255;
      ppm[(y1 * ppm_w + x) * 3 + 1] = 0;
      ppm[(y1 * ppm_w + x) * 3 + 2] = 0;
      // bbox bottom border
      ppm[(y2 * ppm_w + x) * 3] = 255;
      ppm[(y2 * ppm_w + x) * 3 + 1] = 0;
      ppm[(y2 * ppm_w + x) * 3 + 2] = 0;
    }

    for (int y = y1; y <= y2; ++y) {
      // bbox left border
      ppm[(y * ppm_w + x1) * 3] = 255;
      ppm[(y * ppm_w + x1) * 3 + 1] = 0;
      ppm[(y * ppm_w + x1) * 3 + 2] = 0;
      // bbox right border
      ppm[(y * ppm_w + x2) * 3] = 255;
      ppm[(y * ppm_w + x2) * 3 + 1] = 0;
      ppm[(y * ppm_w + x2) * 3 + 2] = 0;
    }
  }

  outfile.write(reinterpret_cast<char *>(ppm), ppm_w * ppm_h * 3);
}

int main(int argc, char **argv) {
  if (cmdOptionExists(argv, argv + argc, "-h")) {
    std::cout << "usage: yolov5s [-h]" << std::endl;
    std::cout << "               [-o ONNX MODEL FILE PATH]" << std::endl;
    std::cout << "               [-i PPM IMAGE(640*640*3) FILE PATH]"
              << std::endl;
    std::cout << "               [-n NAME OF MODEL INPUT, DEFAULT IS: images]"
              << std::endl;
    std::cout << "               [-s INPUT SHAPE, DEFAULT IS: (1,3,640,640)]"
              << std::endl;
    std::cout << "               [-c SCORE THRESHOLD, DEFAULT IS: 0.25]"
              << std::endl;
    std::cout << "               [-u IOU THRESHOLD, DEFAULT IS: 0.45]"
              << std::endl;

    return 0;
  }

  const char *onnx_path;
  if (cmdOptionExists(argv, argv + argc, "-o")) {
    onnx_path = getCmdOption(argv, argv + argc, "-o");
  } else {
    // onnx_path = "../../models/yolov5s.onnx";
    std::cout << "[ERROR] Must specify ONNX model path" << std::endl;
    exit(-1);
  }

  const char *img_path;
  if (cmdOptionExists(argv, argv + argc, "-i")) {
    img_path = getCmdOption(argv, argv + argc, "-i");
  } else {
    img_path = "../../data/yolov5/zidane640.ppm";
  }

  const char *input_names;
  if (cmdOptionExists(argv, argv + argc, "-n")) {
    input_names = getCmdOption(argv, argv + argc, "-n");
  } else {
    input_names = "images";
  }

  const char *input_shapes;
  if (cmdOptionExists(argv, argv + argc, "-s")) {
    input_shapes = getCmdOption(argv, argv + argc, "-s");
  } else {
    input_shapes = "1,3,640,640";
  }

  float conf_thres = 0.25;
  float iou_thres = 0.45;

  char *conf_thres_char;
  if (cmdOptionExists(argv, argv + argc, "-c")) {
    conf_thres_char = getCmdOption(argv, argv + argc, "-c");
    conf_thres = atof(conf_thres_char);
  }

  char *iou_thres_char;
  if (cmdOptionExists(argv, argv + argc, "-u")) {
    iou_thres_char = getCmdOption(argv, argv + argc, "-u");
    iou_thres = atof(iou_thres_char);
  }

  int precision_type = 2; //

  // Note: Now only support inference on the preset 640 * 640 ppm
  int img_w = 640;
  int img_h = 640;

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
  std::vector<void *> outputs =
      alloc_host_memory(outputs_shape_info, batch_size);

  // 5. prepare data
  std::string fname, label;
  std::ifstream ppm_file;

  ppm_file.open(img_path);

  int ppm_h, ppm_w, ppm_max;
  std::string magic;
  ppm_file >> magic >> ppm_w >> ppm_h >> ppm_max;
  std::cout << "magic: " << magic << ", w: " << ppm_w << ", h: " << ppm_h
            << ", max: " << ppm_max << std::endl;

  ppm_file.seekg(1, ppm_file.cur);

  std::vector<uint8_t> ppm_data;
  size_t ppm_size = ppm_w * ppm_h * 3;
  ppm_data.resize(ppm_size);

  uint8_t *p = static_cast<uint8_t *>(ppm_data.data());
  ppm_file.read(reinterpret_cast<char *>(p), ppm_size);

  // HWC->CHW
  std::vector<float> tensor;
  for (int channel = 0; channel < 3; ++channel) {
    for (size_t i = channel; i < ppm_size;) {
      tensor.push_back(p[i] / 255.0);
      i += 3;
    }
  }

  ppm_file.close();

  // 6. run
  std::vector<void *> inputs_list;
  inputs_list.push_back(tensor.data());

  // warmup
  for (int i = 0; i < 3; ++i) {
    auto ret = engine->run_with_batch(
        batch_size, inputs_list.data(), outputs.data(),
        TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST);
    if (!ret) {
      std::cout << "engine run_with_batch failed." << std::endl;
      exit(-1);
    }
  }

  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  auto ret = engine->run_with_batch(
      batch_size, inputs_list.data(), outputs.data(),
      TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST);
  if (!ret) {
    std::cout << "engine run_with_batch failed." << std::endl;
    exit(-1);
  }

  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  auto time_diff =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  std::cout << "-------------------- " << std::endl;
  std::cout << "gcu inference took time:" << (int)time_diff << " ms"
            << std::endl;
  std::cout << "-------------------- " << std::endl;

  // 7. postprocess output
  std::vector<float> pred_output;

  size_t datum_num = outputs_shape_info[0].mem_size /
                     outputs_shape_info[0].dtype_size; // 22500 * 85
  float *pred_p = static_cast<float *>(outputs[0]);
  for (size_t i = 0; i < datum_num; ++i) {
    pred_output.push_back(pred_p[i]);
  }

  size_t bbox_num = outputs_shape_info[0].dims[1];
  size_t bbox_len = outputs_shape_info[0].dims[2];
  auto result = postProcess(pred_output, img_h, img_w, bbox_num, bbox_len,
                            conf_thres, iou_thres);

  // for debug
  // std::cout << "size of result: " << result.size() << std::endl;
  //// dump
  // for (auto r : result) {
  // r.dump();
  //}

  // 8: Drawing and save
  writePPMFileWithBBox("./result.ppm", static_cast<uint8_t *>(ppm_data.data()),
                       result, ppm_h, ppm_w, ppm_max);

  // 9. free host outputs memory
  free_host_memory(outputs);

  // 10. release
  TopsInference::release_engine(engine);
  TopsInference::release_device(tops_handler_);
  TopsInference::topsInference_finish();

  return 0;
}
