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

#include "arg.hpp"
#include <TopsInference/TopsInferRuntime.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

template <typename T> struct Mat {
  Mat(int ch, int w, int h) {
    shape = {ch, h, w};
    data = new T[size()];
  }
  explicit Mat(const std::array<int, 3> &sp) {
    shape = sp;
    data = new T[size()];
  }
  ~Mat() { delete data; }
  size_t size() { return shape[0] * shape[1] * shape[2]; }
  T &operator[](size_t idx) { return data[idx]; }
  const T *ptr(size_t idx) const { return data + idx; }
  T *ptr(size_t idx) { return data + idx; }
  T *data;
  std::array<int, 3> shape;
};

template <typename T0, typename T1>
void resize(const T0 *pIn, T1 *pOut, int widthIn, int heightIn, int widthOut,
            int heightOut, const std::function<T1(T0)> &func) {
  for (int i = 0; i < heightOut; i++) {
    int i_in = i * heightIn / heightOut;
    for (int j = 0; j < widthOut; j++) {
      int j_in = j * widthIn / widthOut;
      pOut[i * widthOut + j] = func(pIn[i_in * widthIn + j_in]);
    }
  }
}

template <typename Iterator, typename Func, typename Distance>
void batch(Iterator begin, Iterator end, Distance k, Func f) {
  Iterator batchbegin;
  Iterator batchend;
  batchend = batchbegin = begin;

  do {
    if (std::distance(batchend, end) < k)
      batchend = end;
    else
      std::advance(batchend, k);
    f(batchbegin, batchend);
    batchbegin = batchend;
  } while (std::distance(batchbegin, end) > 0);
}

using Box = std::array<float, 6>;
using Boxes = std::vector<Box>;

static std::unique_ptr<Boxes> nms(const Boxes &pred, float thresh) {
  auto nms_pred = std::make_unique<Boxes>();
  std::vector<float> x1(pred.size());
  std::vector<float> y1(pred.size());
  std::vector<float> x2(pred.size());
  std::vector<float> y2(pred.size());
  std::vector<float> areas(pred.size());
  struct {
    float score;
    int id;
  } data_id;
  std::vector<decltype(data_id)> scoreid(pred.size());

  for (int i = 0; i < pred.size(); i++) {
    x1[i] = pred[i][0];
    y1[i] = pred[i][1];
    x2[i] = pred[i][2];
    y2[i] = pred[i][3];
    areas[i] = (x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1);
    data_id.score = pred[i][4];
    data_id.id = i;
    scoreid[i] = data_id;
  }

  std::sort(scoreid.begin(), scoreid.end(),
            [](decltype(data_id) a, decltype(data_id) b) -> bool {
              return a.score > b.score;
            });
  std::vector<float> keep;
  std::vector<float> suppressed(pred.size());
  for (int ti = 0; ti < pred.size(); ti++) {
    int i = scoreid[ti].id;
    if (suppressed[i] == 1)
      continue;
    keep.push_back(i);
    nms_pred->push_back(pred[i]);
    float ix1 = x1[i];
    float iy1 = y1[i];
    float ix2 = x2[i];
    float iy2 = y2[i];
    float iarea = areas[i];
    for (int tj = ti + 1; tj < pred.size(); tj++) {
      int j = scoreid[tj].id;
      if (suppressed[j] == 1)
        continue;
      float xx1 = std::max(ix1, x1[j]);
      float yy1 = std::max(iy1, y1[j]);
      float xx2 = std::min(ix2, x2[j]);
      float yy2 = std::min(iy2, y2[j]);
      float w = std::max(0.F, xx2 - xx1 + 1);
      float h = std::max(0.F, yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (iarea + areas[j] - inter);
      if (ovr >= thresh) {
        suppressed[j] = 1;
      }
    }
  }
  return nms_pred;
}
static int k_step;
static int k_proposal_count;
static int k_kn_classes;
static std::unique_ptr<Boxes>
yolov5_postprocess(const float *p, const std::array<int, 3> &rawshape,
                   const std::array<int, 3> &shape) {
  auto boxes = std::make_unique<Boxes>();
  const int step = k_step;
  const int proposal_count = k_proposal_count;
  const float score_threshold = 0.5;
  const float nms_threshold = 0.45;
  const float scalew = shape[2] * 1.0 / rawshape[2];
  const float scaleh = shape[1] * 1.0 / rawshape[1];
  std::map<int, Boxes> cls_of_boxes;
  for (size_t i = 0; i < proposal_count; i++) {
    float obj_score = p[4];
    if (obj_score < score_threshold) {
      p += step;
      continue;
    }
    int c_id = -1;
    float c_score = 0;
    for (size_t j = 5; j < step; j++) {
      if (p[j] > c_score) {
        c_score = p[j];
        c_id = j - 5;
      }
    }
    Box box = {(p[0] - p[2] / 2) / scalew,
               (p[1] - p[3] / 2) / scaleh,
               (p[0] + p[2] / 2) / scalew,
               (p[1] + p[3] / 2) / scaleh,
               c_score,
               static_cast<float>(c_id)};
    if (cls_of_boxes.find(c_id) == cls_of_boxes.end()) {
      cls_of_boxes.emplace(c_id, Boxes({box}));
    } else {
      cls_of_boxes[c_id].push_back(box);
    }
    p += step;
  }

  for (const auto &v : cls_of_boxes) {
    auto nms_v = nms(v.second, nms_threshold);
    for (const auto &nv : *nms_v) {
      boxes->push_back(nv);
    }
  }

  return boxes;
}

int main(int argc, const char **targv) {
  arg::ArgParser argparser("sampleSimpleSerialPipeline",
                           "This program is a simple demonstration about dtu "
                           "serial pipeline execution.");
  argparser.add(
      {"--vg"},
      "number of VG (In this demostration, two models use the same vg)", true);
  argparser.add({"--det_buffersize"}, "buffer size (detection model) ", true);
  argparser.add({"--det_modelpath"}, "onnx model path (detection model) ",
                true);
  argparser.add({"--det_inputname"}, "onnx model input name (detection model) ",
                true);
  argparser.add({"--det_shape"}, "input shape (detection model) ", true);
  argparser.add({"--cls_buffersize"}, "buffer size (classification model) ",
                true);
  argparser.add({"--cls_modelpath"}, "onnx model path (classification model) ",
                true);
  argparser.add({"--cls_inputname"},
                "onnx model input name (classification model) ", true);
  argparser.add({"--cls_shape"}, "input shape (classification model) ", true);
  argparser.add({"--imagepath"}, "input image path", true);
  argparser.add({"--image_shape"}, "input image shape", true);
  argparser.add({"--loop"}, "number of running loop", true);
  argparser.enable_help();
  auto err = argparser.parse(argc, targv);
  if (err) {
    std::cout << err << std::endl;
    argparser.help();
    return -1;
  }
  if (argparser.exists("help")) {
    argparser.help();
    return 0;
  }

  auto loop = argparser.get<uint32_t>("loop");
  auto nvg = argparser.get<uint32_t>("vg");
  auto det_modelpath = argparser.get<std::string>("det_modelpath");
  auto det_inputname = argparser.get<std::string>("det_inputname");
  auto det_inputshape = argparser.get<std::string>("det_shape");
  auto cls_modelpath = argparser.get<std::string>("cls_modelpath");
  auto cls_inputname = argparser.get<std::string>("cls_inputname");
  auto cls_inputshape = argparser.get<std::string>("cls_shape");
  auto raw_imagepath = argparser.get<std::string>("imagepath");
  auto raw_imageshape = argparser.get<std::string>("image_shape");
  auto det_buffersize = argparser.get<uint32_t>("det_buffersize");
  auto cls_buffersize = argparser.get<uint32_t>("cls_buffersize");

  auto fcheck = [](const std::string &path,
                   const std::string &postfix = "") -> bool {
    if (!postfix.empty()) {
      if (path.substr(path.find_last_of('.') + 1) != postfix) {
        std::cerr << "\033[1;31m  file \"" << path << "\" does not have \"."
                  << postfix << "\" extension.\033[0m\n";
        return false;
      }
    }
    struct stat buffer;
    if (stat(path.c_str(), &buffer) != 0) {
      std::cerr << "\033[1;31m " << postfix << " file \"" << path
                << "\" does not exists.\033[0m\n";
      return false;
    }
    return true;
  };

  if (!fcheck(det_modelpath, "onnx") || !fcheck(cls_modelpath, "onnx") ||
      !fcheck(raw_imagepath, "data")) {
    return -1;
  }

  auto parseshape = [](const std::string &rawstr) -> std::array<int, 3> {
    std::regex ws_re(",");
    std::vector<std::string> shapestrs(
        std::sregex_token_iterator(rawstr.begin(), rawstr.end(), ws_re, -1),
        std::sregex_token_iterator());
    if (shapestrs.size() != 4) {
      std::cerr
          << "\033[1;31m \"" << rawstr
          << "\" format not valid, need 3 value, e.g. 1,3,224,224.\033[0m\n";
      abort();
    }
    std::array<int, 3> tshape;
    for (int i = 1; i < 4; i++) {
      tshape[i - 1] = std::stoi(shapestrs[i]);
    }

    return tshape;
  };

  auto rawshape = parseshape(raw_imageshape);
  auto det_shape = parseshape(det_inputshape);
  auto cls_shape = parseshape(cls_inputshape);

  Mat<uint8_t> raw(rawshape);
  {
    std::ifstream rawfile(raw_imagepath,
                          std::ios_base::in | std::ios_base::binary);
    if (rawfile.is_open()) {
      rawfile.read(reinterpret_cast<char *>(raw.ptr(0)), raw.size());
      rawfile.close();
    }
  }

  auto resizeraw =
      [](const Mat<uint8_t> &raw,
         const std::vector<std::function<float(uint8_t)>> &preprocess,
         const std::array<int, 3> &outshape) -> std::unique_ptr<Mat<float>> {
    assert(outshape[0] == raw.shape[0] && outshape[0] > 0 && outshape[1] > 0 &&
           outshape[2] > 0 && "resize output shape not valid");
    int stridein = raw.shape[1] * raw.shape[2];
    int strideout = outshape[2] * outshape[1];
    auto resizedmat = std::make_unique<Mat<float>>(outshape);
    for (int i = 0; i < outshape[0]; i++) {
      resize(raw.ptr(i * stridein), resizedmat->ptr(i * strideout),
             raw.shape[2], raw.shape[1], outshape[2], outshape[1],
             preprocess[i]);
    }
    return resizedmat;
  };

  auto crop = [](const Mat<uint8_t> &raw,
                 const Box &box) -> std::unique_ptr<Mat<uint8_t>> {
    int x0 = std::min(std::max(0, static_cast<int>(box[0])), raw.shape[2] - 1);
    int y0 = std::min(std::max(0, static_cast<int>(box[1])), raw.shape[1] - 1);
    int x1 = std::min(std::max(0, static_cast<int>(box[2])), raw.shape[2] - 1);
    int y1 = std::min(std::max(0, static_cast<int>(box[3])), raw.shape[1] - 1);
    assert(x1 - x0 > 0 && y1 - y0 > 0 && "box not valid");
    std::array<int, 3> outshape{raw.shape[0], y1 - y0, x1 - x0};
    auto cropedmat = std::make_unique<Mat<uint8_t>>(outshape);
    int stridein = raw.shape[1] * raw.shape[2];
    int strideout = outshape[1] * outshape[2];

    for (int i = 0; i < raw.shape[0]; i++) {
      const uint8_t *ip = raw.ptr(i * stridein);
      uint8_t *op = cropedmat->ptr(i * strideout);
      for (int h = y0; h < y1; h++)
        for (int w = x0; w < x1; w++) {
          op[outshape[2] * (h - y0) + (w - x0)] = ip[raw.shape[2] * h + w];
        }
    }
    return cropedmat;
  };

  auto det_preprocess = [](uint8_t a) -> float { return a / 255.0; };
  //[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
  auto cls_preprocess_t = [](float mean, float std, uint8_t a) -> float {
    return (a / 255.0 - mean) / std;
  };
  const std::vector<std::function<float(uint8_t)>> cls_preprocess_channels = {
      [cls_preprocess_t](auto &&PH1) {
        return cls_preprocess_t(0.485, 0.229, std::forward<decltype(PH1)>(PH1));
      },
      [cls_preprocess_t](auto &&PH1) {
        return cls_preprocess_t(0.456, 0.224, std::forward<decltype(PH1)>(PH1));
      },
      [cls_preprocess_t](auto &&PH1) {
        return cls_preprocess_t(0.406, 0.225, std::forward<decltype(PH1)>(PH1));
      },

  };
  auto det_input = resizeraw(
      raw, {det_preprocess, det_preprocess, det_preprocess}, det_shape);

  auto createngine = [](const std::string &modelpath,
                        const std::string &inputnames,
                        const std::string &inputshapes) {
    std::string execpath =
        modelpath.substr(0, modelpath.find_last_of('.')) + ".exec";
    TopsInference::IParser *parser =
        TopsInference::create_parser(TopsInference::TIF_ONNX);
    TopsInference::IOptimizer *optimizer = TopsInference::create_optimizer();
    parser->setInputNames(inputnames.c_str());
    parser->setInputShapes(inputshapes.c_str());
    TopsInference::INetwork *network = parser->readModel(modelpath.c_str());
    TopsInference::IOptimizerConfig *optimizer_config = optimizer->getConfig();
    optimizer_config->setBuildFlag(
        TopsInference::BuildFlag::TIF_KTYPE_MIX_FP16);
    auto *engine = optimizer->build(network);
    engine->saveExecutable(execpath.c_str());
    std::cout << "[INFO] save engine file: " << execpath << '\n';
    TopsInference::release_parser(parser);
    TopsInference::release_optimizer(optimizer);
    TopsInference::release_engine(engine);
  };

  auto loadengine = [](const std::string &execpath, const uint32_t cachesize)
      -> std::tuple<TopsInference::IEngine *,
                    std::vector<std::unique_ptr<std::vector<float>>>> {
    TopsInference::IEngine *engine = nullptr;
    engine = TopsInference::create_engine();
    engine->loadExecutable(execpath.c_str());
    std::cout << "[INFO] load engine file: " << execpath << '\n';

    std::vector<std::unique_ptr<std::vector<float>>> outputs;
    outputs.reserve(engine->getOutputNum());
    for (int i = 0; i < engine->getOutputNum(); i++) {
      auto outputs_shape_info = engine->getOutputShape(i);
      size_t all_size = cachesize;
      for (int j = 0; j < outputs_shape_info.nbDims; j++)
        all_size *= outputs_shape_info.dimension[j];
      outputs.emplace_back(std::make_unique<std::vector<float>>(all_size, 0));
    }

    return std::make_tuple(engine, std::move(outputs));
  };

  static std::mutex creating_model;
  static std::mutex printing;
  auto run = [loadengine, det_modelpath, det_inputname, det_inputshape,
              cls_modelpath, cls_inputname, cls_inputshape, det_shape,
              cls_shape, rawshape, resizeraw, &raw, cls_preprocess_channels,
              crop, createngine, det_buffersize, cls_buffersize,
              fcheck](const uint32_t vg, float *det_input, int times = 1) {
    {
      std::lock_guard<std::mutex> lock(printing);
      std::cout << "[INFO] used vg: " << vg << "\n";
    }
    std::vector<uint32_t> clusterids;
    for (uint32_t i = 0; i < vg; i++) {
      clusterids.push_back(i);
    }

    auto *handle =
        TopsInference::set_device(0, clusterids.data(), clusterids.size());
    assert(handle != nullptr && "[ERROR] set device failed!");
    std::string det_execpath =
        det_modelpath.substr(0, det_modelpath.find_last_of('.')) + ".exec";
    std::string cls_execpath =
        cls_modelpath.substr(0, cls_modelpath.find_last_of('.')) + ".exec";
    {

      std::lock_guard<std::mutex> lock(creating_model);
      if (!fcheck(det_execpath)) {
        createngine(det_modelpath, det_inputname, det_inputshape);
      }
    }
    {

      std::lock_guard<std::mutex> lock(creating_model);
      if (!fcheck(cls_execpath)) {
        createngine(cls_modelpath, cls_inputname, cls_inputshape);
      }
    }

    auto det_engine_and_outputs = loadengine(det_execpath, det_buffersize);
    auto cls_engine_and_outputs = loadengine(cls_execpath, cls_buffersize);

    auto *det_engine = std::get<0>(det_engine_and_outputs);

    {
      assert(det_engine->getOutputNum() > 0 && "number of outputs not valid!");
      auto parsedshape = det_engine->getOutputShape(0);
      assert(parsedshape.nbDims == 3 && "output dim not valid!");
      k_proposal_count = parsedshape.dimension[1];
      k_step = parsedshape.dimension[2];
    }

    auto *cls_engine = std::get<0>(cls_engine_and_outputs);

    {
      assert(cls_engine->getOutputNum() > 0 && "number of outputs not valid!");
      auto parsedshape = cls_engine->getOutputShape(0);
      assert(parsedshape.nbDims == 2 && "output dim not valid!");
      k_kn_classes = parsedshape.dimension[1];
    }

    const auto &det_outputsraw = std::get<1>(det_engine_and_outputs);
    const auto &cls_outputsraw = std::get<1>(cls_engine_and_outputs);

    std::vector<void *> det_outputs;
    det_outputs.reserve((det_outputsraw.size()));
    for (const auto &p : det_outputsraw) {
      det_outputs.emplace_back(p->data());
    }

    std::vector<void *> cls_outputs;
    cls_outputs.reserve((cls_outputsraw.size()));
    for (const auto &p : cls_outputsraw) {
      cls_outputs.emplace_back(p->data());
    }

    uint64_t det_input_singlesize = det_shape[0] * det_shape[1] * det_shape[2];
    uint64_t cls_input_singlesize = cls_shape[0] * cls_shape[1] * cls_shape[2];
    uint64_t det_output_singlesize = k_proposal_count * k_step;
    std::vector<float> det_inputsdata(det_input_singlesize * det_buffersize, 0);
    std::vector<float> cls_inputsdata(cls_input_singlesize * cls_buffersize, 0);

    // warmup
    {

      {
        void *inputs[] = {det_inputsdata.data()};

        det_engine->run_with_batch(
            det_buffersize, inputs, det_outputs.data(),
            TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST);
      }
      {
        void *inputs[] = {cls_inputsdata.data()};
        cls_engine->run_with_batch(
            cls_buffersize, inputs, cls_outputs.data(),
            TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST);
      }
    }

    std::vector<float *> inputs_rawdata(times, det_input);

    uint32_t imgidx = 0;
    auto t0 = std::time(nullptr);
    batch(inputs_rawdata.begin(), inputs_rawdata.end(), det_buffersize,
          [&](decltype(inputs_rawdata)::iterator s,
              decltype(inputs_rawdata)::iterator e) {
            auto size = std::distance(s, e);
            {
              assert(size <= det_buffersize && "size not valid");

              uint64_t offset = 0;
              std::for_each(
                  s, e,
                  [&det_inputsdata, &offset, &det_input_singlesize](auto p) {
                    memcpy(det_inputsdata.data() + offset, p,
                           det_input_singlesize * sizeof(float));
                    offset += det_input_singlesize;
                  });

              std::vector<void *> inputs = {det_inputsdata.data()};

              det_engine->run_with_batch(
                  size, inputs.data(), det_outputs.data(),
                  TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST);
            }

            assert(det_outputsraw[0]->size() >= size * det_output_singlesize &&
                   "size not valid");

            using ClsData = struct ClsData {
              ClsData(std::unique_ptr<Mat<float>> &&a, Box b, uint32_t c)
                  : data(std::move(a)), bb(b), imgidx(c) {}
              std::unique_ptr<Mat<float>> data;
              Box bb;
              uint32_t imgidx;
            };
            std::vector<ClsData> clsdatas;
            for (uint64_t i = 0, offset = 0; i < size;
                 i++, offset += det_output_singlesize) {
              auto det_output = yolov5_postprocess(
                  det_outputsraw[0]->data() + offset, rawshape, det_shape);
              for (const auto &out : *det_output) {
                clsdatas.emplace_back(resizeraw(*crop(raw, out),
                                                cls_preprocess_channels,
                                                cls_shape),
                                      out, static_cast<uint32_t>(imgidx + i));
              }
            }

            assert(clsdatas.size() <= cls_buffersize && "size not valid");

            {
              uint64_t offset = 0;
              for (const auto &clsdata : clsdatas) {
                memcpy(cls_inputsdata.data() + offset, clsdata.data->ptr(0),
                       cls_input_singlesize * sizeof(float));
                offset += cls_input_singlesize;
              }
              std::vector<void *> inputs = {cls_inputsdata.data()};

              cls_engine->run_with_batch(
                  clsdatas.size(), inputs.data(), cls_outputs.data(),
                  TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST);

              offset = 0;
              for (const auto &r : clsdatas) {
                auto *p = static_cast<float *>(cls_outputs[0]) + offset;
                int cls_maxidx = -1;
                float cls_maxvalue = 0.F;
                for (int i = 0; i < 1000; i++) {
                  if (p[i] > cls_maxvalue) {
                    cls_maxvalue = p[i];
                    cls_maxidx = i;
                  }
                }

                std::cout << ">>>>>>>>>img: " << r.imgidx << "   box : ["
                          << r.bb[0] << ", " << r.bb[1] << ", " << r.bb[2]
                          << ", " << r.bb[3] << ", " << r.bb[4] << ", "
                          << r.bb[5] << "], " << cls_maxidx << "\n";
                offset += k_kn_classes;
              }
            }

            imgidx += size;
          });
    auto t1 = std::time(nullptr);
    std::cout << "[INFO] running time: " << t1 - t0 << " seconds\n";

    TopsInference::release_engine(det_engine);
    TopsInference::release_engine(cls_engine);
    TopsInference::release_device(handle);
  };

  TopsInference::topsInference_init();

  run(nvg, det_input->ptr(0), loop);

  std::cout << "DONE\n";
  TopsInference::topsInference_finish();
  return 0;
}