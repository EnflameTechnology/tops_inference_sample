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

#include <TopsInference/TopsInferRuntime.h>
#ifdef PROFILE
#include <easy/details/profiler_colors.h>
#include <easy/profiler.h>
#endif
#include "arg.hpp"
#include "chan.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <tuple>
#include <utility>
#include <vector>
#ifndef PROFILE
#define EASY_PROFILER_ENABLE
#define EASY_BLOCK(V)
#define EASY_END_BLOCK
#endif
template <typename T> struct Mat {
  Mat(int ch, int w, int h) {
    shape = {ch, h, w};
    data.resize(size());
  }
  explicit Mat(const std::array<int, 3> &sp) {
    shape = sp;
    data.resize(size());
  }
  size_t size() { return shape[0] * shape[1] * shape[2]; }
  T &operator[](size_t idx) { return data[idx]; }
  const T *ptr(size_t idx) const { return data.data() + idx; }
  T *ptr(size_t idx) { return data.data() + idx; }
  std::vector<T> data;
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
int batch(Iterator begin, Iterator end, Distance k, Func f) {
  Iterator batchbegin;
  Iterator batchend;
  batchend = batchbegin = begin;
  int nc = 0;
  do {
    if (std::distance(batchend, end) < k)
      batchend = end;
    else
      std::advance(batchend, k);
    f(batchbegin, batchend, nc++);
    batchbegin = batchend;
  } while (std::distance(batchbegin, end) > 0);
  return nc;
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
static int k_step = 0;
static int k_proposal_count = 0;
static int k_kn_classes = 0;
static std::unique_ptr<Boxes>
yolov5_postprocess(const float *p, const std::array<int, 3> &rawshape,
                   const std::array<int, 3> &inputshape) {
  auto boxes = std::make_unique<Boxes>();

  const float score_threshold = 0.5;
  const float nms_threshold = 0.45;
  const float scalew = inputshape[2] * 1.0 / rawshape[2];
  const float scaleh = inputshape[1] * 1.0 / rawshape[1];
  std::map<int, Boxes> cls_of_boxes;
  for (size_t i = 0; i < k_proposal_count; i++) {
    float obj_score = p[4];
    if (obj_score < score_threshold) {
      p += k_step;
      continue;
    }
    int c_id = -1;
    float c_score = 0;
    for (size_t j = 5; j < k_step; j++) {
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
    p += k_step;
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
  EASY_PROFILER_ENABLE;
  arg::ArgParser argparser(
      "sampleParallelPipeline",
      "This program is a demonstration about dtu parallel pipeline execution.");
  argparser.add({"--det_vg"}, "number of VG (detection model) ", true);
  argparser.add({"--det_buffersize"}, "buffer size (detection model) ", true);
  argparser.add({"--det_nstream"}, "number of streams (detection model) ",
                true);
  argparser.add({"--det_modelpath"}, "onnx model path (detection model) ",
                true);
  argparser.add({"--det_inputname"}, "onnx model input name (detection model) ",
                true);
  argparser.add({"--det_shape"}, "input shape (detection model) ", true);
  argparser.add({"--cls_vg"}, "number of VG (classification model) ", true);
  argparser.add({"--cls_buffersize"}, "buffer size (classification model) ",
                true);
  argparser.add({"--cls_nstream"}, "number of streams (classification model) ",
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
  auto det_vg = argparser.get<uint32_t>("det_vg");
  auto det_buffersize = argparser.get<uint32_t>("det_buffersize");
  auto det_nstream = argparser.get<uint32_t>("det_nstream");
  auto cls_vg = argparser.get<uint32_t>("cls_vg");
  auto cls_buffersize = argparser.get<uint32_t>("cls_buffersize");
  auto cls_nstream = argparser.get<uint32_t>("cls_nstream");

  std::array<std::string, 5> modelpathes_and_names = {
      argparser.get<std::string>("det_modelpath"),
      argparser.get<std::string>("cls_modelpath"),
      argparser.get<std::string>("imagepath"),
      argparser.get<std::string>("det_inputname"),
      argparser.get<std::string>("cls_inputname")};

  const auto &det_modelpath = modelpathes_and_names[0];
  const auto &cls_modelpath = modelpathes_and_names[1];
  const auto &raw_imagepath = modelpathes_and_names[2];
  const auto &det_inputnames = modelpathes_and_names[3];
  const auto &cls_inputnames = modelpathes_and_names[4];

  std::array<std::string, 3> strshapes = {
      argparser.get<std::string>("image_shape"),
      argparser.get<std::string>("det_shape"),
      argparser.get<std::string>("cls_shape")};
  const auto &det_inputshapes = strshapes[1];
  const auto &cls_inputshapes = strshapes[2];

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

  if (!std::all_of(modelpathes_and_names.begin(),
                   modelpathes_and_names.begin() + 2,
                   [&fcheck](const std::string &p) -> bool {
                     return fcheck(p, "onnx");
                   }) &&
      !fcheck(*(modelpathes_and_names.begin() + 3), "data")) {
    return -1;
  }

  std::array<std::array<int, 3>, 3> shapes;
  std::transform(
      strshapes.begin(), strshapes.end(), shapes.begin(),
      [](const std::string &rawstr) -> std::array<int, 3> {
        std::regex ws_re(",");
        std::vector<std::string> shapestrs(
            std::sregex_token_iterator(rawstr.begin(), rawstr.end(), ws_re, -1),
            std::sregex_token_iterator());
        if (shapestrs.size() != 4) {
          std::cerr << "\033[1;31m \"" << rawstr
                    << "\" format not valid, need 3 value, e.g. "
                       "1,3,224,224.\033[0m\n";
          abort();
        }
        std::array<int, 3> tshape;
        for (int i = 1; i < 4; i++) {
          tshape[i - 1] = std::stoi(shapestrs[i]);
        }

        return tshape;
      });

  const auto &rawshape = shapes[0];
  const auto &det_shape = shapes[1];
  const auto &cls_shape = shapes[2];
  auto readimg =
      [](const std::string &p,
         const std::array<int, 3> &shape) -> std::unique_ptr<Mat<uint8_t>> {
    auto raw = std::make_unique<Mat<uint8_t>>(shape);
    std::ifstream rawfile(p, std::ios_base::in | std::ios_base::binary);
    if (rawfile.is_open()) {
      rawfile.read(reinterpret_cast<char *>(raw->ptr(0)), raw->size());
      rawfile.close();
    }
    return raw;
  };

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

  auto det_inputpreprocess = [](uint8_t a) -> float { return a / 255.0; };
  //[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
  auto cls_inputpreprocess_t = [](float mean, float std, uint8_t a) -> float {
    return (a / 255.0 - mean) / std;
  };
  const std::vector<std::function<float(uint8_t)>>
      cls_inputpreprocess_channels = {
          [cls_inputpreprocess_t](auto &&PH1) {
            return cls_inputpreprocess_t(0.485, 0.229,
                                         std::forward<decltype(PH1)>(PH1));
          },
          [cls_inputpreprocess_t](auto &&PH1) {
            return cls_inputpreprocess_t(0.456, 0.224,
                                         std::forward<decltype(PH1)>(PH1));
          },
          [cls_inputpreprocess_t](auto &&PH1) {
            return cls_inputpreprocess_t(0.406, 0.225,
                                         std::forward<decltype(PH1)>(PH1));
          },

      };

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
  auto loadengine = [](const std::string &execpath, const uint32_t batchsize,
                       const uint32_t nstream)
      -> std::tuple<
          TopsInference::IEngine *,
          std::vector<std::vector<std::unique_ptr<std::vector<float>>>>> {
    TopsInference::IEngine *engine = nullptr;
    engine = TopsInference::create_engine();
    engine->loadExecutable(execpath.c_str());
    std::cout << "[INFO] load engine file: " << execpath << '\n';

    std::vector<std::vector<std::unique_ptr<std::vector<float>>>>
        stream_putputs;
    stream_putputs.resize(nstream);
    for (auto &outputs : stream_putputs) {
      outputs.reserve(engine->getOutputNum());
      for (int i = 0; i < engine->getOutputNum(); i++) {
        auto outputs_shape_info = engine->getOutputShape(i);
        size_t all_size = batchsize;
        for (int j = 0; j < outputs_shape_info.nbDims; j++)
          all_size *= outputs_shape_info.dimension[j];
        outputs.emplace_back(std::make_unique<std::vector<float>>(all_size, 0));
      }
    }

    return std::make_tuple(engine, std::move(stream_putputs));
  };

  Chan<std::pair<std::vector<std::unique_ptr<Mat<float>>>,
                 std::vector<std::unique_ptr<Mat<uint8_t>>>>>
      det_inputchannel;

  auto det_preprocess = [&readimg, &det_inputpreprocess, &resizeraw,
                         &det_inputchannel, &raw_imagepath, &det_buffersize,
                         &det_nstream, &rawshape, &det_shape, loop]() {
    const size_t n_emulate_img = loop;
    std::vector<std::string> imgpathes(n_emulate_img, raw_imagepath);

    batch(
        imgpathes.begin(), imgpathes.end(), det_buffersize * det_nstream,
        [&readimg, &resizeraw, &det_inputpreprocess, &rawshape, &det_shape,
         &det_inputchannel](std::vector<std::string>::iterator b,
                            std::vector<std::string>::iterator e,
                            __attribute__((unused)) int nc) {
          decltype(det_inputchannel)::type out;
          for (auto p = b; p != e; p++) {
            EASY_BLOCK("readimg");
            out.second.push_back(readimg(*p, rawshape));
            EASY_END_BLOCK;
            EASY_BLOCK("resizeraw detinput resizeraw");
            out.first.push_back(resizeraw(
                *out.second.back(),
                {det_inputpreprocess, det_inputpreprocess, det_inputpreprocess},
                det_shape));
            EASY_END_BLOCK;
          }

          det_inputchannel.send(std::move(out));
        });
    det_inputchannel.close();
  };

  static std::mutex printing;
  static std::mutex creating_model;

  Chan<std::pair<std::vector<std::unique_ptr<Boxes>>,
                 std::vector<std::unique_ptr<Mat<uint8_t>>>>>
      cls_preprocesschannel;

  auto det_runmodel = [&fcheck, &createngine, &loadengine, &det_inputchannel,
                       &cls_preprocesschannel, &det_buffersize, &det_nstream,
                       &det_inputnames, &det_modelpath, &det_shape, &rawshape,
                       &det_inputshapes, &det_vg]() {
    {
      std::lock_guard<std::mutex> lock(printing);
      std::cout << "[INFO] used vg: " << det_vg << "\n";
    }
    std::vector<uint32_t> clusterids;
    for (uint32_t i = 0; i < det_vg; i++) {
      clusterids.push_back(i);
    }
    auto *handle =
        TopsInference::set_device(0, clusterids.data(), clusterids.size());
    assert(handle != nullptr && "[ERROR] set device failed!\n");
    std::string execpath =
        det_modelpath.substr(0, det_modelpath.find_last_of('.')) + ".exec";
    {
      std::lock_guard<std::mutex> lock(creating_model);
      if (!fcheck(execpath)) {
        createngine(det_modelpath, det_inputnames, det_inputshapes);
      }
    }

    const auto &loaded = loadengine(execpath, det_buffersize, det_nstream);
    const auto &engine = std::get<0>(loaded);
    const auto &output_hostdata = std::get<1>(loaded);

    assert(engine->getOutputNum() > 0 && "number of outputs not valid!");
    auto parsedshape = engine->getOutputShape(0);
    assert(parsedshape.nbDims == 3 && "output dim not valid!");
    k_proposal_count = parsedshape.dimension[1];
    k_step = parsedshape.dimension[2];
    std::vector<TopsInference::topsInferStream_t> streams(det_nstream);
    for (int i = 0; i < det_nstream; i++) {
      TopsInference::topsInferStream_t s;
      if (!TopsInference::create_stream(&s)) {
        std::cerr << "\033[1;31m[ERROR] TopsInference::create_stream "
                     "failed!\033[0m\n";
        abort();
      }
      streams.emplace(streams.begin() + i, s);
    }

    std::vector<std::vector<float>> input_hostdata(det_nstream);

    for (auto &e : input_hostdata) {
      e.resize(det_shape[0] * det_shape[1] * det_shape[2] * det_buffersize);
    }

    // warmup
    {
      void *inputs[] = {input_hostdata[0].data()};
      std::vector<void *> outputs;
      outputs.reserve((output_hostdata[0].size()));
      for (const auto &p : output_hostdata[0]) {
        outputs.emplace_back(p->data());
      }
      engine->run_with_batch(
          det_buffersize, inputs, outputs.data(),
          TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST);
    }

    while (!det_inputchannel.closed()) {
      EASY_BLOCK("det_inputchannel receive data");
      auto receiveddata = det_inputchannel.receive();
      auto inputdata = std::move(receiveddata.first);
      auto rawinputdata = std::move(receiveddata.second);
      EASY_END_BLOCK;
      {
        std::vector<std::vector<uint32_t>> real_batchsizes;
        uint32_t idx = 0;
        if (inputdata.size() < det_nstream) {
          // run only in stream-0
          real_batchsizes.resize(1);
          real_batchsizes[0].reserve(inputdata.size());
          uint64_t offset = 0;
          for (const auto &p : inputdata) {
            memcpy(static_cast<float *>(input_hostdata[0].data()) + offset,
                   p->ptr(0), p->size() * sizeof(float));
            offset += p->size();
            real_batchsizes[0].emplace_back(idx++);
          }
          void *inputs[] = {input_hostdata[0].data()};

          std::vector<void *> outputs;
          outputs.reserve((output_hostdata[0].size()));
          for (const auto &p : output_hostdata[0]) {
            outputs.emplace_back(p->data());
          }
          engine->run_with_batch(
              inputdata.size(), inputs, outputs.data(),
              TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST);
        } else {

          auto stride = static_cast<int>(
              std::ceil(1.0F * inputdata.size() / det_nstream));
          auto used_nstream = batch(
              inputdata.begin(), inputdata.end(), stride,
              [&streams, &engine, &output_hostdata, &input_hostdata, &idx,
               &real_batchsizes,
               &det_buffersize](decltype(inputdata)::iterator b,
                                decltype(inputdata)::iterator e, int nc) {
                std::vector<uint32_t> binfo;
                assert(nc < streams.size() &&
                       "\033[1;31m[ERROR] streams index not valid!\033[0m\n");
                int offset = 0;
                uint32_t sample_size = 0;
                EASY_BLOCK("det copy input data");
                for (auto p = b; p != e; p++) {
                  memcpy(static_cast<float *>(input_hostdata[nc].data()) +
                             offset,
                         (*p)->ptr(0), (*p)->size() * sizeof(float));
                  offset += (*p)->size();
                  sample_size += 1;
                  binfo.push_back(idx);
                  idx += 1;
                }
                EASY_END_BLOCK;

                real_batchsizes.push_back(binfo);
                void *inputs[] = {input_hostdata[nc].data()};

                std::vector<void *> outputs;
                outputs.reserve((output_hostdata[nc].size()));
                for (const auto &p : output_hostdata[nc]) {
                  outputs.emplace_back(p->data());
                }

                EASY_BLOCK("det run_with_batch stream");
                assert((sample_size <= det_buffersize) &&
                       "sample size note valid");
                engine->run_with_batch(
                    sample_size, inputs, outputs.data(),
                    TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST,
                    streams[nc]);
                EASY_END_BLOCK;
              });

          assert(used_nstream == det_nstream &&
                 "\033[1;31m[ERROR] used_nstream number not valid!\033[0m\n");

          // sync the streams
          EASY_BLOCK("sync det streams");
          for (int i = 0; i < det_nstream; i++) {
            TopsInference::synchronize_stream(streams[i]);
          }
          EASY_END_BLOCK;
        }
        // post process
        EASY_BLOCK("det post process");
        std::vector<std::unique_ptr<Boxes>> detresults;
        detresults.reserve(inputdata.size());
        assert(output_hostdata.size() == real_batchsizes.size() &&
               "size not valid!");
        for (int i = 0; i < output_hostdata.size(); i++) {
          const auto &outputs = output_hostdata[i];
          const auto &binfo = real_batchsizes[i];

          assert(k_proposal_count * k_step * det_buffersize >=
                     outputs[0]->size() &&
                 "size not valid!");
          auto *ptr = outputs[0]->data();
          uint64_t offset = 0;
          for (int i = 0; i < binfo.size(); i++) {
            auto r = yolov5_postprocess(ptr + offset, rawshape, det_shape);
            if (!r->empty()) {
              detresults.emplace_back(std::move(r));
            }
            offset += k_proposal_count * k_step;
          }
        }

        decltype(cls_preprocesschannel)::type out;
        assert(detresults.size() == rawinputdata.size() &&
               "[ERROR] raws and boxes size not equal!\n");
        out.first = std::move(detresults);
        out.second = std::move(rawinputdata);
        EASY_END_BLOCK;

        EASY_BLOCK("det sending");
        cls_preprocesschannel.send(std::move(out));
        EASY_END_BLOCK;
      }
    }

    // close channel
    cls_preprocesschannel.close();

    for (int i = 0; i < det_nstream; i++) {
      TopsInference::destroy_stream(streams[i]);
    }

    TopsInference::release_engine(engine);
    TopsInference::release_device(handle);
  };
  using ClsData = struct {
    std::unique_ptr<Mat<float>> data;
    Box bb;
    uint32_t imgidx;
  };
  Chan<std::vector<ClsData>> cls_runchannel;
  auto cls_preprocess = [&crop, &resizeraw, &cls_preprocesschannel,
                         &cls_inputpreprocess_channels, &cls_runchannel,
                         &cls_shape]() {
    uint32_t globalidx = 0;
    while (!cls_preprocesschannel.closed()) {
      auto received = cls_preprocesschannel.receive();
      auto boxes = std::move(received.first);
      auto raws = std::move(received.second);

      decltype(cls_runchannel)::type clsinputs;
      assert(raws.size() == boxes.size() &&
             "[ERROR] raws and boxes size not equal!\n");
      for (uint32_t i = 0; i < raws.size(); i++) {
        EASY_BLOCK("cls input image resize etc.");
        for (const auto &box : *boxes[i]) {
          auto cls_img = resizeraw(*crop(*(raws[i]), box),
                                   cls_inputpreprocess_channels, cls_shape);
          clsinputs.push_back(ClsData{std::move(cls_img), box, globalidx});
        }
        globalidx += 1;
      }
      cls_runchannel.send(std::move(clsinputs));
      EASY_END_BLOCK;
    }
    cls_runchannel.close();
  };

  using ClsResult = struct ClsResult {
    ClsResult(const Box &a, int b, uint32_t c) : bb(a), cls(b), imgidx(c) {}
    Box bb;
    int cls;
    uint32_t imgidx;
  };
  Chan<std::vector<ClsResult>> done_channel(1024);
  auto cls_runmodel = [&fcheck, &createngine, &loadengine, &cls_runchannel,
                       &done_channel, &det_vg, &cls_vg, &cls_modelpath,
                       &cls_inputnames, &cls_inputshapes, &cls_shape,
                       &cls_buffersize, &cls_nstream]() {
    {
      std::lock_guard<std::mutex> lock(printing);
      std::cout << "[INFO] used vg: " << cls_vg << "\n";
    }
    std::vector<uint32_t> clusterids;
    for (uint32_t i = det_vg; i < det_vg + cls_vg; i++) {
      clusterids.push_back(i);
    }
    auto *handle =
        TopsInference::set_device(0, clusterids.data(), clusterids.size());
    assert(handle != nullptr && "[ERROR] set device failed!\n");
    std::string execpath =
        cls_modelpath.substr(0, cls_modelpath.find_last_of('.')) + ".exec";
    {
      std::lock_guard<std::mutex> lock(creating_model);
      if (!fcheck(execpath)) {
        createngine(cls_modelpath, cls_inputnames, cls_inputshapes);
      }
    }
    uint32_t datasize = cls_buffersize;
    const auto &loaded = loadengine(execpath, datasize, cls_nstream);
    const auto &engine = std::get<0>(loaded);
    const auto &output_hostdata = std::get<1>(loaded);

    assert(engine->getOutputNum() > 0 && "number of outputs not valid!");
    auto parsedshape = engine->getOutputShape(0);
    assert(parsedshape.nbDims == 2 && "output dim not valid!");
    k_kn_classes = parsedshape.dimension[1];

    std::vector<std::vector<float>> input_hostdata(cls_nstream);
    for (auto &e : input_hostdata) {
      e.resize(cls_shape[0] * cls_shape[1] * cls_shape[2] * cls_buffersize);
    }

    std::vector<TopsInference::topsInferStream_t> streams(cls_nstream);
    for (int i = 0; i < cls_nstream; i++) {
      TopsInference::topsInferStream_t s;
      if (!TopsInference::create_stream(&s)) {
        std::cerr << "\033[1;31m[ERROR] TopsInference::create_stream "
                     "failed!\033[0m\n";
        abort();
      }
      streams.emplace(streams.begin() + i, s);
    }

    // warmup
    {
      void *inputs[] = {input_hostdata[0].data()};
      std::vector<void *> outputs;
      outputs.reserve((output_hostdata[0].size()));
      for (const auto &p : output_hostdata[0]) {
        outputs.emplace_back(p->data());
      }
      engine->run_with_batch(
          cls_buffersize, inputs, outputs.data(),
          TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST);
    }

    while (!cls_runchannel.closed()) {
      EASY_BLOCK("cls_runchannel data receive");
      auto inputelements = cls_runchannel.receive();
      EASY_END_BLOCK;
      {
        std::vector<std::vector<ClsResult>> real_batchsizes;

        if (inputelements.size() < cls_nstream) {
          real_batchsizes.resize(1);
          real_batchsizes[0].reserve(inputelements.size());
          // run only in stream-0
          uint64_t offset = 0;

          for (const auto &inputhost : inputelements) {
            memcpy(input_hostdata[0].data() + offset, inputhost.data->ptr(0),
                   inputhost.data->size() * sizeof(float));
            offset += inputhost.data->size();
            real_batchsizes[0].emplace_back(inputhost.bb, 0, inputhost.imgidx);
          }
          void *inputs[] = {input_hostdata[0].data()};
          std::vector<void *> outputs;
          outputs.reserve((output_hostdata[0].size()));
          for (const auto &p : output_hostdata[0]) {
            outputs.emplace_back(p->data());
          }
          engine->run_with_batch(
              inputelements.size(), inputs, outputs.data(),
              TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST);

        } else {
          auto stride = static_cast<int>(
              std::ceil(1.0F * inputelements.size() / cls_nstream));

          auto used_nstream = batch(
              inputelements.begin(), inputelements.end(), stride,
              [&streams, &engine, &output_hostdata, &input_hostdata,
               &real_batchsizes,
               &cls_buffersize](decltype(inputelements)::iterator b,
                                decltype(inputelements)::iterator e, int nc) {
                assert(nc < streams.size() &&
                       "\033[1;31m[ERROR] streams index not valid!\033[0m\n");
                int offset = 0;
                int sample_size = 0;
                std::vector<ClsResult> rs;
                rs.reserve(std::distance(b, e));
                EASY_BLOCK("cls memcpy input data");
                for (auto p = b; p != e; p++) {
                  memcpy(static_cast<float *>(input_hostdata[nc].data()) +
                             offset,
                         (*p).data->ptr(0), (*p).data->size() * sizeof(float));
                  offset += (*p).data->size();
                  sample_size += 1;
                  rs.emplace_back(std::move(p->bb), 0, p->imgidx);
                }
                EASY_END_BLOCK;
                real_batchsizes.push_back(rs);

                void *inputs[] = {input_hostdata[nc].data()};

                std::vector<void *> outputs;
                outputs.reserve((output_hostdata[nc].size()));
                for (const auto &p : output_hostdata[nc]) {
                  outputs.emplace_back(p->data());
                }

                EASY_BLOCK("cls run_with_batch stream");
                assert((sample_size <= cls_buffersize) &&
                       "sample size note valid");
                engine->run_with_batch(
                    sample_size, inputs, outputs.data(),
                    TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST,
                    streams[nc]);
                EASY_END_BLOCK;
              });
          assert(used_nstream == cls_nstream &&
                 "[ERROR] used_nstream number not valid!\n");

          // sync the streams
          EASY_BLOCK("cls sync the streams");
          for (int i = 0; i < cls_nstream; i++) {
            TopsInference::synchronize_stream(streams[i]);
          }
          EASY_END_BLOCK;
        }
        // post process
        EASY_BLOCK("cls post process");
        size_t ninstance = 0;
        for (const auto &l : real_batchsizes) {
          ninstance += l.size();
        }
        decltype(done_channel)::type clsresults;
        clsresults.reserve(ninstance);
        for (int i = 0; i < real_batchsizes.size(); i++) {
          auto *p = output_hostdata[i][0]->data();
          assert(output_hostdata[i][0]->size() >
                     real_batchsizes[i].size() * k_kn_classes &&
                 "[ERROR] size not valid!\n");
          uint64_t offset = 0;
          for (const auto &rs : real_batchsizes[i]) {
            int cls_maxidx = -1;
            float cls_maxvalue = 0.F;
            for (int j = 0; j < k_kn_classes; j++) {
              auto v = *(p + offset + j);
              if (v > cls_maxvalue) {
                cls_maxvalue = v;
                cls_maxidx = j;
              }
            }
            clsresults.emplace_back(std::move(rs.bb), cls_maxidx, rs.imgidx);
            offset += k_kn_classes;
          }
        }
        EASY_END_BLOCK;
        EASY_BLOCK("cls sending");
        done_channel.send(std::move(clsresults));
        EASY_END_BLOCK
      }
    }
    done_channel.close();

    for (int i = 0; i < cls_nstream; i++) {
      TopsInference::destroy_stream(streams[i]);
    }

    TopsInference::release_engine(engine);
    TopsInference::release_device(handle);
  };

  auto report = [&done_channel]() {
    bool start = false;
    time_t t0 = 0;
    while (!done_channel.closed()) {
      if (!start) {
        t0 = std::time(nullptr);
        start = true;
      }
      auto result = done_channel.receive();
      for (const auto &r : result) {
        std::cout << ">>>>>>>>>img: " << r.imgidx << "   box : [" << r.bb[0]
                  << ", " << r.bb[1] << ", " << r.bb[2] << ", " << r.bb[3]
                  << ", " << r.bb[4] << ", " << r.bb[5] << "], " << r.cls
                  << "\n";
      }
    }
    auto t1 = std::time(nullptr);
    std::cout << "[INFO] running time: " << t1 - t0 << " seconds\n";
    return;
  };

  TopsInference::topsInference_init();

  // let each threads begin to run
  auto future_det_preprocess = std::async(std::launch::async, det_preprocess);
  auto future_det_runmodel = std::async(std::launch::async, det_runmodel);
  auto future_cls_preprocess = std::async(std::launch::async, cls_preprocess);
  auto future_cls_runmodel = std::async(std::launch::async, cls_runmodel);
  auto future_report = std::async(std::launch::async, report);

  future_report.wait();

  TopsInference::topsInference_finish();
#ifdef PROFILE
  profiler::dumpBlocksToFile("result.prof");
#endif
  std::cout << "done\n";
  return 0;
}