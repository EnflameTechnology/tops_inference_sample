
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

#ifndef __SAMPLE_BERT_H__
#define __SAMPLE_BERT_H__

#include <string>
#include <vector>

#include "TopsInference/TopsInferRuntime.h"

#define ENFLAME_ASSERT(_expr, _fmt, ...)                                       \
  do {                                                                         \
    if (!_expr) {                                                              \
      fprintf(stderr, "[ENFLAME ASSERT : %s (%d)] :\n", __FILE__, __LINE__);   \
      fprintf(stderr, _fmt, ##__VA_ARGS__);                                    \
      fprintf(stderr, "\n");                                                   \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define ENFLAME_LOG_WARN(_fmt, ...)                                            \
  do {                                                                         \
    fprintf(stdout, "[ENFLAME WARN : %s (%d)] :\n", __FILE__, __LINE__);       \
    fprintf(stdout, _fmt, ##__VA_ARGS__);                                      \
    fprintf(stdout, "\n");                                                     \
  } while (0)

#define ENFLAME_CHECK(_expr, error_mgr)                                        \
  do {                                                                         \
    if (!_expr) {                                                              \
      int error_count = error_mgr->getErrorCount();                            \
      fprintf(stderr, "ENFLAME ERROR : %s (%d) : \n", __FILE__, __LINE__);     \
      for (int i = 0; i < error_count; i++) {                                  \
        std::string error_msg = error_mgr->getErrorMsg(i);                     \
        fprintf(stderr, "%s\n", error_msg.c_str());                            \
      }                                                                        \
      error_mgr->clear();                                                      \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

namespace sample_bert {

enum class TESTMODE {
  NORMAL = 0,
  PERFORMANCE = 1,
};

struct BertInferenceParam {
  std::string model_file;               //< model file
  std::string input_file;               //< input file
  std::string output_file;              //< output file
  std::string vocab_file;               //< vocab file
  std::string engine_file;              //< engine file to save
  std::string precision{"mix"};         //< precision when build engine
  int card_id{0};                       //< device id
  int cluster_id{0};                    //< cluster id
  int batch_size{1};                    //< batch size to build engine
  int max_seq_len{256};                 //< length of max sequence
  int max_query_len{64};                //< length of max query
  int doc_stride{128};                  //< doc stride
  TESTMODE test_mode{TESTMODE::NORMAL}; //< test mode
};

/**
 * @brief global init topsinference
 *
 */
void global_init_tops();

/**
 * @brief global release topsinference
 *
 */
void global_release_tops();

/**
 * @brief class to do Bert inference
 *
 */
class SampleBert {
public:
  explicit SampleBert(BertInferenceParam &param);
  ~SampleBert();

  /**
   * @brief do sample
   * 1. do topsinference init [maybe]
   * 2. load model
   * 3. generate engine(save engine maybe)
   * 4. forward
   * 5. wait
   * 6. write output [maybe]
   */
  void exec();

protected:
  void normal_test(int cluster_id = 0);
  void performance_test();

protected:
  void init(const std::vector<uint32_t>& cluster_ids);
  void release();
  /**
   * @brief init TopsInference, read model and build engine
   *
   */
  void pre_process();
  void forward();

private:
  void post_process(const std::vector<float> &start_logits,
                    const std::vector<float> &end_logits);

private:
  //! information from param
  std::string m_model_file;
  std::string m_input_file;
  std::string m_output_file;
  std::string m_vocab_file;
  std::string m_engine_file;
  TopsInference::BuildFlag m_precision{
      TopsInference::BuildFlag::TIF_KTYPE_DEFAULT};
  int m_card_id{0};
  int m_cluster_id{0};
  int m_batch_size{1};
  int m_max_seq_len{256};
  int m_doc_stride{128};
  int m_max_query_len{64};
  TESTMODE m_test_mode{TESTMODE::NORMAL};

  //! information from model
  std::string m_input_shape;
  std::string m_output_shape;

  //! resource according to topsinference
  TopsInference::IErrorManager *m_error_mgr{nullptr};
  TopsInference::handler_t m_tops_handle{nullptr};
  TopsInference::IEngine *m_engine{nullptr};
  TopsInference::topsInferStream_t m_stream{nullptr};

  //! input data
  std::vector<int> m_input_ids;
  std::vector<int> m_input_mask;
  std::vector<int> m_segment_ids;
  int m_running_batch{0};
};
}

#endif
