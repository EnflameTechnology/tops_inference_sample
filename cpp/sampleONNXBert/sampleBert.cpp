
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

#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include <unistd.h>

#include "sampleBert.h"
#include "utils/json.hpp"
#include "utils/tokenizer.h"

namespace {
//======================== Json Read ============================//
struct BertJsonOneQuestion {
  std::string _question;
  std::string _id;
};
void from_json(const nlohmann::json &json_obj, BertJsonOneQuestion &t) {
  json_obj.at("question").get_to(t._question);
  json_obj.at("id").get_to(t._id);
}

struct BertJsonQuestionPacket {
  std::vector<BertJsonOneQuestion> _qas;
};
void from_json(const nlohmann::json &json_obj, BertJsonQuestionPacket &t) {
  auto qas_size = json_obj.size();
  t._qas.resize(qas_size);
  for (int i = 0; i < qas_size; ++i) {
    t._qas[i] = json_obj[i];
  }
}

struct BertJsonQuery {
  std::string _context;
  BertJsonQuestionPacket _question_packet;
};
void from_json(const nlohmann::json &json_obj, BertJsonQuery &t) {
  json_obj.at("context").get_to(t._context);
  t._question_packet = json_obj["qas"];
}

struct BertJsonParagraph {
  std::vector<BertJsonQuery> _queries;
};
void from_json(const nlohmann::json &json_obj, BertJsonParagraph &t) {
  auto query_size = json_obj.size();
  t._queries.resize(query_size);
  for (int i = 0; i < query_size; ++i) {
    t._queries[i] = json_obj[i];
  }
}

struct BertJsonBlock {
  std::string _title;
  BertJsonParagraph _paragraphs;
};
void from_json(const nlohmann::json &json_obj, BertJsonBlock &t) {
  json_obj.at("title").get_to(t._title);
  json_obj.at("paragraphs").get_to(t._paragraphs);
}

struct BertJsonData {
  std::vector<BertJsonBlock> _blocks;
};
void from_json(const nlohmann::json &json_obj, BertJsonData &t) {
  auto block_size = json_obj.size();
  t._blocks.resize(block_size);
  for (int i = 0; i < block_size; ++i) {
    t._blocks[i] = json_obj[i];
  }
}

struct BertJsonFile {
  std::string _version;
  BertJsonData _data;
};
void from_json(const nlohmann::json &json_obj, BertJsonFile &t) {
  json_obj.at("version").get_to(t._version);
  t._data = json_obj["data"];
}

//======================== Json Write ============================//
struct BertJsonWriteAnswer {
  std::string _question_context;
  std::string _answer;
};
void to_json(nlohmann::json &json_obj, const BertJsonWriteAnswer &t) {
  json_obj["question"] = t._question_context;
  json_obj["answer"] = t._answer;
}

struct BertJsonWriteData {
  std::string _context;
  std::vector<BertJsonWriteAnswer> _results;
};
void to_json(nlohmann::json &json_obj, const BertJsonWriteData &t) {
  json_obj["context"] = t._context;
  auto res_size = t._results.size();
  for (int i = 0; i < res_size; ++i) {
    json_obj["results"].push_back(t._results[i]);
  }
}

struct BertJsonWriteFile {
  std::vector<BertJsonWriteData> _datas;
};
void to_json(nlohmann::json &json_obj, const BertJsonWriteFile &t) {
  auto data_size = t._datas.size();
  for (int i = 0; i < data_size; ++i) {
    json_obj["data"].push_back(t._datas[i]);
  }
}
}; // namespace

namespace {
/**
 * @brief one feature of one question. created by doc stride
 *
 */
struct BertOneFeature {
  std::vector<std::string> _tokens; //< tokens of this feature
  std::unordered_map<int, int> _token_to_orig_map;
  std::unordered_map<int, int> _token_is_max_context;
};

/**
 * @brief one question of one paragraph. contains an id and a set of features
 *
 */
struct BertOneQuestion {
  std::string _question_context;         //< question context
  std::vector<BertOneFeature> _features; //< features
};

/**
 * @brief use this struct to store a set of questions and one paragraph
 *
 */
struct BertOneParagraph {
  std::string _context;                    //< paragraph's context
  std::vector<std::string> _doc_tokens;    //< tokens of paragraph's context
  std::vector<BertOneQuestion> _questions; //< questions of this context
};

std::vector<BertOneParagraph> input_paragraphs;
} // namespace

namespace string_tools {
//! join like python " ".join(strings)
std::string join(std::vector<std::string> &strings) {
  std::string ret;
  for (auto &&s : strings) {
    ret = ret + s + " ";
  }
  ret.pop_back();
  return ret;
};

//! split string with delim
std::vector<std::string> split(const std::string &text, char delim) {
  std::vector<std::string> ret;
  std::stringstream ss(text);
  std::string tok;
  while (std::getline(ss, tok, delim)) {
    ret.push_back(tok);
  }
  return ret;
};

//! strip spaces in a string
std::pair<std::string, std::unordered_map<int, int>>
strip_spaces(const std::string &text) {
  std::string ns_chars;
  std::unordered_map<int, int> ns_to_s_map;
  std::pair<std::string, std::unordered_map<int, int>> ret;
  int idx = 0;
  for (int i = 0; i < text.size(); ++i) {
    if (text[i] == ' ')
      continue;
    ns_to_s_map[idx] = i;
    ns_chars.push_back(text[i]);
    idx++;
  }
  ret.first = std::move(ns_chars);
  ret.second = std::move(ns_to_s_map);
  return ret;
}

//! get final text
std::string get_final_text(const std::string &pred_text,
                           const std::string &orig_text) {
  BasicTokenizer tokenizer(true);
  //! tokenize orig_text and compare with pred_text
  auto token_orig_text = tokenizer.tokenize(orig_text);
  auto tok_text = join(token_orig_text);
  auto start_pos = tok_text.find(pred_text, 0);
  if (start_pos == std::string::npos) {
    return orig_text;
  }
  auto end_pos = start_pos + pred_text.size() - 1;
  auto orig_ss_text = strip_spaces(orig_text);
  auto tok_ss_text = strip_spaces(tok_text);
  auto &orig_ns_text = orig_ss_text.first;
  auto &orig_ns_to_s_map = orig_ss_text.second;
  auto &tok_ns_text = tok_ss_text.first;
  auto &tok_ns_to_s_map = tok_ss_text.second;

  if (orig_ns_text.size() != tok_ns_text.size()) {
    return orig_text;
  }

  std::unordered_map<int, int> tok_s_to_ns_map;
  for (auto &&iter : tok_ns_to_s_map) {
    tok_s_to_ns_map.insert({iter.second, iter.first});
  }

  auto orig_start_pos = -1;
  auto iter0 = tok_s_to_ns_map.find(start_pos);
  if (iter0 != tok_s_to_ns_map.end()) {
    auto iter1 = orig_ns_to_s_map.find(iter0->second);
    if (iter1 != orig_ns_to_s_map.end()) {
      orig_start_pos = iter1->second;
    }
  }

  if (orig_start_pos == -1) {
    return orig_text;
  }

  auto orig_end_pos = -1;
  auto iter2 = tok_s_to_ns_map.find(end_pos);
  if (iter2 != tok_s_to_ns_map.end()) {
    auto iter3 = orig_ns_to_s_map.find(iter2->second);
    if (iter3 != orig_ns_to_s_map.end()) {
      orig_end_pos = iter3->second;
    }
  }

  if (orig_end_pos == -1) {
    return orig_text;
  }

  auto count = orig_end_pos + 1 - orig_start_pos;
  auto output_text = orig_text.substr(orig_start_pos, count);
  return output_text;
};
} // namespace string_tools

namespace sample_bert {

void global_init_tops() { TopsInference::topsInference_init(); }

void global_release_tops() { TopsInference::topsInference_finish(); }

SampleBert::SampleBert(BertInferenceParam &param) {
  m_model_file = param.model_file;
  m_input_file = param.input_file;
  m_output_file = param.output_file;
  m_vocab_file = param.vocab_file;
  m_engine_file = param.engine_file;
  m_card_id = param.card_id;
  m_cluster_id = param.cluster_id;
  m_batch_size = param.batch_size;
  m_max_seq_len = param.max_seq_len;
  m_doc_stride = param.doc_stride;
  m_max_query_len = param.max_query_len;
  m_test_mode = param.test_mode;
  if (param.precision == "fp32") {
    m_precision = TopsInference::BuildFlag::TIF_KTYPE_DEFAULT;
  } else if (param.precision == "fp16") {
    m_precision = TopsInference::BuildFlag::TIF_KTYPE_FLOAT16;
  } else {
    ENFLAME_ASSERT((param.precision == "mix"), "Invalid precision : %s\n",
                   param.precision.c_str());
    m_precision = TopsInference::BuildFlag::TIF_KTYPE_MIX_FP16;
  }
}

SampleBert::~SampleBert() {
  //! in case release() is not called
  release();
}

void SampleBert::exec() {
  if (m_test_mode == TESTMODE::NORMAL) {
    normal_test(m_cluster_id);
  } else {
    performance_test();
  }
}

void SampleBert::normal_test(int cluster_id) {
  //! init
  std::vector<uint32_t> cluster_ids{uint32_t(cluster_id)};
  init(cluster_ids);

  //! preprocess input data
  pre_process();

  //! fowarding and write to file
  forward();

  //! release
  release();
}

void SampleBert::performance_test() {
  //! init 6VG
  std::vector<uint32_t> cluster_ids{0, 1, 2, 3, 4, 5};
  init(cluster_ids);

  //! forwarding
  static const int running_bs = 150;
  static const int iters = 100;
  std::vector<int> input_ids(running_bs * m_max_seq_len);
  std::vector<int> segment_ids(running_bs * m_max_seq_len, 1);
  std::vector<int> input_mask(running_bs * m_max_seq_len, 1);
  std::vector<void *> inputs{segment_ids.data(), input_mask.data(),
                             input_ids.data()};

  std::vector<float> end_logits(iters * running_bs * m_max_seq_len);
  std::vector<float> start_logits(iters * running_bs * m_max_seq_len);

  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  //! run with batch
  for (int i = 0; i < iters; ++i) {
    auto e_data = &end_logits[i * running_bs * m_max_seq_len];
    auto s_data = &start_logits[i * running_bs * m_max_seq_len];
    std::vector<void *> outputs{e_data, s_data};
    auto ret = m_engine->run_with_batch(
        running_bs, inputs.data(), outputs.data(),
        TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST, m_stream);
    ENFLAME_CHECK(ret, m_error_mgr);
  }
  //! wait stream
  ENFLAME_CHECK(m_stream->synchronize(), m_error_mgr);
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  auto time_diff =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  auto FPS = running_bs * iters / (time_diff / 1000.0);
  fprintf(stdout, "Running Batch : %d\n", running_bs);
  fprintf(stdout, "Running Iters : %d\n", iters);
  fprintf(stdout, "Running Time : %.2fs\n", time_diff / 1000.0);
  fprintf(stdout, "FPS : %.2f/s\n", FPS);
}

void SampleBert::init(const std::vector<uint32_t> &cluster_ids) {
  //! set device to cluste id
  m_tops_handle = TopsInference::set_device(m_card_id, cluster_ids.data(),
                                            cluster_ids.size());

  //! error manager
  m_error_mgr = TopsInference::create_error_manager();

  //! try load as engine file
  m_engine = TopsInference::create_engine(m_error_mgr);
  ENFLAME_CHECK(m_engine, m_error_mgr);
  auto ret = m_engine->loadExecutable(m_model_file.c_str());

  if (ret) {
    ENFLAME_LOG_WARN("Load Engine File : %s\n", m_model_file.c_str());
  } else {
    //! try load as onnx file
    //! parser
    auto parser =
        TopsInference::create_parser(TopsInference::TIF_ONNX, m_error_mgr);
    ENFLAME_CHECK(parser, m_error_mgr);

    //! set input and shape, then read model
    //! TODO: hard code to set name. user should change it when changing model
    parser->setInputNames("segment_ids:0,input_mask:0,input_ids:0");
    //! auto set input shape to
    //! "batch_size,m_max_seq_len:batch_size,m_max_seq_len:batch_size,m_max_seq_len"
    std::string one_shape =
        std::to_string(m_batch_size) + "," + std::to_string(m_max_seq_len);
    std::string input_shapes = one_shape + ":" + one_shape + ":" + one_shape;
    parser->setInputShapes(input_shapes.c_str());
    auto network = parser->readModel(m_model_file.c_str());
    ENFLAME_CHECK(network, m_error_mgr);

    //! optimizer
    auto optimizer = TopsInference::create_optimizer(m_error_mgr);
    ENFLAME_CHECK(optimizer, m_error_mgr);

    //! build engine
    auto optimizer_config = optimizer->getConfig();
    optimizer_config->setBuildFlag(m_precision);
    m_engine = optimizer->build(network);
    ENFLAME_CHECK(m_engine, m_error_mgr);

    //! save exectutable
    if (m_engine_file.size() != 0) {
      auto ret = m_engine->saveExecutable(m_engine_file.c_str());
      ENFLAME_CHECK(ret, m_error_mgr);
    }
    TopsInference::release_network(network);
    TopsInference::release_optimizer(optimizer);
    TopsInference::release_parser(parser);
  }

  //! create stream
  ret = TopsInference::create_stream(&m_stream);
  ENFLAME_CHECK(ret, m_error_mgr);
}

void SampleBert::release() {
  if (m_stream) {
    TopsInference::destroy_stream(m_stream);
    m_stream = nullptr;
  }

  if (m_engine) {
    TopsInference::release_engine(m_engine);
    m_engine = nullptr;
  }

  if (m_tops_handle) {
    TopsInference::release_device(m_tops_handle);
    m_tops_handle = nullptr;
  }

  if (m_error_mgr) {
    TopsInference::release_error_manager(m_error_mgr);
    m_error_mgr = nullptr;
  }
}

void SampleBert::pre_process() {
  //! read json file
  nlohmann::json bert_json;
  std::ifstream model_input(m_input_file);
  ENFLAME_ASSERT(model_input.is_open(), "%s %s\n", m_input_file.c_str(),
                 strerror(errno));
  model_input >> bert_json;

  BertJsonFile json_file;
  from_json(bert_json, json_file);

  //! tokenize vocabulary file
  ENFLAME_ASSERT(access(m_vocab_file.c_str(), R_OK) == 0, "%s %s\n",
                 m_vocab_file.c_str(), strerror(errno));
  auto tokenizer = BertTokenizer(m_vocab_file.c_str(), true);

  auto is_whitespace = [](char c) {
    if (c == ' ')
      return true;
    if (c == '\t')
      return true;
    if (c == '\r')
      return true;
    if (c == '\n')
      return true;
    return false;
  };

  using string_vector = std::vector<std::string>;
  using int_vector = std::vector<int>;
  auto blocks = json_file._data._blocks;
  for (auto &&block : blocks) {
    for (auto &&query : block._paragraphs._queries) {
      //! a new BertOneParagraph
      BertOneParagraph paragraph;
      auto &context = paragraph._context;
      auto &doc_tokens = paragraph._doc_tokens;
      auto &questions = paragraph._questions;
      int idx = 0;
      context = query._context;
      //! split all words in context into doc_tokens
      while (idx < context.size()) {
        while (is_whitespace(context[idx]))
          idx++;
        int word_len = 1;
        while (idx + word_len < context.size() &&
               !is_whitespace(context[idx + word_len]))
          word_len++;
        doc_tokens.push_back(context.substr(idx, word_len));
        idx += word_len + 1;
      }

      //! tokenize doc tokens
      int_vector tok_to_orig_index;
      int_vector orig_to_tok_index;
      string_vector all_doc_tokens;
      int total_tokens = 0;
      for (int i = 0; i < doc_tokens.size(); ++i) {
        orig_to_tok_index.push_back(total_tokens);
        auto sub_tokens = tokenizer.tokenize(doc_tokens[i]);
        total_tokens += sub_tokens.size();
        for (auto &&sub_token : sub_tokens) {
          tok_to_orig_index.push_back(i);
          all_doc_tokens.push_back(sub_token);
        }
      }

      //! for each question, prepare input data
      for (auto &&qas : query._question_packet._qas) {
        auto qas_context = qas._question;
        //! one question struct
        BertOneQuestion question;
        question._question_context = qas_context;
        auto &features = question._features;

        //! token question and cut off by m_max_query_len
        auto query_tokens = tokenizer.tokenize(qas_context);
        if (query_tokens.size() > m_max_query_len) {
          query_tokens.resize(m_max_query_len);
        }

        //! max len
        int max_tokens_for_doc = m_max_seq_len - query_tokens.size() - 3;

        //! cut off docs by doc stride
        std::vector<std::pair<int, int>> doc_spans;
        int start_offset = 0;
        int nr_all_doc_tokens = all_doc_tokens.size();
        while (start_offset < nr_all_doc_tokens) {
          auto len =
              std::min(max_tokens_for_doc, nr_all_doc_tokens - start_offset);
          doc_spans.emplace_back(start_offset, len);
          if (len != max_tokens_for_doc)
            break;
          start_offset += std::min(len, m_doc_stride);
        }

        //! deal each doc_span
        for (int i = 0; i < doc_spans.size(); ++i) {
          BertOneFeature feature;
          auto &tokens = feature._tokens;
          auto &token_to_orig_map = feature._token_to_orig_map;
          auto &token_is_max_context = feature._token_is_max_context;
          int_vector segment_ids;

          tokens.push_back({"[CLS]"});
          segment_ids.push_back(0);
          for (auto &&token : query_tokens) {
            tokens.push_back(token);
            segment_ids.push_back(0);
          }
          tokens.push_back({"[SEP]"});
          segment_ids.push_back(0);

          auto start_offset = doc_spans[i].first;
          auto len = doc_spans[i].second;
          for (int j = 0; j < len; ++j) {
            auto split_token_index = start_offset + j; // start_offset + j
            auto tokens_len = tokens.size();
            token_to_orig_map[tokens_len] =
                tok_to_orig_index[split_token_index];

            //! check whether token at split_token_index is in the best doc span
            auto check_is_max_context =
                [](std::vector<std::pair<int, int>> &doc_spans,
                   int cur_idx_in_doc_spans, int pos) {
                  float best_score = 0;
                  int best_idx_in_doc_spans = -1;
                  for (int i = 0; i < doc_spans.size(); ++i) {
                    auto len = doc_spans[i].second;
                    auto start_idx = doc_spans[i].first;
                    auto end_idx = start_idx + len - 1;
                    if (pos < start_idx)
                      continue;
                    if (pos > end_idx)
                      continue;
                    auto nr_left_context = pos - start_idx;
                    auto nr_right_context = end_idx - pos;
                    auto score = std::min(nr_left_context, nr_right_context) +
                                 0.01 * len;
                    if (score > best_score) {
                      best_score = score;
                      best_idx_in_doc_spans = i;
                    }
                  }
                  return cur_idx_in_doc_spans == best_idx_in_doc_spans;
                };

            token_is_max_context[tokens_len] =
                check_is_max_context(doc_spans, i, split_token_index);
            tokens.push_back(all_doc_tokens[split_token_index]);
            segment_ids.push_back(1);
          }
          tokens.push_back({"[SEP]"});
          segment_ids.push_back(1);

          auto input_ids = tokenizer.tokens_to_ids(tokens);
          std::vector<int> input_mask(input_ids.size(), 1);

          //! padding
          auto input_ids_len = input_ids.size();
          while (input_ids_len < m_max_seq_len) {
            input_ids.push_back(0);
            input_mask.push_back(0);
            segment_ids.push_back(0);
            input_ids_len++;
          }
          m_input_ids.insert(m_input_ids.end(), input_ids.begin(),
                             input_ids.end());
          m_input_mask.insert(m_input_mask.end(), input_mask.begin(),
                              input_mask.end());
          m_segment_ids.insert(m_segment_ids.end(), segment_ids.begin(),
                               segment_ids.end());
          //! add one feature
          features.emplace_back(feature);
          //! add running batch
          m_running_batch++;
        }
        //! add one question
        questions.emplace_back(question);
      }
      //! add one paragraph
      input_paragraphs.emplace_back(paragraph);
    }
  }
}

void SampleBert::forward() {
  std::vector<void *> inputs{m_segment_ids.data(), m_input_mask.data(),
                             m_input_ids.data()};

  //! TODO: changing output memory according to model
  std::vector<float> end_logits(m_running_batch * m_max_seq_len);
  std::vector<float> start_logits(m_running_batch * m_max_seq_len);
  std::vector<void *> outputs{end_logits.data(), start_logits.data()};

  //! run with batch
  auto ret = m_engine->run_with_batch(
      m_running_batch, inputs.data(), outputs.data(),
      TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST, m_stream);
  ENFLAME_CHECK(ret, m_error_mgr);

  //! wait stream
  ENFLAME_CHECK(m_stream->synchronize(), m_error_mgr);

  //! post process
  post_process(start_logits, end_logits);
}

void SampleBert::post_process(const std::vector<float> &start_logits,
                              const std::vector<float> &end_logits) {
  //! get top10's index at logits
  auto cmp_index = [](std::pair<float, int> a, std::pair<float, int> b) {
    return a.first > b.first;
  };
  auto get_best10_indexes =
      [&cmp_index](std::vector<std::pair<float, int>> &logits) {
        std::sort(logits.begin(), logits.end(), cmp_index);
        std::vector<int> indexes;
        const size_t nbest = 10;
        for (int i = 0; i < std::min(nbest, logits.size()); ++i) {
          indexes.push_back(logits[i].second);
        }
        return indexes;
      };

  int nr_paragraphs = input_paragraphs.size();
  //! each feature has start_logits and end_logits
  int start_pos = 0;
  BertJsonWriteFile bert_json_write_file;
  for (int i = 0; i < nr_paragraphs; ++i) {
    auto &paragraph = input_paragraphs[i];
    auto &doc_tokens = paragraph._doc_tokens;
    auto &questions = paragraph._questions;
    auto &context = paragraph._context;
    int nr_questions = questions.size();
    BertJsonWriteData bert_json_write_data;
    bert_json_write_data._context = context;
    for (int j = 0; j < nr_questions; ++j) {
      auto &question = questions[j];
      auto &features = question._features;
      int nr_features = features.size();
      auto &question_context = question._question_context;
      //! create a BertJsonWriteAnswer
      BertJsonWriteAnswer bert_json_write_ans;
      bert_json_write_ans._question_context = question_context;
      int best_start_idx = -1;
      int best_end_idx = -1;
      int best_feature_idx = -1;
      auto max_score = std::numeric_limits<float>::lowest();
      for (int k = 0; k < nr_features; ++k) {
        auto &feature = features[k];
        auto &tokens = feature._tokens;
        auto &token_to_orig_map = feature._token_to_orig_map;
        auto &token_is_max_context = feature._token_is_max_context;
        std::vector<std::pair<float, int>> s_logits;
        std::vector<std::pair<float, int>> e_logits;
        for (int idx = start_pos; idx < start_pos + m_max_seq_len; ++idx) {
          s_logits.emplace_back(start_logits[idx], idx - start_pos);
          e_logits.emplace_back(end_logits[idx], idx - start_pos);
        }
        auto start_idx = get_best10_indexes(s_logits);
        auto end_idx = get_best10_indexes(e_logits);
        for (auto &&s_idx : start_idx) {
          if (s_idx >= tokens.size())
            continue;
          if (token_to_orig_map.find(s_idx) == token_to_orig_map.end())
            continue;
          auto iter = token_is_max_context.find(s_idx);
          if (iter == token_is_max_context.end())
            continue;
          if (iter->second == false)
            continue;
          for (auto &&e_idx : end_idx) {
            if (e_idx >= tokens.size())
              continue;
            if (token_to_orig_map.find(e_idx) == token_to_orig_map.end())
              continue;
            if (e_idx < s_idx)
              continue;
            //! valid s_idx and e_idx
            //! find best pair of s_idx and e_idx
            auto score =
                start_logits[start_pos + s_idx] + end_logits[start_pos + e_idx];
            if (score > max_score) {
              max_score = score;
              best_start_idx = s_idx;
              best_end_idx = e_idx;
              best_feature_idx = k;
            }
          }
        }
        start_pos += m_max_seq_len;
      }
      //! no prediction
      if (best_start_idx == -1) {
        bert_json_write_ans._answer = "No predictions";
        bert_json_write_data._results.push_back(bert_json_write_ans);
        continue;
      }

      //! construct prediction
      auto &feature = features[best_feature_idx];
      auto &tokens = feature._tokens;
      auto &token_to_orig_map = feature._token_to_orig_map;
      auto &token_is_max_context = feature._token_is_max_context;

      std::vector<std::string> tok_tokens{tokens.begin() + best_start_idx,
                                          tokens.begin() + best_end_idx + 1};
      auto orig_doc_start = token_to_orig_map[best_start_idx];
      auto orig_doc_end = token_to_orig_map[best_end_idx];
      std::vector<std::string> orig_tokens{doc_tokens.begin() + orig_doc_start,
                                           doc_tokens.begin() + orig_doc_end +
                                               1};

      //! predict text
      std::string predict_text = string_tools::join(tok_tokens);

      //! De-tokenize WordPieces that have been split off.
      while (true) {
        auto pre_pos = predict_text.find(" ##", 0);
        if (pre_pos == std::string::npos)
          break;
        predict_text.erase(pre_pos, 3);
      }
      while (true) {
        auto pre_pos = predict_text.find("##", 0);
        if (pre_pos == std::string::npos)
          break;
        predict_text.erase(pre_pos, 2);
      }

      //! Clean whitespace
      if (*predict_text.begin() == ' ') {
        predict_text.erase(0, 1);
      }
      if (*predict_text.rbegin() == ' ') {
        predict_text.pop_back();
      }
      auto split_predict_text = string_tools::split(predict_text, ' ');
      predict_text = string_tools::join(split_predict_text);

      //! original text
      std::string origin_text = string_tools::join(orig_tokens);

      bert_json_write_ans._answer =
          string_tools::get_final_text(predict_text, origin_text);

      //! add to json
      bert_json_write_data._results.push_back(bert_json_write_ans);
    }
    bert_json_write_file._datas.push_back(bert_json_write_data);
  }

  //! if output file exist, write to file. Or to stdout
  nlohmann::json bert_json = bert_json_write_file;
  if (!m_output_file.empty()) {
    std::ofstream output(m_output_file);
    output << bert_json.dump(2);
  } else {
    fprintf(stdout, "%s", bert_json.dump(2).c_str());
  }
}
} // namespace sample_bert
