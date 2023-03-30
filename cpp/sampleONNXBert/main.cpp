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

#include <string>

#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <unistd.h>

#include "sampleBert.h"

extern char *optarg;

void parseArgs(int argc, char *argv[], sample_bert::BertInferenceParam &param) {
  static struct option long_options[] = {
      {"model_file", required_argument, nullptr, 'm'},
      {"input_file", required_argument, nullptr, 'i'},
      {"output_file", required_argument, nullptr, 'o'},
      {"vocab_file", required_argument, nullptr, 'v'},
      {"save_engine", required_argument, nullptr, 'S'},
      {"precision", required_argument, nullptr, 'p'},
      {"card_id", required_argument, nullptr, 'd'},
      {"cluster_id", required_argument, nullptr, 'D'},
      {"batch_size", required_argument, nullptr, 'b'},
      {"max_seq_len", required_argument, nullptr, 'l'},
      {"max_query_len", required_argument, nullptr, 'q'},
      {"doc_stride", required_argument, nullptr, 's'},
      {"performance", no_argument, nullptr, 't'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, no_argument, nullptr, 0}};
  int ret = -1;
  int option_index;
  const char *opt = "m:i:o:v:S:p:d:D:l:q:s:th";
  while ((ret = getopt_long(argc, argv, opt, long_options, &option_index)) !=
         -1) {
    switch (ret) {
    case 'm': {
      param.model_file = std::string(optarg);
      break;
    }
    case 'i': {
      param.input_file = std::string(optarg);
      break;
    }
    case 'o': {
      param.output_file = std::string(optarg);
      break;
    }
    case 'v': {
      param.vocab_file = std::string(optarg);
      break;
    }
    case 'S': {
      param.engine_file = std::string(optarg);
      break;
    }
    case 'p': {
      param.precision = std::string(optarg);
      break;
    }
    case 'd': {
      param.card_id = atoi(optarg);
      break;
    }
    case 'D': {
      param.cluster_id = atoi(optarg);
      break;
    }
    case 'b': {
      param.batch_size = atoi(optarg);
      break;
    }
    case 'l': {
      param.max_seq_len = atoi(optarg);
      break;
    }
    case 'q': {
      param.max_query_len = atoi(optarg);
      break;
    }
    case 's': {
      param.doc_stride = atoi(optarg);
      break;
    }
    case 't': {
      param.test_mode = sample_bert::TESTMODE::PERFORMANCE;
      break;
    }
    case 'h': {
      fprintf(stdout,
              "Usage of sampleBert.\n"
              "\t-m\t--model_file\tmodel file, onnx model file or "
              "TopsInference Engine file. [essential]\n"
              "\t-i\t--input_file\tinput file for inference. Must be a json "
              "file. [essential]\n"
              "\t-o\t--output_file\toutput file to dump inference result in "
              "json format. [optional]\n"
              "\t-v\t--vocab_file\tvocab file to tokenize. [essential]\n"
              "\t-S\t--save_engine\tengine file to save. Note : must have "
              "write access. [optional]\n"
              "\t-p\t--precision\tprecision used to inference[fp32, fp16, "
              "mix]. Default is mix. [optional]\n"
              "\t-d\t--card_id\tcard to use. Default is 0. [optional]\n"
              "\t-D\t--cluster_id\tcluster to use. Default is 0. [optional]\n"
              "\t-b\t--batch_size\tbatch size to build engine. Default is 1. "
              "[optional]\n"
              "\t-l\t--max_seq_len\tmax sequence length. Ignored when use "
              "saved engine file. Default is 256. [optional]\n"
              "\t-q\t--max_query_len\tmax query length. Ignored when use saved "
              "engine file. Default is 64. [optional]\n"
              "\t-s\t--doc_stride\tdoc stride. Default is 128. [optional]\n"
              "\t-t\t--perfomance\tswitch to performance test mode when set. "
              "Default is normal test. [optional]\n"
              "\t-h\t--help\t\thelp information.\n");
      exit(0);
    }
    default: { exit(-1); }
    }
  }
  ENFLAME_ASSERT(param.model_file.size(), "Need Model File\n");
  if (param.test_mode != sample_bert::TESTMODE::PERFORMANCE) {
    ENFLAME_ASSERT(param.input_file.size(), "Need Input File\n");
    ENFLAME_ASSERT(param.vocab_file.size(), "Need Vocab File\n");
  }
}

int main(int argc, char *argv[]) {
  //! 1.parse arguements
  sample_bert::BertInferenceParam param;
  parseArgs(argc, argv, param);

  //! 2. global init
  sample_bert::global_init_tops();

  //! 3.create Sample Bert instance
  sample_bert::SampleBert sample(param);

  //! 4.do test
  sample.exec();

  //! global release
  sample_bert::global_release_tops();

  return 0;
}
