
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

#ifndef __TOKENIZER__H__
#define __TOKENIZER__H__

#include <map>
#include <vector>

using namespace std;

vector<string> whitespace_tokenize(string text);

map<string, int> read_vocab(const char *filename);

class BasicTokenizer {
public:
  bool do_lower_case_;
  vector<string> never_split_;

  BasicTokenizer(bool do_lower_case = false,
                 vector<string> never_split = {"[UNK]", "[SEP]", "[PAD]",
                                               "[CLS]", "[MASK]"}) {
    do_lower_case_ = do_lower_case;
    never_split_ = never_split;
  }

  string _clean_text(string text);

  vector<string> _run_split_on_punc(string text);

  string _run_strip_accents(string text);

  string _tokenize_chinese_chars(string text);

  string utf8chr(int cp);

  bool _is_chinese_char(int cp);

  vector<string> tokenize(string text);

  void truncate_sequences(vector<string> &textA, vector<string> &textB,
                          const char *truncation_strategy, int max_seq_length);
};

class WordpieceTokenizer {
public:
  map<string, int> vocab_;
  string unk_token_;
  int max_input_chars_per_word_;

  WordpieceTokenizer(){};

  WordpieceTokenizer(map<string, int> vocab, string unk_token = "[UNK]",
                     int max_input_chars_per_word = 100) {
    vocab_ = vocab;
    unk_token_ = unk_token;
    max_input_chars_per_word_ = max_input_chars_per_word;
  }

  void add_vocab(map<string, int> vocab);

  vector<string> tokenize(string text);
};

class BertTokenizer {
public:
  map<string, int> vocab;
  map<int, string> ids_to_tokens;
  bool do_lower_case_;
  bool do_basic_tokenize_;
  int maxlen_;
  BasicTokenizer basic_tokenizer;
  WordpieceTokenizer wordpiece_tokenizer;

  BertTokenizer(){};

  BertTokenizer(const char *vocab_file, bool do_lower_case = false,
                int max_len = 512, bool do_basic_tokenize = true,
                vector<string> never_split = {"[UNK]", "[SEP]", "[PAD]",
                                              "[CLS]", "[MASK]"}) {
    vocab = read_vocab(vocab_file);
    for (map<string, int>::iterator i = vocab.begin(); i != vocab.end(); ++i)
      ids_to_tokens[i->second] = i->first;
    do_basic_tokenize_ = do_basic_tokenize;
    do_lower_case_ = do_lower_case;
    basic_tokenizer = BasicTokenizer(do_lower_case_);
    wordpiece_tokenizer.add_vocab(vocab);
    maxlen_ = max_len;
  }

  void add_vocab(const char *vocab_file);

  vector<string> tokenize(string text);

  vector<float> convert_tokens_to_ids(vector<string> tokens);

  vector<int> tokens_to_ids(vector<string> tokens);

  void encode(string textA, string textB, vector<float> &input_ids,
              vector<float> &input_mask, vector<float> &segment_ids,
              int max_seq_length = 512,
              const char *truncation_strategy = "longest_first");
};
#endif
