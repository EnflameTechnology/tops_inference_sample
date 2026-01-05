# 一、介绍
本目录下代码用于使能 topsinference2.0 进行 NLP_Bert 推理的 Python 代码。

## 1、依赖
```shell
python3 -m pip install -r requirement.txt
```

## 2、运行
```
usage: python3 sampleBert.py [-h] [--model_file MODEL_FILE] [--input_file INPUT_FILE] [--output_file OUTPUT_FILE] [--vocab_file VOCAB_FILE] [--save_engine SAVE_ENGINE] [--precision PRECISION] [--card_id CARD_ID] [--cluster_id CLUSTER_ID] [--batch_size BATCH_SIZE] [--max_seq_len MAX_SEQ_LEN]
                     [--max_query_len MAX_QUERY_LEN] [--doc_stride DOC_STRIDE]

optional arguments:
  -h, --help            show this help message and exit
  --model_file MODEL_FILE
  --input_file INPUT_FILE
  --output_file OUTPUT_FILE
  --vocab_file VOCAB_FILE
  --engine SAVE_ENGINE
  --precision PRECISION
  --card_id CARD_ID
  --cluster_id CLUSTER_ID
  --batch_size BATCH_SIZE
  --max_seq_len MAX_SEQ_LEN
  --max_query_len MAX_QUERY_LEN
  --doc_stride DOC_STRIDE

```

* `--model_file`：onnx模型文件[here](http://artifact.enflame.cn/artifactory/enflame_model_zoo/official/nlp/bert/bert_base-squad-nvidia-op13-fp32-N.onnx)

* `--input_file`: /PATH/TO/tops_inference_samples/data/bert/inputs.json
* `--output_file`: 结果保存文件（json）
* `--vocab_fiel`: /PATH/TO/tops_inference_samples/models/vocab.txt
* `--engine`: 使用编译好的engine文件
* `--precision`: 编译精度，可为"default", "fp16", "mix" 
* `--card_id`: 使用哪张卡用于推理， 默认为0
* `--cluster_id`: 使用卡中的哪个cluster用于推理, 默认为0
* `--batch_size`: 默认为1 
* `--max_seq_len`：控制输入序列的最大长度
* `--max_query_len`：控制查询的最大长度
* `--doc_stride`：控制文档滑动窗口的步长，确保长文档被正确分割和处理

## 3、示例
```
using onnx
python3 sampleBert.py --model_file=/PATH/TO/tops_inference_samples/models/bert_base-squad-nvidia-op13-fp32-N.onnx --input_file=/PATH/TO/tops_inference_samples/data/bert/inputs.json --output_file=/PATH/TO/tops_inference_samples/data/bert/output.json --vocab_file=/PATH/TO/tops_inference_samples/models/vocab.txt

# using engine
python3 sampleBert.py --engine=/PATH/TO/tops_inference_samples/engines/bert_base-squad-nvidia-op13-fp32-N.engine --input_file=/PATH/TO/tops_inference_samples/data/bert/inputs.json --output_file=/PATH/TO/tops_inference_samples/data/bert/output.json --vocab_file=/PATH/TO/tops_inference_samples/models/vocab.txt
```

## 4、注意事项
bert 推理以及后处理与模型强相关。