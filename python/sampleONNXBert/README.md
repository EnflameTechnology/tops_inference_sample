# TopsInference2.0 Bert Python Sample 

## 内容
* [目录与结构](#目录结构)
* [简介](#简介)
* [特点](#特点)
* [运行](#运行)

## 目录结构
### 1.1 结构
代码仓库的目录结构如下
```shell
├── sampleBert.py
├── squad.py 
├── tokenizetion.py
└── README.md    
```
### 1.2 依赖
通过如下命令安装依赖
```shell
python3 -m pip install -r requirement.txt
```

## 简介
样例实现了通过 Topsinference 2.0 Python 接口完成 ONNX Bert 模型推理端到端的全流程。样例的输入为 json 格式的文本文件，如 [input](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/cpp/sampleONNXBert/README.md#44-input) 所示。Bert 推理需要单词文件，准备在 [vacab](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/cpp/sampleONNXBert/vocab/vocab.txt) 下。
命令行参数参考 [TopsInference2.0 Bert C++ Sample](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/cpp/sampleONNXBert/README.md#简介) ：
```python
parser.add_argument("--model_file", type=str)
parser.add_argument("--input_file", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--vocab_file", type=str)
parser.add_argument("--save_engine", type=str)
parser.add_argument("--precision", type=str, default="mix")
parser.add_argument("--card_id", type=int, default=0)
parser.add_argument("--cluster_id", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_seq_len", type=int, default=64)
parser.add_argument("--max_query_len", type=int, default=128)
parser.add_argument("--doc_stride", type=int, default=128)
```

## 特点
样例实现了 Bert 端到端推理的全流程，整个 Pipeline 如下：
```
      --------------- 
      |读入 ONNX 模型| 
      ---------------
            |
---------------------------
|生成 TopsInference Engine|
---------------------------
            |
--------------------------
| 读入并处理输入 json 文件|
--------------------------
            |
    --------------------
    |tokenize/embedding|
    --------------------
            |
       ------------ 
       |forwarding|
       ------------
            |
         --------
        |恢复文本|
         --------
            |
    -----------------
    |写入到 json 文件|
    -----------------
```

### 3.1 注意事项
TopsInference 在创建 Engine 时，需要传入模型的 input name，这就需要用户事先获取模型信息。
本样例中，将 input name 硬编码在代码中，如果用户想要更换模型，需要用户手动修改 [sampleBert.py](https://github.com/EnflameTechnology/tops_inference_sample/blob/main/python/sampleONNXBert/sampleBert.py#L89)。
```python
input_names = "segment_ids:0,input_mask:0,input_ids:0"
```
此外，由于模型是多输出模型，代码中的后处理亦与输出顺序强相关，更换模型后，需要注意推理结果读取的数据顺序。
当前样例使用的模型为：[Bert-base-squad-op13-fp32-nvidia_seqN.onnx]()

## 运行
参考 [TopsInference2.0 Bert C++ Sample](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/cpp/sampleONNXBert/README.md#运行)。
