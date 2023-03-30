# TopsInference2.0 Bert C++ Sample 

## 内容
* [目录与结构](#目录结构)
* [简介](#简介)
* [特点](#特点)
* [准备](#准备)
* [运行](#运行)

## 目录结构
### 1.1 结构
代码仓库的目录结构如下
```shell
├── utils           --三方工具
├── vocab           --单词表
├── inputs          --样例输入
├── CMakeLists.txt  --cmake 构建脚本
├── sampleBert.h
├── sampleBert.cpp 
├── main.cpp
└── README.md    
```
### 1.2 依赖
推理涉及到文本的 token 处理，需要如下三方库：
* **boost**

## 简介
样例实现了通过 Topsinference 2.0 C++ 接口完成 ONNX Bert 模型推理端到端的全流程。样例的输入为 json 格式的文本文件，如 [input](#44-input) 所示。Bert 推理需要单词文件，准备在 [vacab](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/cpp/sampleONNXBert/vocab/vocab.txt) 下。
命令行参数如下：
```
-m --model_file model file, onnx model file or TopsInference Engine file. [essential]
-i --input_file input file for inference. Must be a json file. [essential]
-o --output_file output file to dump inference result in json format. [optional]
-v --vocab_file vocab file to tokenize. [essential]
-S --save_engine engine file to save. Note : must have write access. [optional]
-p --precision precision used to inference[fp32, fp16, mix]. Default is mix. [optional]
-d --card_id card to use. Default is 0. [optional]
-D --cluster_id cluster to use. Default is 0. [optional]
-b --batch_size batch size to build engine. Default is 1. [optional] 
-l --max_seq_len max sequence length. Ignored when use saved engine file. Default is 256. [optional]
-q --max_query_len max query length. Ignored when use saved engine file. Default is 64. [optional]
-s --doc_stride doc stride. Default is 128. [optional]
-t --perfomance switch to performance test mode when set. Default is normal test. [optional]
-h --help  help information.
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
样例程序同时提供了性能测试模式，使用随机数作为输入，测试推理性能，获得 FPS 数据。

### 3.1 注意事项
TopsInference 在创建 Engine 时，需要传入模型的 input name，这就需要用户事先获取模型信息。
本样例中，将 input name 硬编码在代码中，如果用户想要更换模型，需要用户手动修改 [sampleBert.cpp](https://github.com/EnflameTechnology/tops_inference_sample/blob/main/cpp/sampleONNXBert/sampleBert.cpp#L391) 并重新编译。
```c++
parser->setInputNames("segment_ids:0,input_mask:0,input_ids:0");
```
此外，由于模型是多输出模型，代码中的后处理亦与输出顺序强相关，更换模型后，需要注意推理结果读取的数据顺序。
当前样例使用的模型为：[Bert-base-squad-op13-fp32-nvidia_seqN.onnx]()

## 准备
### 4.1 创建 build 目录
```shell
mkdir build && cd build
```

### 4.2 执行 cmake 命令
```shell
cmake ..
```

### 4.3 编译
```shell
make
```

### 4.4 input
```json
{
  "version": "1.4",
  "data": [
    {
      "paragraphs": [
        {
          "context": "In its early years, the new convention center failed to meet attendance and revenue expectations.[12] By 2002, many Silicon Valley businesses were choosing the much larger Moscone Center in San Francisco over the San Jose Convention Center due to the latter's limited space. A ballot measure to finance an expansion via a hotel tax failed to reach the required two-thirds majority to pass. In June 2005, Team San Jose built the South Hall, a $6.77 million, blue and white tent, adding 80,000 square feet (7,400 m2) of exhibit space",
          "qas": [
            {
              "question": "where is the businesses choosing to go?",
              "id": "1"
            },
            {
              "question": "how may votes did the ballot measure need?",
              "id": "2"
            },
            {
              "question": "By what year many Silicon Valley businesses were choosing the Moscone Center?",
              "id": "3"
            }
          ]
        }
      ],
      "title": "Conference Center"
    }
  ]
}

```

## 运行
编译完成后，在 build 目录下会生成 SampleBert 的可执行程序。
* 标记为 essential 的命令行参数为必带项。

基础的测试如下：
``` shell
path/to/SampleBert -m Bert-base-squad-op13-fp32-nvidia_seqN.onnx --input path/to/input --vocab path/to/vocab
```
此时，推理结果将会打印在终端。
