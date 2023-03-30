# 燧原科技 TopsInference2.0 接口使用样例

## 简介
本文档包含共7个样例，以便向用户展示 TopsInference2.0 接口（以下简称为2.0接口）的变化以及使用方式。它们分别是4个基于 C++ 开发的样例以及3个基于 Python 开发的样例。所有的样例代码均发布在[Github:tops_inference_sample](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/) 。

## 1 总览
|        标题         |             样例名称及代码仓库            |                      简介                         |
|  ----------------   | ------------------------------  | ------------------------------------------------- |
| [Bert C++ Sample](#311-bert-c-sample) | [cpp:sampleONNXBert](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/cpp/sampleONNXBert) | 基于 C++ 的应用，使用 GCU 完成 ONNX Bert 模型推理，包含 CPU 上的前后处理过程，可以获得端到端的结果 |
| [ResNet C++ Sample](#312-resnet-c-sample) | [cpp:sampleONNXResNet](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/cpp/sampleONNXResNet) | 使用 GCU 和 2.0 接口完成 ResNet50 ONNX 模型推理的 C++ 应用  |
| [Yolov5 C++ Sample](#313-yolov5-c-sample) | [cpp:sampleONNXYolov5](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/cpp/sampleONNXYolov5) | 一个端到端的 C++ 应用，使用 GCU 完成 Yolov5 ONNX 模型的推理，包含图片的前后处理 |
| [Paralle Pipeline C++ Sample](#314-paralle-pipeline-c-sample) | [cpp:sampleParallelPipeline](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/cpp/sampleParallelPipeline) | C++ 模拟 GCU 并行业务模式，检测和分类同一时刻运行在不同的线程中，同一线程中对多 batch 通过 stream 分组异步执行 |
| [Serial Pipeline C++ Sample](#315-serial-pipeline-c-sample) | [cpp:sampleSimpleSerialPipeline](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/cpp/sampleSimpleSerialPipeline)  | C++ 模拟 GCU 串行业务线模式（同一线程中先检测后分类） |
| [Bert Python Sample](#321-bert-python-sample) | [python:sampleONNXBert](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/python/sampleONNXBert) | 基于 Python 的应用，使用 GCU 完成 ONNX Bert 模型推理，包含 CPU 上的前后处理过程，可以获得端到端的结果 |
| [ResNet Python Sample](#322-resnet-python-sample) | [python:sampleONNXResNet](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/python/sampleONNXResNet)  | 使用 GCU 和 2.0 接口完成 ResNet50 ONNX 模型推理的 Python 应用 |
| [Yolov5 Python Sample](#323-yolov5-python-sample) | [python:sampleONNXYolov5](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/python/sampleONNXYolov5) | 一个端到端的 Python 应用，使用 GCU 完成 Yolov5 ONNX 模型的推理，包含图片的前后处理 |

## 2 必要条件
无论开发 C++ 或是 Python 应用程序时，用户都需要保证2.0接口被正确安装，详细参见[TopsInference 安装](http://docs.enflame-tech.com/2-install/sw_install/content/source/index.html) 

## 3 各 Sample 简介
此次基于2.0接口开发的样例程序，以具有代表性的 Bert、ResNet、Yolov5 的推理向用户展示2.0接口的变化。同时基于 C++ 实现多线程单模型并行推理、单线程多模型串行推理样例应用以模拟业务场景。

### 3.1 C++ Sample
C++ Bert、ResNet、Yolov5 模型推理及前后处理，此外，亦开发了模拟业务场景 Pipeline 的应用。
#### 3.1.1 Bert C++ Sample
样例展示通过 C++ 2.0 接口在 GCU 上进行 ONNX Bert 模型的推理。样例的构建运行等参见 [GitHub:cpp/sampleONNXBert/README.md](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/cpp/sampleONNXBert/README.md)。该样例包含了 CPU 上的文字前后处理过程，获得端到端的结果。其输入要求 json 格式的文本文件，输出为 json 格式的文本文件。
#### 3.1.2 ResNet C++ Sample
样例展示 GCU 上完成 ONNX ResNet50 模型推理的全过程。样例构建运行参考 [GitHub:cpp/sampleONNXResNet/README.md](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/cpp/sampleONNXResNet/README.md)。样例完成图片的前处理后送入 GCU 推理。在屏幕上打印接口的耗时和分类的索引信息。
#### 3.1.3 Yolov5 C++ Sample
样例完成 GCU 上使用 YoloV5 完成端到端的图片识别任务。样例构建运行参考 [GitHub:cpp/sampleONNXYolov5/README.md](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/cpp/sampleONNXYolov5/README.md)。
#### 3.1.4 Paralle Pipeline C++ Sample
样例模拟多线程同时使用 GCU，并行完成检测和分类推理任务的业务场景。样例构建运行参考 [GitHub:cpp/sampleParallelPipeline/README.md](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/cpp/sampleParallelPipeline/README.md)。
#### 3.1.5 Serial Pipeline C++ Sample
样例模拟串行的业务场景，在单线程中先后使用 GCU 完成检测和分类。样例构建运行参考 [GitHub:cpp/sampleSimpleSerialPipeline/README.md](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/cpp/sampleSimpleSerialPipeline/README.md)

### 3.2 Python Sample
Python Bert、ResNet、Yolov5 模型推理及前后处理。
#### 3.2.1 Bert Python Sample
同 [Bert C++ Sample](#311-bert-c-sample)，该样例展示通过 Python 2.0 接口在 GCU 上进行 ONNX Bert 模型的推理。样例的构建运行等参见 [GitHub:python/sampleONNXBert/README.md](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/python/sampleONNXBert/README.md)。
#### 3.2.2 ResNet Python Sample
同 [ResNet C++ Sample](#312-resnet-c-sample)，Python 代码仓库见 [GitHub:python/sampleONNXBert](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/python/sampleONNXResNet)。
#### 3.2.3 Yolov5 Python Sample
同 [Yolov5 C++ Sample](#313-yolov5-c-sample)，Python 代码仓库见 [GitHub:python/sampleONNXYolov5](https://github.com/EnflameTechnology/tops_inference_sample/tree/main/python/sampleONNXYolov5)。
