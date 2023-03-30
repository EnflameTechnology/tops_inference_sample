# 一、介绍
本目录下代码用于使能 topsinference2.0 进行 NLP_Bert 推理的 Python 代码。

## 1、依赖
```shell
python3 -m pip install -r requirement.txt
```

## 2、注意事项
bert 推理以及后处理与模型强相关。
因此，本例程将 input name、input shape 、output memory 的申请以及推理结果的后处理等硬编码在源代码中，
如果用户想要更换模型，需要用户手动修改上述部分的代码并重新运行。
当前，代码与 Bert-base-squad-op13-fp32-nvidia_seqN.onnx 强绑定，主要体现在：
* **input name 在代码中固定**
* **output memory 申请时使用的 shape 并不是从模型读取的**
* **后处理流程的对推理结果的读取与模型绑定，即从哪块buffer取到什么样的数据是先验的**