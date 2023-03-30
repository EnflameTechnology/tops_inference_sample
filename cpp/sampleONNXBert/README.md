# 一、介绍
本目录下代码用于使能 topsinference2.0 进行 NLP_Bert 推理的 C++ 代码。Sample 使用 CMake 构建，步骤如下：

## 1、创建 build 目录
```shell
mkdir build && cd build
```

## 2、执行 cmake 命令
```shell
cmake ..
```

## 3、编译
```shell
make
```

## 4、执行 sample
编译之后，在 build 目录下会生成 SampleBert 的可执行程序。其命令行参数可以通过 -h/--help 查看

## 5、注意事项
topsinference 的执行需要在读入模型时传入模型的 input name 和 input shape，这就需要用户事先了解模型。
因此，本例程将 input name、input shape 、output memory 的申请以及推理结果的后处理等硬编码在源代码中，
如果用户想要更换模型，需要用户手动修改上述部分的代码并重新编译。
当前，代码与 Bert-base-squad-op13-fp32-nvidia_seqN.onnx 强绑定，主要体现在：
* **input name 在代码中固定**
* **output memory 申请时使用的 shape 并不是从模型读取的**
* **后处理流程的对推理结果的读取与模型绑定，即从哪块buffer取到什么样的数据是先验的**
测试数据和词表已同步在 input 和 vocab 文件夹下


# 二、Sample 目录结构以及依赖

## 1、目录结构

```shell
├── utils           --utils third party source code
├── sampleBert.h    --head file
├── sampleBert.cpp  --source file
├── main.cpp        --main source file
└── README.md    
```

## 2、依赖
Bert 推理涉及到文本文件的 token 处理，因此需要的三方库如下：
* **boost**
