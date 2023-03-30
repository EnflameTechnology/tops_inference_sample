# TopsInference2.0 ResNet C++ Sample

## Content
* [Structure](#Structure)
* [Introduction](#Introduction)
* [Feature](#Feature)
* [Setup](#Setup)
* [Run](#Run)

## Directory and Structure
### Structure
Code structured as below
```shell
├── CMakeLists.txt
├── README.md
└── resnet.cpp
../utils                    --Utilization structures & functions
├── tops_stream_struct.h
├── tops_utils.cpp
├── tops_utils.h
```
### Dependence
**Note:** Please make sure that you have already saved the model to ./models.

## Introduction
This is a sample of do inferenece on resnet model with the TopsInference2.0 API.
Available command options：
```
-h help information
-o onnx model file path
-i processed image(224*224*3) file path
-n name of model input, default is: images
-s input shape, default is: (1,3,224,224)
```

## Feature
Pipeline as below：
```
        -----------------------------
        |        Initialize         |
        -----------------------------
                      |
        -----------------------------
        |      Load ONNX Model      |
        -----------------------------
                      |
    -------------------------------------
    |  Create/Load TopsInference Engine  |
    -------------------------------------
                      |
        -----------------------------
        |      Read Data File       |
        -----------------------------
                      |
        -----------------------------
        |         Inference         |
        -----------------------------
                      |
        -----------------------------
        |        Deinitialize       |
        -----------------------------
```

### Attention
Input name is needed when create engine file, so need to fetch the model information in advance. Pass the input name with -n option.

## Setup
Create and enter build directory, running `cmake` and `make` to compile.
The binary 'resnet' will be made in the same directory.

### create build directory
```shell
mkdir build && cd build
```

### cmake
```shell
cmake ..
```

### Compile
```shell
make
```

## Run
``` shell
./resnet [--arg value]

For example, to inference on one image:
./resnet -o ../../../models/resnet50_v1.5-torchvision-op13-fp32-N.onnx  -i ../../../data/resnet/ILSVRC2012_val_00006740.JPEG.dat
```
