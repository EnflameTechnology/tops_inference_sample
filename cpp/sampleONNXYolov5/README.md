# TopsInference2.0 Yolov5s C++ Sample 

## Content
* [Directory and Structure](#Directory and Structure)
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
└── yolov5.cpp
../utils                    --Utilization structures & functions
├── tops_stream_struct.h
├── tops_utils.cpp
├── tops_utils.h
```
### Dependence
**Note:** Please make sure that you have already saved the model to ./data.

## Introduction
This is a sample of do inferenece on yolov5 model with the TopsInference2.0 API.
Available command options：
```
-h help information
-o onnx model file path
-i PPM image(640*640*3) file path
-n name of model input, default is: images
-s input shape, default is: (1,3,640,640)
-c score threshold, default is: 0.25
-u iou threshold, default is: 0.45
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
        |     Input Preprocess      |
        -----------------------------
                      |
        -----------------------------
        |         Inference         |
        -----------------------------
                      |
        -----------------------------
        |       Output Process      |
        -----------------------------
                      |
        -----------------------------
        |        Draw and Save      |
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
The binary 'yolov5s' will be made in the same directory.

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
./yolov5s [--arg value]

For example, to inference on one image:
./yolov5s -o ../../models/yolov5s.onnx  -i ../../data/yolov5/zidane640.ppm
```

Detection result will be output on the console and image will be save in result.ppm in current directory.

PPM image could be viewed on  https://kylepaulsen.com/stuff/NetpbmViewer/
