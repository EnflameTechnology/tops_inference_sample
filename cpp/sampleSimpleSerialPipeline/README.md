## Table of Content
- [Table of Content](#table-of-content)
- [Directory Structure](#directory-structure)
- [Introduction](#introduction)
- [Build and Run](#build-and-run)
- [Note](#note)


## Directory Structure
``` bash
|-- CMakeLists.txt # The CMake file.
|-- README.md # The readme file..
|-- sampleSimpleSerialPipeline.cpp # The main code.
```
## Introduction
The main purpose of this example is to demonstrate how to perform inference using two models in GCU. The two models run sequentially, and you can refer to another example (sampleParallelPipeline) for parallel execution.

To run this example, you need to download two models, `resnet50` and `yolov5s`, from the `modelzoo` of GCU.

The operation of this example is straightforward. It reads the ONNX format models, parses them, and compiles them into executable binary files using GCU's graph compilation engine. Then it reads `input.data` for inference.

Specifically, the `yolov5s` model runs first, obtaining the coordinates of some target bounding boxes in the image. These bounding box regions are then cropped and resized to the input size of the `resnet50` model. Finally, the `resnet50` model performs inference to obtain the specific class of target at the bounding box location.


## Build and Run
You can build this example using cmake:
``` bash

mkdir build && cd build && cmake ../

```

Alternatively, you can just use a long compiling command such as:
``` bash

g++  -I/usr/include/TopsInference  -Wall  -O3 -Werror -Wno-sign-compare -std=c++17 -L/usr/lib -L/usr/local/lib/ -lTopsInference -lpthread -ldl -Wl,-fuse-ld=gold -o ./sampleSimpleSerialPipeline sampleSimpleSerialPipeline.cpp

```

The complete example runs as follows:
``` bash

## Install Topsinference && TopsSDK (If you already installed, just jump this step)
dpkg -i <SDKPATH>/framework/tops-inference_<version>_<arch>_internal.deb 
dpkg -i <SDKPATH>/sdk/tops-sdk_<version>_<arch>_internal.deb 

##build:
mkdir build && cd build && cmake ../

# Usage: sampleSimpleSerialPipeline [options...]
# Options:
#     --vg                   number of VG (In this dempstration, two models use the same vg)
#     --det_buffersize       buffer size (detection model) 
#     --det_modelpath        onnx model path (detection model) 
#     --det_inputname        onnx model input name (detection model) 
#     --det_shape            input shape (detection model) 
#     --cls_buffersize       buffer size (classification model) 
#     --cls_modelpath        onnx model path (classification model) 
#     --cls_inputname        onnx model input name (classification model) 
#     --cls_shape            input shape (classification model) 
#     --imagepath            input image path       
#     --image_shape          input image shape      
#     --loop                 number of running loop 
#     -h, --help             Shows this page         (Optional)

#gen input.data
python3.6 gendata.py

#test
./sampleSimpleSerialPipeline \
--vg=6 \
--det_buffersize=32 \
--det_modelpath=./yolov5s-v6.0-640-op13-fp32-N.onnx \
--det_inputname=images \
--det_shape=1,3,640,640 \
--cls_buffersize=512 \
--cls_modelpath=./resnet50-v1.5-op13-fp32-N.onnx \
--cls_inputname=input \
--cls_shape=1,3,224,224 \
--imagepath=./input.data \
--loop=10000 \
--image_shape=1,3,1080,810


#will printing like this：
# ...
# >>>>>>>>>img: 9997   box : [226.428, 401.625, 343.103, 860.625, 0.981445, 0], 148
# >>>>>>>>>img: 9997   box : [-0.0395508, 558.035, 66.6431, 872.965, 0.980957, 0], 881
# >>>>>>>>>img: 9997   box : [3.16406, 211.992, 798.609, 790.383, 0.94043, 5], 575
# >>>>>>>>>img: 9997   box : [660.073, 625.403, 690.349, 715.315, 0.808594, 41], 881
# >>>>>>>>>img: 9998   box : [56.7158, 404.578, 219.349, 904.078, 0.996582, 0], 796
# >>>>>>>>>img: 9998   box : [670.663, 405.633, 808.853, 884.461, 0.984863, 0], 792
# >>>>>>>>>img: 9998   box : [226.428, 401.625, 343.103, 860.625, 0.981445, 0], 148
# >>>>>>>>>img: 9998   box : [-0.0395508, 558.035, 66.6431, 872.965, 0.980957, 0], 881
# >>>>>>>>>img: 9998   box : [3.16406, 211.992, 798.609, 790.383, 0.94043, 5], 575
# >>>>>>>>>img: 9998   box : [660.073, 625.403, 690.349, 715.315, 0.808594, 41], 881
# >>>>>>>>>img: 9999   box : [56.7158, 404.578, 219.349, 904.078, 0.996582, 0], 796
# >>>>>>>>>img: 9999   box : [670.663, 405.633, 808.853, 884.461, 0.984863, 0], 792
# >>>>>>>>>img: 9999   box : [226.428, 401.625, 343.103, 860.625, 0.981445, 0], 148
# >>>>>>>>>img: 9999   box : [-0.0395508, 558.035, 66.6431, 872.965, 0.980957, 0], 881
# >>>>>>>>>img: 9999   box : [3.16406, 211.992, 798.609, 790.383, 0.94043, 5], 575
# >>>>>>>>>img: 9999   box : [660.073, 625.403, 690.349, 715.315, 0.808594, 41], 881
# 2022-08-25 03:25:10.749203: D [T 18858] /home/enflame_sse_ci/jenkins/workspace/JF_sw_daily_cmake_build@2/tops/sdk/lib/TopsInference/Engine/TopsInferenceEngineImpl.h:66 : engine -- { model_id: 1 }, was Free
# 2022-08-25 03:25:10.856771: D [T 18858] /home/enflame_sse_ci/jenkins/workspace/JF_sw_daily_cmake_build@2/tops/sdk/lib/TopsInference/Engine/TopsInferenceEngineImpl.h:66 : engine -- { model_id: 2 }, was Free
# [INFO] running time: 93 seconds
# DONE
```

## Note

The input `shape` parameter, such as `--det_shape`, must meet the definition of the model file `--cls_modelpath`. In addition, the `buffersize` parameter ensures that it does not exceed the memory limit of GCU. Please refer to the relevant GCU documentation for specific values.