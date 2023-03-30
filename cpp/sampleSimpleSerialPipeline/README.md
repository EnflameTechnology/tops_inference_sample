``` bash

## Install Topsinference & TopsSDK
dpkg -i <SDKPATH>/framework/tops-inference_<version>_<arch>_internal.deb 
dpkg -i <SDKPATH>/sdk/tops-sdk_<version>_<arch>_internal.deb 

#build with one command line:
g++  -I/usr/include/TopsInference -I/usr/include/dtu/libprofile -I/usr/include/dtu -Wall  -O3 -Werror -Wno-sign-compare -std=c++17 -L/usr/lib -L/usr/local/lib/ -lTopsInference -lpthread -ldl -Wl,-fuse-ld=gold -o ./sampleSimpleSerialPipeline sampleSimpleSerialPipeline.cpp
##or using cmake:
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