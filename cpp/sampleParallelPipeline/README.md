``` bash
## Install Topsinference & TopsSDK
dpkg -i <SDKPATH>/framework/tops-inference_<version>_<arch>_internal.deb 
dpkg -i <SDKPATH>/sdk/tops-sdk_<version>_<arch>_internal.deb 

#build with one command line:
g++  -I/usr/include/TopsInference -I/usr/include/dtu/libprofile -I/usr/include/dtu -Wall  -O3 -Werror -Wno-sign-compare -std=c++17 -L/usr/lib -L/usr/local/lib/ -lTopsInference -lpthread -ldl -Wl,-fuse-ld=gold -o ./sampleParallelPipeline sampleParallelPipeline.cpp
#or using cmake:
mkdir build && cd build && cmake ../

# Usage: sampleParallelPipeline [options...]
# Options:
#     --det_vg               number of VG (detection model) 
#     --det_buffersize       buffer size (detection model) 
#     --det_nstream          number of streams (detection model) 
#     --det_modelpath        onnx model path (detection model) 
#     --det_inputname        onnx model input name (detection model) 
#     --det_shape            input shape (detection model) 
#     --cls_vg               number of VG (classification model) 
#     --cls_buffersize       buffer size (classification model) 
#     --cls_nstream          number of streams (classification model) 
#     --cls_modelpath        onnx model path (classification model) 
#     --cls_inputname        onnx model input name (classification model) 
#     --cls_shape            input shape (classification model) 
#     --imagepath            input image path       
#     --image_shape          input image shape      
#     --loop                 number of running loop 
#     -h, --help             Shows this page         (Optional)

#gen input.data
python3.6 gendata.py

#test:
./sampleParallelPipeline \
--det_vg=2 \
--det_buffersize=32 \
--det_nstream=1 \
--det_modelpath=./yolov5s-v6.0-640-op13-fp32-N.onnx \
--det_inputname=images \
--det_shape=1,3,640,640 \
--cls_vg=4 \
--cls_buffersize=512 \
--cls_nstream=1 \
--cls_modelpath=./resnet50-v1.5-op13-fp32-N.onnx \
--cls_inputname=input \
--cls_shape=1,3,224,224 \
--imagepath=./input.data \
--loop=10000 \
--image_shape=1,3,1080,810

#will printing like this：
# ....
# >>>>>>>>>img: 9997   box : [3.11451, 211.71, 798.957, 790.45, 0.940344, 5], 575
# >>>>>>>>>img: 9997   box : [659.922, 625.505, 690.233, 715.509, 0.807789, 41], 881
# >>>>>>>>>img: 9998   box : [56.6898, 404.448, 219.401, 904.455, 0.996933, 0], 796
# >>>>>>>>>img: 9998   box : [670.511, 405.385, 808.846, 884.635, 0.985094, 0], 792
# >>>>>>>>>img: 9998   box : [226.464, 401.69, 343.154, 860.978, 0.981963, 0], 148
# >>>>>>>>>img: 9998   box : [-0.0180397, 557.767, 66.6281, 872.934, 0.981138, 0], 881
# >>>>>>>>>img: 9998   box : [3.11451, 211.71, 798.957, 790.45, 0.940344, 5], 575
# >>>>>>>>>img: 9998   box : [659.922, 625.505, 690.233, 715.509, 0.807789, 41], 881
# >>>>>>>>>img: 9999   box : [56.6898, 404.448, 219.401, 904.455, 0.996933, 0], 796
# >>>>>>>>>img: 9999   box : [670.511, 405.385, 808.846, 884.635, 0.985094, 0], 792
# >>>>>>>>>img: 9999   box : [226.464, 401.69, 343.154, 860.978, 0.981963, 0], 148
# >>>>>>>>>img: 9999   box : [-0.0180397, 557.767, 66.6281, 872.934, 0.981138, 0], 881
# >>>>>>>>>img: 9999   box : [3.11451, 211.71, 798.957, 790.45, 0.940344, 5], 575
# >>>>>>>>>img: 9999   box : [659.922, 625.505, 690.233, 715.509, 0.807789, 41], 88
# [INFO] running time: 51 seconds

```