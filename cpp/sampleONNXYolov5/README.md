This is a sample of do inferenece on yolov5 model with the TopsInference2.0 API.

## Forward Dependence:
1. Make sure that the driver and software stack are set and available before running

2. Attention that this sample do not have any 3rd party library dependency to do image processing, only require that the input image is PPM format with dimention 640x640x3.

## Compile this sample
  Enter current directory, running `cmake` and `make` to compile.
  The binary 'sample_onnx_yolov5' will be found in the same directory.
  ```
  cmake .
  make
  ```

## Run the sample
1. Some configurable arguments can be set up in the following format
   ```
   ./yolov5s [--arg value]
   ```
usage: yolov5s [-h]
               [-o ONNX MODEL FILE PATH]
               [-i PPM IMAGE(640*640*3) FILE PATH]
               [-n NAME OF MODEL INPUT, DEFAULT IS: images]
               [-s INPUT SHAPE, DEFAULT IS: (1,3,640,640)]
               [-c SCORE THRESHOLD, DEFAULT IS: 0.25]
               [-u IOU THRESHOLD, DEFAULT IS: 0.45]

**note:** Please make sure that you have already saved the model to ./data. You can download the yolov5 model by [here](http://10.16.11.32/inference/Scorpio/yolo-v5/onnx/) .
| model   | download |
| -----   | -----    |
| yolov5s | [download](http://10.16.11.32/inference/Scorpio/yolo-v5/onnx/yolov5s.onnx)|
| yolov5m | [download](http://10.16.11.32/inference/Scorpio/yolo-v5/onnx/yolov5m.onnx)|
| yolov5n | [download](http://10.16.11.32/inference/Scorpio/yolo-v5/onnx/yolov5n.onnx)|
| yolov5l | [download](http://10.16.11.32/inference/Scorpio/yolo-v5/onnx/yolov5l.onnx)|
| yolov5x | [download](http://10.16.11.32/inference/Scorpio/yolo-v5/onnx/yolov5x.onnx)|

```
For one image:
./yolov5s -o ../../models/yolov5s.onnx  -i ../../data/yolov5/zidane640.ppm
```

2. Output result will be output on the console and image will be save in result.ppm in current directory.

PPM image could be viewed on  https://kylepaulsen.com/stuff/NetpbmViewer/
