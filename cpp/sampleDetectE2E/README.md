## Description
This is one End2End sample of detection task, include decode media stream/file with TopsVideo API, image process with TopsCV API and do inferenece on yolov5m model with the TopsInference API.

This sample fully used topsvideo decode post online module so that decoder will return  resized and frame-extrated RGB24 picture.

Totally information for this pipeline and test result can get at [HERE](http://wiki.enflame.cn/pages/viewpage.action?pageId=191435920).

## Dependences:
Make sure all of the following are install

1. Enflame Driver
2. TopsRuntime
3. FFmpeg-GCU
4. TopsCV
5. TopsInference

## Compile this sample
  Enter current directory, running `cmake` and `make` to compile.
  The binary 'SampleE2E' will be found in the same directory.
  ```
  mkdir -p build && cd build
  cmake ..
  make
  ```

## Run the sample
1. Some configurable arguments can be set up in the following format
```Bash
   ./SampleE2E [--arg value]
   
  usage: yolov5s [-h]
               [-v video config path including video file/ stream address and some other related configurations]
               [-m model config path]
               [-i card id or saying device id]
               [-c number of threads doing cv operation]
               [-n number of threads doing inference]
```

**note:** Please make sure that you have already download models. You can download the yolov5 model by [here](http://10.16.11.32/inference/Scorpio/yolo-v5/onnx/) .
| model   | download |
| -----   | -----    |
| yolov5s | [download](http://10.16.11.32/inference/Scorpio/yolo-v5/onnx/yolov5s.onnx)|
| yolov5m | [download](http://10.16.11.32/inference/Scorpio/yolo-v5/onnx/yolov5m.onnx)|
| yolov5n | [download](http://10.16.11.32/inference/Scorpio/yolo-v5/onnx/yolov5n.onnx)|
| yolov5l | [download](http://10.16.11.32/inference/Scorpio/yolo-v5/onnx/yolov5l.onnx)|
| yolov5x | [download](http://10.16.11.32/inference/Scorpio/yolo-v5/onnx/yolov5x.onnx)|

2. Demo example

  Run on one media file:
```Bash
./sampleE2E -v /your/path/to/video_config -m /your/path/to/model_config -i 0 -c 2 -n 2
```

## Performance metrics

1. Record API/Transaction metrics in application to logs(stdout or log file) as below format
```json
	{ "message": "e2e_metric", "ops": "nn_done", "value": 20 }
```

2. Metric explaination

    | field   | description |
    | -----   | -----    |
    | message | Identify this is one metric record.|
    | ops | Metric name, e.g. decode_done, resize_done, nn_done etc.|
    | value | Metric value, e.g. time cost(in millisecond) for the API/Transaction.|

3. ELK setup

    3.1 Refert to https://github.com/joeshow79/docker-elk to setup the ELK servers for the metrics.

    3.2 Create one index named x2_e2e_metric on kibana GUI console.

    3.2 Create visulization and dashboard as you wish.

4. Run and collect mertrics

  ```Bash
    ./sampleE2E -v /your/path/to/video_config -m /your/path/to/model_config -i 0 -c 2 -n 2 | grep "message\": \"e2e_metric\"" |  nc -q0 [LOGSTASH_SERVER_IP] 50000      

    or 

    ./sampleE2E -v /your/path/to/video_config -m /your/path/to/model_config -i 0 -c 2 -n 2 (let say logs saved to file named e2e.log)

    tail -f e2e.log  | grep "message\": \"e2e_metric\"" |  nc -q0 [LOGSTASH_SERVER_IP] 50000      
  ```
