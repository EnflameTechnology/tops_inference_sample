This is an end-to-end sample of Yolov5.

## Forward Dependence:
1. Make sure that the TopsRider driver and software stack are set and available before running

2. Install required library with
```
python3 -m pip install -r requirement.txt
```
## Run this sample
1. Generally, "data_path", "result_dir", are required setting by users while others with default value.
- device: `str`, on which device to run inference.
- card_id: `int`, on which card to run inference processing.
- cluster_ids: `int`, on which cluster to run inference. In one-cluster mode, only an integer range from 0 to 4 is available. You can just use the default in this case.
- data_path: path to dataset. Raw images are required for this sample as any preprocessing will be executed inside the scripts.
- model_path: `str`, Which model to run inference, should be one of yolov5s, yolov5m, yolov5n, yolov5l, yolov5x.
- engine: `str`, engine file produced by before run
- input_names: `str`, input tensor names, which must be consistent with the content of model file. When there are multi input, names are seperated by a comma
- input_shape: `str`, input tensor shapes, which must be consistent with the content of model file. When there are multi input, shapes are seperated by a colon.
- output_names: `str`, output tensor names, which must be consistent with the content of model file. When there are multi input, names are seperated by a comma
- dispaly: `bool`, `True` for the display the result, `False` for save the result.
- result_dir: `str`, the directory of result images. 

2. Execute under current directory step by step

**note:** Please make sure that you have already saved the model to ./data. You can download the yolov5 model by [here](http://10.16.11.32/inference/Scorpio/yolo-v5/onnx/) .
| model   | download |
| -----   | -----    |
| yolov5s | [download](http://10.16.11.32/inference/Scorpio/yolo-v5/onnx/yolov5s.onnx)|
| yolov5m | [download](http://10.16.11.32/inference/Scorpio/yolo-v5/onnx/yolov5m.onnx)|
| yolov5n | [download](http://10.16.11.32/inference/Scorpio/yolo-v5/onnx/yolov5n.onnx)|
| yolov5l | [download](http://10.16.11.32/inference/Scorpio/yolo-v5/onnx/yolov5l.onnx)|
| yolov5x | [download](http://10.16.11.32/inference/Scorpio/yolo-v5/onnx/yolov5x.onnx)|

**Example**

```
For single image:
python3 yolov5.py --data_path=/PATH/TO/topsinference_samples/data/bus.jpg --model=/PATH/TO/topsinference_samples/models/yolov5s.onnx

Using engine
python3 yolov5.py --data_path=/PATH/TO/topsinference_samples/data/bus.jpg --model=/PATH/TO/topsinference_samples/models/yolov5s.onnx --engine=/PATH/TO/engine/file

For a batch of images under one directory:
python3 yolov5.py --data_path=/PATH/TO/topsinference_samples/data/yolov5 --model=/PATH/TO/topsinference_samples/models/yolov5s.onnx --result_dir ./data/result/

```
