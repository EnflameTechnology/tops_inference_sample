# TopsInference2.0 Yolov5s Python Sample 

## Content
* [Directory and Structure](#Directory and Structure)
* [Introduction](#Introduction)
* [Feature](#Feature)
* [Run](#Run)

## Directory and Structure
### Structure
Code structured as below
```shell
├── coco_labels.py
├── README.md
├── requirements.txt
└── yolov5.py
```
### Dependence
```shell
python3 -m pip install -r requirement.txt to install the dependencies.
```

**Note:** Please make sure that you have already saved the model to ./data. 

## Introduction
This is a sample of do inferenece on yolov5 model with the TopsInference2.0 API.
Available command options：
```
Generally, "data_path", "result_dir", are required setting by users while others with default value.
--device: `str`, on which device to run inference.
--card_id: `int`, on which card to run inference processing.
--cluster_ids: `int`, on which cluster to run inference. In one-cluster mode, only an integer range from 0 to 4 is available. You can just use the default in this case.
--data_path: path to dataset. Raw images are required for this sample as any preprocessing will be executed inside the scripts.
--model_path: `str`, Which model to run inference, should be one of yolov5s, yolov5m, yolov5n, yolov5l, yolov5x.
--engine: `str`, engine file produced by before run
--input_names: `str`, input tensor names, which must be consistent with the content of model file. When there are multi input, names are seperated by a comma
--input_shape: `str`, input tensor shapes, which must be consistent with the content of model file. When there are multi input, shapes are seperated by a colon.
--output_names: `str`, output tensor names, which must be consistent with the content of model file. When there are multi input, names are seperated by a comma
--dispaly: `bool`, `True` for the display the result, `False` for save the result.
--result_dir: `str`, the directory of result images. 
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
Input name is needed when create engine file, so need to fetch the model information in advance. Pass the input name with --input_names option.

## Run
``` shell
For single image:
python3 yolov5.py --data_path=/PATH/TO/topsinference_samples/data/bus.jpg --model=/PATH/TO/topsinference_samples/models/yolov5s.onnx

Using engine
python3 yolov5.py --data_path=/PATH/TO/topsinference_samples/data/bus.jpg --model=/PATH/TO/topsinference_samples/models/yolov5s.onnx --engine=/PATH/TO/engine/file

For a batch of images under one directory:
python3 yolov5.py --data_path=/PATH/TO/topsinference_samples/data/yolov5 --model=/PATH/TO/topsinference_samples/models/yolov5s.onnx --result_dir ./data/result/
```

Detection result will be drawn on the output image and be saved in directory specified by --result_dir.
