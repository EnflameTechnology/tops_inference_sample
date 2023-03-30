# TopsInference2.0 ResNet Python Sample

## Content
* [Structure](#Structure)
* [Introduction](#Introduction)
* [Feature](#Feature)
* [Run](#Run)

## Directory and Structure
### Structure
Code structured as below
```shell
├── imagenet_labels.json
├── README.md
├── requirements.txt
└── resnet.py
```
### Dependence
```shell
python3 -m pip install -r requirement.txt to install the dependencies.
```

**Note:** Please make sure that you have already saved the model to ./models. 

## Introduction
This is a sample of do inferenece on resnet model with the TopsInference2.0 API.
Available command options：
```
Generally, "data_path", "model" or "engine", are required setting by users while others with default value.
--device: `str`, on which device to run inference.
--card_id: `int`, on which card to run inference processing.
--cluster_ids: `int`, on which cluster to run inference. In one-cluster mode, only an integer range from 0 to 6 is available. You can just use the default in this case.
--data_path: path to dataset. Raw images are required for this sample as any preprocessing will be executed inside the scripts.
--model_path: `str`, Which model to run inference, should be resnet.
--engine: `str`, engine file produced by before run
--input_names: `str`, input tensor names, which must be consistent with the content of model file. When there are multi input, names are seperated by a comma
--input_shape: `str`, input tensor shapes, which must be consistent with the content of model file. When there are multi input, shapes are seperated by a colon.
--output_names: `str`, output tensor names, which must be consistent with the content of model file. When there are multi input, names are seperated by a comma
- save_processed_img: `bool`, save the processed numpy image data to the original image dirs, which can be used to resnet cpp example
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
        |        Deinitialize       |
        -----------------------------
```

### Attention
Input name is needed when create engine file, so need to fetch the model information in advance. Pass the input name with --input_names option.

## Run
``` shell
For single image:
python3 resnet.py --data_path=../../data/resnet/ILSVRC2012_val_00006740.JPEG --model=/PATH/TO/models/resnet50_v1.5-torchvision-op13-fp32-N.onnx

Using engine
python3 resnet.py --data_path=../../data/resnet/ILSVRC2012_val_00006740.JPEG --engine=/PATH/TO/engine/file

For a batch of images under one directory(default JPEG file):
python3 resnet.py --data_path=../../data/resnet/ --model=/PATH/TO/models/resnet50_v1.5-torchvision-op13-fp32-N.onnx
```
