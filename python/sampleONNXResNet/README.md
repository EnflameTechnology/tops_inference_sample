This is an end-to-end sample of Resnet.

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
- model_path: `str`, Which model to run inference, should be resnet.
- engine: `str`, engine file produced by before run
- input_names: `str`, input tensor names, which must be consistent with the content of model file. When there are multi input, names are seperated by a comma
- input_shape: `str`, input tensor shapes, which must be consistent with the content of model file. When there are multi input, shapes are seperated by a colon.
- save_processed_img: `bool`, save the processed numpy image data to the original image dirs
- output_names: `str`, output tensor names, which must be consistent with the content of model file. When there are multi input, names are seperated by a comma

2. Execute under current directory step by step

**note:** Please make sure that you have already saved the model to ./models. You can download the resnet model by [here](http://artifact.enflame.cn/artifactory/enflame_model_zoo/official/vision/classification/resnet50/resnet50_v1.5-torchvision-op13-fp32-N.onnx) .
| model   | download |
| -----   | -----    |
| resnet-v1.5 | [download](http://artifact.enflame.cn/artifactory/enflame_model_zoo/official/vision/classification/resnet50/resnet50_v1.5-torchvision-op13-fp32-N.onnx)|

**Example**

```
For single image:
python3 resnet.py --data_path=/PATH/TO/topsinference_samples/data/resnet/ILSVRC2012_val_00006740.jpg --model=/PATH/TO/topsinference_samples/models/resnet50_v1.5-torchvision-op13-fp32-N.onnx

Using engine
python3 resnet.py --data_path=/PATH/TO/topsinference_samples/data/resnet/ILSVRC2012_val_00006740.jpg --model=/PATH/TO/topsinference_samples/models/resnet50_v1.5-torchvision-op13-fp32-N.onnx --engine=/PATH/TO/engine/file

For a batch of images under one directory:
python3 resnet.py --data_path=/PATH/TO/topsinference_samples/data/resnet/ --model=/PATH/TO/topsinference_samples/models/resnet50_v1.5-torchvision-op13-fp32-N.onnx

```
