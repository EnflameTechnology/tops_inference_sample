This is a sample of do inferenece on resnet model with the TopsInference2.0 API.

## Forward Dependence:
1. Make sure that the driver and software stack are set and available before running

2. Attention that this sample do not have any 3rd party library dependency to do image processing, only require that the input image is processed data with dimention 1x3x224x224. You can use the python samples "sampleONNXResNet" to generate the processed image data by setting "save_processed_img" to True.

## Compile this sample
  Enter current directory, running `cmake` and `make` to compile.
  The binary 'resnet' will be found in the same directory.
  ```
  cmake .
  make
  ```

## Run the sample
1. Some configurable arguments can be set up in the following format
   ```
   ./resnet [--arg value]
   ```
usage: resnet [-h]
               [-o ONNX MODEL FILE PATH]
               [-i PROCESSED IMAGE(224*224*3) FILE PATH]
               [-n NAME OF MODEL INPUT, DEFAULT IS: input]
               [-s INPUT SHAPE, DEFAULT IS: (1,3,224,224)]

**note:** Please make sure that you have already saved the model to ./data. You can download the resnet model by [here](http://artifact.enflame.cn/artifactory/enflame_model_zoo/official/vision/classification/resnet50/resnet50_v1.5-torchvision-op13-fp32-N.onnx) .
| model   | download |
| -----   | -----    |
| resnet-v1.5 | [download](http://artifact.enflame.cn/artifactory/enflame_model_zoo/official/vision/classification/resnet50/resnet50_v1.5-torchvision-op13-fp32-N.onnx)|

```
For one image:
./resnet -o ../../models/resnet50_v1.5-torchvision-op13-fp32-N.onnx  -i ../../data/resnet/ILSVRC2012_val_00006740.JPEG.dat
```
