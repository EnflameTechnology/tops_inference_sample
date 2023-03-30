#!/usr/bin/env python3
# =======================================================================
# Copyright 2020-2023 Enflame. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =======================================================================
#

import os
import time
import glob
import argparse
import numbers
from PIL import Image
import numpy as np
import json
import TopsInference


def img_crop(img, top, left, height, width):
    return img.crop((left, top, left + width, top + height))


def img_center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return img_crop(img, crop_top, crop_left, crop_height, crop_width)


def img_resize(img, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size, interpolation)


def model_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="dtu", help="Which device to run inference"
    )
    parser.add_argument(
        "--card_id",
        type=int,
        default=0,
        help="On which card to run inference processing",
    )
    parser.add_argument(
        "--cluster_ids",
        nargs="+",
        type=int,
        default=0,
        help="On which clusters to run inference. Generally now, \
              cluster_ids is a list with max length 6, it can be \
              [0], [0, 1], [0, 1, 2, 3, 4, 5] or any other range from 0 to 4. \
              cluster_ids can also be conveniently set to -1 to delegate \
              [0, 1, 2, 3, 4, 5]",
    )
    parser.add_argument(
        "--input_names",
        default="input",
        help="Input tensor names, which must be consistent with the content \
        of model file When there are multi input, names are seperated by \
        a comma.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="1, 3, 224, 224",
        help="The standrd input shape of onnx model",
    )
    parser.add_argument("--data_format", type=str, default="NCHW")
    parser.add_argument(
        "--model_path",
        type=str,
        default="yolov5s.onnx",
        help="Which model to run inference,\
                        should be one of yolov5s, yolov5m, yolov5n, yolov5l, \
                        yolov5x.",
    )
    parser.add_argument("--engine", type=str, help="engine file to be reused.")
    parser.add_argument(
        "--data_path", type=str, help="path to image file or directory(only jpeg now)"
    )
    parser.add_argument(
        "--save_processed_img",
        type=bool,
        default=False,
        help="save the processed numpy image data to the original image dirs",
    )
    parser.add_argument(
        "--output_names",
        default="output",
        help="Output tensor names, which must be consistent with the content \
        of model file When there are multi input, names are seperated by \
        a comma.",
    )
    arg_pars = parser.parse_args()
    return arg_pars


class Resnet:
    time_preprocess = None
    time_infer = None
    time_postprocess = None
    time_total = None

    def __init__(self, arguments, onnx_file, device="dtu"):
        self.device = device
        # dtu device
        self.handler = TopsInference.set_device(
            arguments.card_id, arguments.cluster_ids
        )
        # parse onnx
        parser = TopsInference.create_parser(TopsInference.ONNX_MODEL)
        parser.set_input_names(arguments.input_names)
        parser.set_output_names(arguments.output_names)
        parser.set_input_shapes(arguments.input_shape)
        parser.set_input_dtypes("DT_FLOAT32")

        if arguments.engine is None:
            network = parser.read(onnx_file)
            optimizer = TopsInference.create_optimizer()
            # optimizer.set_build_flag(TopsInference.KFP16)
            # optimizer.set_build_flag(TopsInference.KDEFAULT)
            optimizer.set_build_flag(TopsInference.KFP16_MIX)
            self.engine = optimizer.build(network)
            engine_path = os.path.join(".", os.path.basename(onnx_file) + ".exec")
            self.engine.save_executable(engine_path)
        else:
            self.engine = TopsInference.PyEngine()
            self.engine.load_tops_executable(arguments.engine)

        assert self.engine is not None

        self.input_width = int(arguments.input_shape.split(",")[-1])
        self.input_height = int(arguments.input_shape.split(",")[2])

        self.save_processed_img = arguments.save_processed_img

    def __call__(self, images, images_name):
        inputs = []
        outputs = []
        time_start = time.time()

        for (image, name) in zip(images, images_name):
            processed_img = self.preprocess(image, self.input_width, self.input_height)
            inputs.append(processed_img)
            if self.save_processed_img:
                processed_img.tofile(name + ".dat")

        time_after_preprocess = time.time()
        # Use sync mode, py_stream=None by default
        py_future = self.engine.run_with_batch(
            sample_nums=len(images),
            input_list=np.array([inputs]),
            buffer_type=TopsInference.TIF_ENGINE_RSC_IN_HOST_OUT_HOST,
        )
        pred = py_future.get()
        pred = np.asarray(pred)
        outputs.append(pred)
        time_after_infer = time.time()
        pred = np.argmax(outputs[0], axis=-1)

        time_finish = time.time()
        self.time_preprocess = (time_after_preprocess - time_start) * 1000
        self.time_infer = (time_after_infer - time_after_preprocess) * 1000
        self.time_postprocess = (time_finish - time_after_infer) * 1000
        self.time_total = (time_finish - time_start) * 1000
        return pred

    def preprocess(self, image, input_width, input_height):
        input_size = (input_width, input_height)
        max_size = max(input_width, input_height)

        image = img_resize(image, 256 if max_size <= 256 else 342)
        image = img_center_crop(image, input_size)

        image_data = np.array(image, dtype="float32").transpose(2, 0, 1)
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_image_data = np.zeros(image_data.shape).astype("float32")
        for i in range(image_data.shape[0]):
            norm_image_data[i, :, :] = (
                image_data[i, :, :] / 255 - mean_vec[i]
            ) / stddev_vec[i]
        norm_image_data = norm_image_data.reshape(3, input_height, input_width).astype(
            "float32"
        )

        return norm_image_data

    def destroy(self):
        TopsInference.release_device(self.handler)


def main(arguments):
    images = []
    images_name = []

    if os.path.isfile(arguments.data_path):
        image_data = Image.open(os.path.join(arguments.data_path)).convert("RGB")
        assert image_data.size[0] > 0, "no image read"
        images.append(image_data)
        images_name.append(os.path.join(arguments.data_path))

    # only jpg
    if os.path.isdir(arguments.data_path):
        jpg_list = glob.glob(arguments.data_path + "/*.JPEG")

        for image in jpg_list:
            image_data = Image.open(os.path.join(arguments.data_path, image)).convert(
                "RGB"
            )
            assert image_data.size[0] > 0, "no image read"
            images.append(image_data)
            images_name.append(os.path.join(arguments.data_path, image))

    with open("imagenet_labels.json", "r") as f:
        labels = json.load(f)

    model_file = arguments.model_path
    classification = Resnet(arguments, onnx_file=model_file, device=arguments.device)

    # warmup
    for i in range(3):
        results = classification(images, images_name)

    results = classification(images, images_name)
    print("{}{}results{}".format("\n" * 2, "*" * 40, "*" * 40))
    for ind, name in enumerate(images_name):
        print(
            "image name: {}, classification: {}".format(name, labels[results[0][ind]])
        )

    time_list = []
    time_list.append(
        [
            classification.time_preprocess,
            classification.time_infer,
            classification.time_postprocess,
            classification.time_total,
        ]
    )

    print("{}{}time cost{}".format("\n" * 2, "*" * 40, "*" * 40))
    print(
        "time_preprocess,{} time_infer,{} time_postprocess,{} time_total".format(
            " " * 3, " " * 10, " " * 5
        )
    )
    for t in time_list:
        print("{}".format(t))


if __name__ == "__main__":
    arguments = model_arguments()
    main(arguments)
