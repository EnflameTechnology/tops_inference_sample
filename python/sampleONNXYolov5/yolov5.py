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

import time
import argparse
import os
import glob
import cv2
import numpy as np
import TopsInference
import coco_labels


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
        default="images",
        help="Input tensor names, which must be consistent with the content \
        of model file When there are multi input, names are seperated by \
        a comma.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="1, 3, 640, 640",
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
        "--output_names",
        default="output",
        help="Output tensor names, which must be consistent with the content \
        of model file When there are multi input, names are seperated by \
        a comma.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--iou_threshold", type=float, default=0.45, help="The IOU threshold during NMS"
    )
    parser.add_argument(
        "--display", action="store_true", help="Display or save the result."
    )
    parser.add_argument("--result_dir", type=str, default="./")

    arg_pars = parser.parse_args()
    return arg_pars


class Yolov5:
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

        self.normH = int(arguments.input_shape.split(",")[2])
        self.normW = int(arguments.input_shape.split(",")[-1])
        self.img_scale = None
        # thresholds
        self.score_th = arguments.score_threshold
        self.nms_th = arguments.iou_threshold

    def __call__(self, images):
        inputs = []
        outputs = []

        time_start = time.time()

        for image in images:
            processed_img = self.preprocess(image)
            processed_img = np.around(processed_img, decimals=5)
            inputs.append(processed_img)

        time_after_preprocess = time.time()

        # Use sync mode, py_stream=None by default
        py_future = self.engine.run_with_batch(
            sample_nums=len(images),
            input_list=np.array([inputs]),
            buffer_type=TopsInference.TIF_ENGINE_RSC_IN_HOST_OUT_HOST,
        )

        time_after_infer = time.time()

        pred = py_future.get()
        pred = np.asarray(pred)
        outputs.append(pred)
        outputs = outputs[0][0]  # 1,1,25200,85

        # postprocessing
        results = []
        for index, image in enumerate(images):
            image_shape = image.shape
            result = self.postprocess(outputs[index], image_shape[1], image_shape[0])
            results.append(result)

        time_finish = time.time()
        self.time_preprocess = (time_after_preprocess - time_start) * 1000
        self.time_infer = (time_after_infer - time_after_preprocess) * 1000
        self.time_postprocess = (time_finish - time_after_infer) * 1000
        self.time_total = (time_finish - time_start) * 1000
        return results

    def destroy(self):
        TopsInference.release_device(self.handler)

    def nms(self, boxes, scores):
        if scores.shape[0] < 1 or boxes.shape[0] != scores.shape[0]:
            return None
        order = scores.argsort()[::-1]
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (y2 - y1 + 1) * (x2 - x1 + 1)
        boxes_left = []
        while len(order) > 0:
            i = order[0]
            boxes_left.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (area[i] + area[order[1:]] - inter)
            ids = np.where(iou <= self.nms_th)[0]
            order = order[ids + 1]
        return boxes_left

    def preprocess(self, img):
        assert isinstance(
            img, np.ndarray
        ), "input image data type should be numpy.ndarray"
        assert img.shape[0] > 0 and img.shape[1] > 0, "input image shape error"

        shape = img.shape[0:2]
        self.im_scale = min(self.normH / img.shape[0], self.normW / img.shape[1])
        new_imsize = int(round(shape[1] * self.im_scale)), int(
            round(shape[0] * self.im_scale)
        )  # w, h
        dw, dh = self.normW - new_imsize[0], self.normH - new_imsize[1]  # wh padding
        dw /= 2
        dh /= 2
        if shape[::-1] != new_imsize:
            img = cv2.resize(img, new_imsize, interpolation=cv2.INTER_LINEAR).astype(
                np.float32
            )
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img).astype(np.float32)
        img /= 255.0
        if len(img.shape) == 3:
            img = img[None]
        return img

    def postprocess(self, buffer, img_width, img_height):
        cands = buffer[..., 4] > self.score_th
        output = []
        scale = min(self.normW / img_width, self.normH / img_height)
        pad = (self.normW - img_width * scale) / 2, (
            self.normH - img_height * scale
        ) / 2
        # for x in buffer:
        x = buffer[cands]
        if not x.shape[0]:
            return []
        x[:, 5:] *= x[:, 4:5]
        boxes = xywh2xyxy(x[:, :4])
        conf = x[:, 4]
        cls = np.argmax(x[:, 5:], axis=1)
        box_num = boxes.shape[0]
        x = np.concatenate(
            (boxes, np.reshape(conf, (box_num, -1)), np.reshape(cls, (box_num, -1))),
            axis=1,
        )
        if x.shape[0] == 1:
            output.append(x)
        else:
            selected_boxes = self.nms(x[:, :4], x[:, 4])
            for box_id in selected_boxes:
                output.append(x[box_id])
        for i, det in enumerate(output):
            if min(det.shape) != 0:
                det[[0, 2]] -= pad[0]
                det[[1, 3]] -= pad[1]
                det[:4] /= scale
                det[[0, 2]].clip(0, img_height)
                det[[1, 3]].clip(0, img_width)
                # det[2].clamp_(0, img_height)
                # det[3].clamp_(0, img_width)
        return output


def display_results(image, results):
    display = image.copy()
    for box in results:
        x1, y1, x2, y2, cls = box[0], box[1], box[2], box[3], box[5]
        conf = str(np.around(box[4].item() * 100, decimals=2)) + "%"

        cv2.rectangle(
            display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=2
        )
        label = coco_labels.COCOLabels(int(cls)).name + ":" + conf
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        y1 = max(y1, label_size[1])
        cv2.putText(
            display,
            label,
            (int(x1), int(y1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            thickness=2,
        )
    return display


def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def main(arguments):
    images = []

    if os.path.isfile(arguments.data_path):
        image = cv2.imread(arguments.data_path)
        assert len(image.shape) > 0, "no image read"

        images.append(image)

    # Only jpg
    if os.path.isdir(arguments.data_path):
        jpg_list = glob.glob(arguments.data_path + "/*.jpg")

        for image in jpg_list:
            image = cv2.imread(os.path.join(arguments.data_path, image))
            assert len(image.shape) > 0, "no image read"

            images.append(image)

    model_file = arguments.model_path
    detector = Yolov5(arguments, onnx_file=model_file, device=arguments.device)

    # warmup
    for i in range(0, 3):
        results = detector(images)

    results = detector(images)

    time_list = []
    time_list.append(
        [
            detector.time_preprocess,
            detector.time_infer,
            detector.time_postprocess,
            detector.time_total,
        ]
    )

    # display or save
    for index, image in enumerate(images):
        if len(results) != 0:
            display = display_results(image, results[index])
        else:
            display = image

        img_name = str(index) + ".jpg"

        if not os.path.exists(arguments.result_dir):
            os.mkdir(arguments.result_dir)

        if isinstance(display, np.ndarray):
            if not arguments.display:
                cv2.imwrite(os.path.join(arguments.result_dir, img_name), display)
            else:
                cv2.imshow("yolov5 results", display)
                cv2.waitKey(0)

    print("{}{}time cost{}".format("\n" * 10, "*" * 40, "*" * 40))
    print(
        "time_preprocess,{} time_infer,{} time_postprocess,{} time_total".format(
            " " * 3, " " * 10, " " * 5
        )
    )
    for t in time_list:
        print("{}".format(t))
    print("the result has been saved to ", arguments.result_dir)


if __name__ == "__main__":
    arguments = model_arguments()
    main(arguments)
