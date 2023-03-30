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

os.environ["ENFLAME_LOG_LEVEL"] = "FATAL"  # DEBUG, ERROR, INFO, FATAL
os.environ["ENFLAME_LOG_DEBUG_MOD"] = ""  # OP/V2/TOIR/ONNX/LOWER/PARSER
os.environ["SDK_LOG_LEVEL"] = "3"

import argparse
import json
import squad
import TopsInference


def pre_process(input_file, vocab_file, max_seq_length, doc_stride, max_query_length):
    # Use read_squad_examples method from run_onnx_squad to read the input file
    eval_examples = squad.read_squad_examples(input_file=input_file)

    # Use convert_examples_to_features method from run_onnx_squad to get parameters from the input
    (
        input_ids,
        input_mask,
        segment_ids,
        extra_data,
    ) = squad.convert_examples_to_features(
        eval_examples, vocab_file, max_seq_length, doc_stride, max_query_length
    )

    return eval_examples, input_ids, input_mask, segment_ids, extra_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--vocab_file", type=str)
    parser.add_argument("--save_engine", type=str)
    parser.add_argument("--precision", type=str, default="mix")
    parser.add_argument("--card_id", type=int, default=0)
    parser.add_argument("--cluster_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--max_query_len", type=int, default=128)
    parser.add_argument("--doc_stride", type=int, default=128)
    args = parser.parse_args()

    examples, input_ids, input_mask, segment_ids, extra_data = pre_process(
        args.input_file,
        args.vocab_file,
        args.max_seq_len,
        args.doc_stride,
        args.max_query_len,
    )

    if args.precision == "default":
        precision_mode = TopsInference.KDEFAULT
    elif args.precision == "fp16":
        precision_mode = TopsInference.KFP16
    elif args.precision == "mix":
        precision_mode = TopsInference.KFP16_MIX
    else:
        assert False, "unknown percision mode: {}".format(args.precision)

    with TopsInference.device(args.card_id, args.cluster_id):
        engine = TopsInference.PyEngine()

        try:
            engine.load_tops_executable(args.model_file)
        except:
            onnx_parser = TopsInference.create_parser(TopsInference.ONNX_MODEL)
            input_names = "segment_ids:0,input_mask:0,input_ids:0"
            onnx_parser.set_input_names(input_names)
            one_input = str(args.batch_size) + "," + str(args.max_seq_len)
            input_shapes = one_input + ":" + one_input + ":" + one_input
            onnx_parser.set_input_shapes(input_shapes)
            module = onnx_parser.read(args.model_file)
            optimizer = TopsInference.create_optimizer()
            optimizer.set_build_flag(precision_mode)
            engine = optimizer.build(module)

            if args.save_engine is not None:
                engine.save_executable(args.save_engine)

        running_bs = len(input_ids)
        inputs = [segment_ids, input_mask, input_ids]
        outputs = []

        engine.run_with_batch(
            running_bs,
            inputs,
            output_list=outputs,
            buffer_type=TopsInference.TIF_ENGINE_RSC_IN_HOST_OUT_HOST,
        )

        start_logits = outputs[1]
        end_logits = outputs[0]

        # postprocessing
        json_file = squad.generate_json(
            examples, extra_data, start_logits, end_logits, True
        )
        print(json_file)
        if args.output_file is not None:
            with open(args.output_file, "w") as output_prediction_file:
                output_prediction_file.write(json.dumps(json_file, indent=2) + "\n")
        else:
            print(json.dumps(json_file, indent=2))
