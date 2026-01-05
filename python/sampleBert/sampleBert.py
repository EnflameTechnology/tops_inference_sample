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
    input_ids, input_mask, segment_ids, extra_data = squad.convert_examples_to_features(
        eval_examples, vocab_file, max_seq_length, doc_stride, max_query_length
    )

    return eval_examples, input_ids, input_mask, segment_ids, extra_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file",
                        type=str,
                        default="bert_base-squad-nvidia-op13-fp32-N.onnx")
    parser.add_argument("--input_file",
                        type=str,
                        help="The input file. It should be in SQuAD json format.")
    parser.add_argument("--output_file",
                        type=str,
                        help="The output file. It should be in json format.")
    parser.add_argument("--vocab_file",
                        type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--engine",
                        type=str,
                        help="engine file to be reused")
    parser.add_argument("--precision",
                        type=str,
                        default="mix")
    parser.add_argument("--card_id",
                        type=int,
                        default=0,
                        help="Which card to run inference on."
    )
    parser.add_argument("--cluster_id",
                        type=int,
                        default=0,
                        help="On which clusters to run inference. Generally now, \
                              cluster_ids is a list with max length 6 for i20 or max length 2 for s60, \
                              it can be [0], [0, 1], [0, 1, 2, 3, 4, 5] or any other range from 0 to 4. \
                              cluster_ids can also be conveniently set to -1 to delegate \
                              [0, 1, 2, 3, 4, 5]"
    )
    parser.add_argument("--batch_size",
                        type=int,
                        default=1)
    parser.add_argument("--max_seq_len",
                        type=int,
                        default=384)
    parser.add_argument("--max_query_len",
                        type=int,
                        default=128)
    parser.add_argument("--doc_stride",
                        type=int,
                        default=128)
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

        if args.engine is not None:
            engine_file = args.engine 
            engine = TopsInference.load(engine_file)
        else:
            assert args.model_file is not None, "model file is required"
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

            engine_file = os.path.join(".", os.path.basename(args.model_file) + ".engine")
            engine.save_executable(engine_file)

        running_bs = len(input_ids)
        inputs = [segment_ids, input_mask, input_ids]
        outputs = []

        ## run inference with run_with_batch
        # engine.run_with_batch(
        #     running_bs,
        #     inputs,
        #     output_list=outputs,
        #     buffer_type=TopsInference.TIF_ENGINE_RSC_IN_HOST_OUT_HOST,
        # )

        ## run inference with runV2
        stream = TopsInference.create_stream()
        py_future = engine.runV2(inputs, py_stream=stream)

        outputs = py_future.get()

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
