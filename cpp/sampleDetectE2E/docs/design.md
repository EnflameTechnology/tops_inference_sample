# 目的
此工程用于测试 S60 多路解码、CV、NN 的并发能力。此文档用于记录 pipeline 设计过程中的一些细节问题。

## 细节1 依赖关系
此工程使用 ffmpeg_gcu，该功能由 TopsVideo 提供。使用 gcu CV 操作，由 TopsCV 提供。NN 推理能力由 TopsInference 提供。

## 细节2 解码
解码部分由 ffmpeg_gcu 提供能力，按照设计，S60 单 die 可支持 128 路 1080P@30 解码。双 die 则 256 路。当前测试受限于 CPU，未完全测试 256路性能，但 context 的限制已经解除。

## 细节3 解码图像转移
实际的工程开发中，解码后的图像位于 Device 上的 RingBuffer 上。此存储空间位于解码单元内部，用户需要快速将其拷走。长期占用会导致解码器内部没有可用空间进行后续解码。

## 细节4 显存来源
每个模块实现了单独的显存池。

## 细节5 模块化
为了让设备都高效的工作，采用设备时分复用的方式：每路视频流起一个解码线程。解码后的数据在线程内部完成抽帧后送入拷贝模块。
整个流程完全异步。

## 细节6 数据传递
数据在模块间传递考虑使用线程安全的队列，前序模块的输出送入队列，后续模块从队列获取输入。

## 细节7 CV 模块、NN 模块
两个模块通过命令行参数设置工作线程的数量，工程线程抢占式的从前序的输出队列中获取数据作为自己的输入。

## 细节8 任务的开始与完成
- 1、解码任务：解码通过调用 av_recieve_frames 获得解码器当前的全部输出帧，同时经过解码器提供的后处理管线(全自动，完成 YUV2RGB + Resize + 抽帧)。
- 2、NN 任务：NN 模块的每个线程从输入队列获取到一个任务，先同步 event 保证该任务已经完成，随后将 NN 任务在与自己绑定的 stream 上同步完成。

## 细节9 decode online 后处理
decode 提供一条固定的在线后处理 pipeline，详见 [TopsVideo](http://docs.enflame.cn/sw/internal/5-program/topscv/FFmpeg_GCU/content/source/FFmpeg_GCU_user_guide.html) 的描述。

## 细节10 命令行参数设计
- dev 设备编号。
- video_config 描述每路视频地址的 json 文件。
- nr_nn_workers NN 模块的工作线程数量。
- model_path NN 模块使用的模型路径或者是 exec 路径。
- decode_only 只运行解码模块
