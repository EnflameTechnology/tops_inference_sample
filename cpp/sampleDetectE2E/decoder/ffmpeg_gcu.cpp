#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>

#include "tops/tops_runtime_api.h"

#include "ffmpeg_gcu.h"

#include "utils/image_utils.h"
#include "utils/timer.h"

#define AV_LOG(_ret, _ctx, _level, _func) \
    do {\
        char error_info[64];\
        memset(error_info, 0, 64);\
        av_strerror(_ret, error_info, 64);\
        av_log(_ctx, _level, "%s : got %s(%d)\n", _func, error_info, _ret);\
    } while(0)

#define GCU_LOG(_ctx, _level, _fmt, ...) \
    do {\
        av_log(_ctx, _level, _fmt, ##__VA_ARGS__);\
    } while(0)

#define ENFLAME_E2E_CHECK(_ret, _func) \
    do {\
        if (_ret != E2E_Status::SUCCESS) {\
            fprintf(stderr, "%s Fail.\n", _func);\
        }\
    } while(0)

#define ENFLAME_RUNTIME_CHECK(_expr)                                           \
  do {                                                                         \
    if (topsError_t::topsSuccess != _expr) {                                   \
      fprintf(stderr, "EnFlame Runtime ERROR : %s @ %s:%d\n", topsGetErrorString(_expr), __FILE__, __LINE__);\
      raise(SIGUSR1);                                                                \
    }                                                                          \
  } while (0)

namespace {
using ImageFormat = e2e_sample::ImageFormat;
ImageFormat av_format_2_e2e_format(AVPixelFormat av_format) {
    switch (av_format) {
        case AV_PIX_FMT_RGB24 : return ImageFormat::RGB24;
        case AV_PIX_FMT_BGR24 : return ImageFormat::BGR24;
        default : return ImageFormat::UNKNOW;
    }
}
}

namespace e2e_sample {

//! construct
FFmpegGCU::FFmpegGCU() {
    av_log_set_level(AV_LOG_ERROR);
    // av_log_set_level(AV_LOG_FATAL);
    m_decoded_images = std::make_unique<ThreadSafeQueue<std::shared_ptr<E2EImage>>>("decoded_images");
    m_images_pool = std::make_unique<ImageBufferPool>();
}

FFmpegGCU::~FFmpegGCU() {
    {
        std::unique_lock<std::mutex> lck(m_streams_mutex);
        for (auto&& stream : m_streams) {
            stream.second->is_working = false;
            if (stream.second->decode_thread.joinable()) {
                stream.second->decode_thread.join();
            }
            clear_av_resource(stream.second);
            clear_rt_resource(stream.second);
        }
    }
    m_decoded_images.reset();
    m_images_pool.reset();
}

void FFmpegGCU::work_func(StreamInfo* pstream) {
    AVPacket packet;
    AVFrame *frame = av_frame_alloc();
    if (!frame) {
        av_log(nullptr, AV_LOG_ERROR, "Can not Alloc Frame.\n");
        return;
    }
    //! change while reopen
    int m_video_stream = -1;
    AVCodecContext *m_avctx = nullptr;
    AVFormatContext *m_input_ctx = nullptr;
    //! unchange config
    auto fer = pstream->config.fer;
    auto reopen = pstream->config.reopen;
    //! info
    std::string stream_name = pstream->config.name;
    size_t video_id = pstream->stream_handle;
    size_t nr_total_frames = pstream->total_frames;
    size_t size_in_bytes_per_image = 0;
    size_t nr_decoded_frames = 0;
    double total_time = 0;
    Timer timer;
    Timer timer_recieve_interval;
    Timer timer_trans;
    double total_transform_time = 0;
    //! tops runtime
    topsStream_t stream = reinterpret_cast<topsStream_t>(pstream->rt_resource.stream);
    topsEvent_t start_event = reinterpret_cast<topsEvent_t>(pstream->rt_resource.start_event);
    topsEvent_t end_event = reinterpret_cast<topsEvent_t>(pstream->rt_resource.end_event);

    enum class STATE {
        START,          //! start
        READ_FRAME,     //! demux
        RECIEVEING,     //! recieving
        LAST_FRAME,     //! last frame
        ERROR_HAND,     //! avcodec error
        QUIT,           //! stop
    };
    STATE state = STATE::START;
    //! process handle
    auto recieve_one_frame = [&](AVCodecContext *av_ctx, AVFrame *av_frame) -> int {
        auto ret = avcodec_receive_frame(av_ctx, av_frame);
        AV_LOG(ret, av_ctx, AV_LOG_ERROR, "avcodec_receive_frame");
        if (ret == 0) {
            //! decode succuess
            nr_decoded_frames++;
            nr_total_frames++;
            size_t interval = timer_recieve_interval.elapsed() * 1000;
            fprintf(stderr, "{\"message\": \"e2e_metric\", \"ops\": \"decode_done\", \"value\": %zu, \"total_frames\": \"%zu\", \"avg_fps\": \"%.2f\", \"video_id\": \"%zu\"}\n",interval , nr_decoded_frames, nr_decoded_frames * 1000 / total_time, video_id);
            timer_recieve_interval.start();
            av_log(av_ctx, AV_LOG_ERROR, "Recieve One Frame : %zu\n", nr_decoded_frames);
            if (fer != 0) {
                //! D2D copy from av_frame to topsCVImage
                std::shared_ptr<E2EImage> gcu_image = this->m_images_pool->get_gcu_image(av_frame->width, av_frame->height, av_format_2_e2e_format((AVPixelFormat)(av_frame->format)));
                gcu_image->frame_id = nr_decoded_frames;
                ENFLAME_RUNTIME_CHECK(topsEventRecord(start_event, stream));
                ENFLAME_RUNTIME_CHECK(topsMemcpyDtoDAsync(gcu_image->buf->extend_data, av_frame->data[0], size_in_bytes_per_image, stream));
                ENFLAME_RUNTIME_CHECK(topsEventRecord(end_event, stream));
                ENFLAME_RUNTIME_CHECK(topsEventSynchronize(end_event));
                // get time
                float duration = 0;
                ENFLAME_RUNTIME_CHECK(topsEventElapsedTime(&duration, start_event, end_event));
                total_transform_time += duration;
                av_log(av_ctx, AV_LOG_ERROR, "TransForm FPS : %lf\n", nr_decoded_frames * 1000.0 / total_transform_time);
                //! add task
                this->m_decoded_images->insert(gcu_image);
                //! write image for debug
                // if (nr_decoded_frames <= 300) {
                //     std::string filename = "video" + std::to_string(video_id) + "@" + std::to_string(nr_decoded_frames) + ".png";
                //     // e2e_utils::save_cpu_image_by_rgb(filename, sw_frame->data[0], sw_frame->width, sw_frame->height);
                //     e2e_utils::save_gcu_image_by_rgb(filename, gcu_image->buf->extend_data, av_frame->width, av_frame->height);
                // }
            }
        }
        av_frame_unref(av_frame);
        return ret;
    };
    timer_recieve_interval.start();
    while (pstream->is_working) {
        switch (state) {
            case STATE::START : {
                m_avctx = pstream->av_resource.avctx;
                m_input_ctx = pstream->av_resource.input_ctx;
                size_in_bytes_per_image = 640 * 640 *3;
                m_video_stream = pstream->video_stream;
                total_time = 0;
                state = STATE::READ_FRAME;
                break;
            }
            case STATE::READ_FRAME : {
                timer.start();
                auto ret = av_read_frame(m_input_ctx, &packet);
                AV_LOG(ret, m_avctx, AV_LOG_ERROR, "av_read_frame");
                if (ret == 0) {
                    if (packet.stream_index == m_video_stream) {
                        //! send packet
                        ret = avcodec_send_packet(m_avctx, &packet);
                        AV_LOG(ret, m_avctx, AV_LOG_ERROR, "avcodec_send_packet");
                        if (ret < 0) { av_packet_unref(&packet); }
                        state = ret < 0 ? STATE::ERROR_HAND : STATE::RECIEVEING;
                    }
                    else {
                        //! Audio frame. Read Frame Again
                        double duration = timer.elapsed();
                        total_time += duration;
                        av_log(m_avctx, AV_LOG_ERROR, "video stream mismatch : packet(%d) vs video(%d) using %lfms\n", packet.stream_index, m_video_stream, duration);
                        state = STATE::READ_FRAME;
                        av_packet_unref(&packet);
                    }
                }
                else if (ret == AVERROR_EOF) {
                    av_packet_unref(&packet);
                    if (packet.stream_index == m_video_stream) {
                        packet.data = nullptr;
                        packet.size = 0;
                        auto ret = avcodec_send_packet(m_avctx, &packet);
                        AV_LOG(ret, m_avctx, AV_LOG_ERROR, "avcodec_send_packet");
                        state = ret < 0 ? STATE::ERROR_HAND : STATE::LAST_FRAME;
                    }
                    else {
                        state = STATE::ERROR_HAND;
                    }
                }
                else {
                    av_packet_unref(&packet);
                    state = STATE::ERROR_HAND;
                }
                break;
            }
            case STATE::RECIEVEING : {
                //! recieving until got ret < 0
                auto ret = recieve_one_frame(m_avctx, frame);
                while (ret == 0) {
                    ret = recieve_one_frame(m_avctx, frame);
                }
                av_packet_unref(&packet);
                double duration = timer.elapsed();
                total_time += duration;
                state = STATE::READ_FRAME;
                // usleep(10000);
                break;
            }
            case STATE::LAST_FRAME : {
                //! receiving until got EOF
                auto ret = recieve_one_frame(m_avctx, frame);
                while (ret != AVERROR_EOF) {
                    ret = recieve_one_frame(m_avctx, frame);
                }
                av_packet_unref(&packet);
                double duration = timer.elapsed();
                total_time += duration;
                state = STATE::ERROR_HAND;
                break;
            }
            case STATE::ERROR_HAND : {
                //! reopen or not
                if (reopen) {
                    this->reopen(pstream);
                    state = STATE::START;
                }
                else {
                    av_log(m_avctx, AV_LOG_DEBUG, "Stop Decoding and Ready to QUIT.\n");
                    state = STATE::QUIT;
                }
                break;
            }
            case STATE::QUIT :
            default : {
                pstream->is_working = false;
                break;
            }
        }
    }
    if (state == STATE::READ_FRAME || state == STATE::RECIEVEING) {
        auto ret = recieve_one_frame(m_avctx, frame);
        while (ret == 0) {
            ret = recieve_one_frame(m_avctx, frame);
        }
        av_packet_unref(&packet);
    }
    if (state == STATE::LAST_FRAME) {
        auto ret = recieve_one_frame(m_avctx, frame);
        while (ret != AVERROR_EOF) {
            ret = recieve_one_frame(m_avctx, frame);
        }
        av_packet_unref(&packet);
    }
    ENFLAME_RUNTIME_CHECK(topsStreamSynchronize(stream));
    av_frame_free(&frame);
    fprintf(stderr, "FFmpegGCU Decode Stream %s Stop(%zu).\n", stream_name.c_str(), pstream->stream_handle);
}

size_t FFmpegGCU::open_stream(const FFmpegGCUConfig& config) {
    m_stream_handle++;
#if DEBUG
    fprintf(stderr, "Open Stream(%s)\t addr(%s)\t @{fer(%d), card(%d), dev(%d), reopen(%d)}\n", config.name.c_str(), config.addr.c_str(), config.fer, config.card_id, config.dev_id, config.reopen);
#endif
    StreamInfo* pstream = new StreamInfo;
    pstream->config = config;
    pstream->stream_handle = m_stream_handle;
    ENFLAME_E2E_CHECK(open(pstream), "open_stream");
    {
        std::unique_lock<std::mutex> lck(m_streams_mutex);
        topsStream_t stream = nullptr;
        ENFLAME_RUNTIME_CHECK(topsStreamCreate(&stream));
        pstream->rt_resource.stream = static_cast<void*>(stream);
        topsEvent_t start_event = nullptr;
        topsEvent_t end_event = nullptr;
        ENFLAME_RUNTIME_CHECK(topsEventCreate(&start_event));
        ENFLAME_RUNTIME_CHECK(topsEventCreate(&end_event));
        pstream->rt_resource.start_event = static_cast<void*>(start_event);
        pstream->rt_resource.end_event = static_cast<void*>(end_event);
        m_streams.emplace(m_stream_handle, pstream);
    }
    pstream->decode_thread = std::thread(&FFmpegGCU::work_func, this, pstream);
    return m_stream_handle;
}

E2E_Status FFmpegGCU::close_stream(size_t stream_handle) {
    fprintf(stderr, "Try Stop Video Stream Handle %zu.\n", stream_handle);
    StreamInfo* pstream = nullptr;
    {
        std::unique_lock<std::mutex> lck(m_streams_mutex);
        auto iter = m_streams.find(stream_handle);
        if (iter != m_streams.end()) {
            pstream = iter->second;
            m_streams.erase(iter);
        }
    }
    if (pstream) {
        pstream->is_working = false;
        if (pstream->decode_thread.joinable()) {
            pstream->decode_thread.join();
        }
        clear_av_resource(pstream);
        clear_rt_resource(pstream);
    }
    else {
        fprintf(stderr, "Stream Handle %zu Does Not Exists\n", stream_handle);
    }
    return E2E_Status::SUCCESS;
}

E2E_Status FFmpegGCU::stop_all() {
    std::unique_lock<std::mutex> lck(m_streams_mutex);
    for (auto&& stream : m_streams) {
        stream.second->is_working = false;
    }
    return E2E_Status::SUCCESS;
}

E2E_Status FFmpegGCU::open(StreamInfo* pstream) {
    auto stream_addr = pstream->config.addr;
    auto card_id = pstream->config.card_id;
    auto dev_id = pstream->config.dev_id;
    AVBufferRef *hw_device_ctx = nullptr;
    AVFormatContext *input_ctx = nullptr;
    AVCodecContext *avctx = nullptr;
    int video_stream = 0;

    auto release = [&]() -> void {
        if (hw_device_ctx) {
            av_buffer_unref(&hw_device_ctx);
        }
        if (avctx) {
            avcodec_free_context(&avctx);
        }
        if (input_ctx) {
            avformat_close_input(&input_ctx);
        }
    };

    av_log(nullptr, AV_LOG_DEBUG, "Open : %s\n", stream_addr.c_str());

    //! find device
    enum AVHWDeviceType type = av_hwdevice_find_type_by_name("topscodec");
    if (type == AV_HWDEVICE_TYPE_NONE) {
        av_log(nullptr, AV_LOG_ERROR, "Device type topscodec not support\n");
        av_log(nullptr, AV_LOG_ERROR, "Available device type :\n");
        while((type = av_hwdevice_iterate_types(type)) != AV_HWDEVICE_TYPE_NONE) {
            av_log(nullptr, AV_LOG_ERROR, "%s\n", av_hwdevice_get_type_name(type));
        }
        return E2E_Status::GCU_NOT_FOUND;
    }
    //! open video
    int ret = avformat_open_input(&input_ctx, stream_addr.c_str(), nullptr, nullptr);
    AV_LOG(ret, nullptr, AV_LOG_ERROR, "avformat_open_input");
    if (ret < 0) {
        release();
        return E2E_Status::VIDEO_NOT_FOUND;
    }
    //! find stream info
    ret = avformat_find_stream_info(input_ctx, nullptr);
    AV_LOG(ret, nullptr, AV_LOG_ERROR, "avformat_find_stream_info");
    if (ret < 0) {
        release();
        return E2E_Status::VIDEO_INFO_ERROR;
    }
    //! find best stream
    AVStream *video = nullptr;
    for (size_t i = 0; i < input_ctx->nb_streams; i++) {
        if (input_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video = input_ctx->streams[i];
            video_stream = i;
            break;
        }
    }
    if (nullptr == video) {
        av_log(nullptr, AV_LOG_ERROR, "Video Stream is nullptr\n");
        release();
        return E2E_Status::INVALID_VIDEO;
    }
    else {
        av_log(nullptr, AV_LOG_DEBUG, "Find Video Stream : %d\n", video_stream);
    }
    AVCodec *decoder = nullptr;
    //! find codec
    switch(video->codecpar->codec_id) {
        case AV_CODEC_ID_H264:
            decoder = avcodec_find_decoder_by_name("h264_topscodec");
            break;
        case AV_CODEC_ID_HEVC:
            decoder = avcodec_find_decoder_by_name("hevc_topscodec");
            break;
        case AV_CODEC_ID_VP8:
            decoder = avcodec_find_decoder_by_name("vp8_topscodec");
            break;
        case AV_CODEC_ID_VP9:
            decoder = avcodec_find_decoder_by_name("vp9_topscodec");
            break;
        case AV_CODEC_ID_MJPEG:
            decoder = avcodec_find_decoder_by_name("mjpeg_topscodec");
            break;
        case AV_CODEC_ID_H263:
            decoder = avcodec_find_decoder_by_name("h263_topscodec");
            break;
        case AV_CODEC_ID_MPEG2VIDEO:
            decoder = avcodec_find_decoder_by_name("mpeg2_topscodec");
            break;
        case AV_CODEC_ID_MPEG4:
            decoder = avcodec_find_decoder_by_name("mpeg4_topscodec");
            break;
        case AV_CODEC_ID_VC1:
            decoder = avcodec_find_decoder_by_name("vc1_topscodec");
            break;
        case AV_CODEC_ID_CAVS:
            decoder = avcodec_find_decoder_by_name("avs_topscodec");
            break;
        case AV_CODEC_ID_AVS2:
            decoder = avcodec_find_decoder_by_name("avs2_topscodec");
            break;
        case AV_CODEC_ID_AV1:
            decoder = avcodec_find_decoder_by_name("av1_topscodec");
            break;
        default:
            decoder = avcodec_find_decoder(video->codecpar->codec_id);
            break;
    }
    if(nullptr == decoder) {
        av_log(nullptr, AV_LOG_ERROR, "Unsupported codec!\n");
        release();
        return E2E_Status::CODEC_UNSUPPORT;
    }
    else {
        av_log(nullptr, AV_LOG_DEBUG, "Decoder Found : %p(%s)\n", decoder, avcodec_get_name(video->codecpar->codec_id));
    }
    //! avctx
    avctx = avcodec_alloc_context3(decoder);
    if (!avctx) {
        av_log(nullptr, AV_LOG_ERROR, "avcodec_alloc_context3 fail\n");
        release();
        return E2E_Status::CODEC_ALLOC_CONTEXT_FAIL;
    }

    ret = avcodec_parameters_to_context(avctx, video->codecpar);
    AV_LOG(ret, avctx, AV_LOG_ERROR, "avcodec_parameters_to_context");
    if (ret < 0) {
        release();
        return E2E_Status::CODEC_PARM_TO_CONTEXT_FAIL;
    }
    //! hw_device_ctx
    ret = av_hwdevice_ctx_create(&hw_device_ctx, type, std::to_string(dev_id).c_str(), nullptr, 0);
    AV_LOG(ret, avctx, AV_LOG_ERROR, "av_hwdevice_ctx_create");
    if (ret < 0) {
        release();
        return E2E_Status::INVALID_DEV_ID;
    }
    avctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

    AVDictionary* dec_opts = nullptr;
    av_dict_set(&dec_opts, "card_id", std::to_string(card_id).c_str(), 0);
    av_dict_set(&dec_opts, "device_id", std::to_string(dev_id).c_str(), 0);
    //! color space trans
    av_dict_set(&dec_opts, "output_pixfmt", "rgb24", 0);
    av_dict_set_int(&dec_opts, "enable_resize", 1, 0);
    //! downsampling scale
    av_dict_set_int(&dec_opts, "resize_w", 640, 0);
    av_dict_set_int(&dec_opts, "resize_h", 640, 0);
    av_dict_set_int(&dec_opts, "resize_m", 0, 0);
    if (pstream->config.fer > 0) {
        av_dict_set_int(&dec_opts, "enable_sfo", 1, 0);
        av_dict_set_int(&dec_opts, "sfo", pstream->config.fer, 0);
    }
    //! open
    ret = avcodec_open2(avctx, decoder, &dec_opts);
    AV_LOG(ret, avctx, AV_LOG_ERROR, "avcodec_open2");
    av_dict_free(&dec_opts);

    if (ret < 0) {
        release();
        return E2E_Status::OPEN_CODEC_FAIL;
    }

    if (!avctx->hw_frames_ctx) {
        av_log(avctx, AV_LOG_ERROR, "hw_frames_ctx is nullptr.\n");
        release();
        return E2E_Status::HW_NULL_CONTEXT;
    }

    //! init pstream
    pstream->total_frames = 0;
    pstream->video_stream = video_stream;
    //! av resource
    pstream->av_resource.input_ctx = input_ctx;
    pstream->av_resource.hw_device_ctx = hw_device_ctx;
    pstream->av_resource.avctx = avctx;

    return E2E_Status::SUCCESS;
}

E2E_Status FFmpegGCU::reopen(StreamInfo* pstream) {
    clear_av_resource(pstream);
    return open(pstream);
}

void FFmpegGCU::clear_av_resource(StreamInfo* pstream) {
    if (pstream->av_resource.avctx) {
        avcodec_free_context(&pstream->av_resource.avctx);
    }
    if (pstream->av_resource.input_ctx) {
        avformat_close_input(&pstream->av_resource.input_ctx);
    }
    if (pstream->av_resource.hw_device_ctx) {
        av_buffer_unref(&pstream->av_resource.hw_device_ctx);
    }
}

void FFmpegGCU::clear_rt_resource(StreamInfo* pstream) {
    if (pstream->rt_resource.end_event) {
        topsEvent_t event = static_cast<topsEvent_t>(pstream->rt_resource.end_event);
        ENFLAME_RUNTIME_CHECK(topsEventDestroy(event));
    }
    if (pstream->rt_resource.start_event) {
        topsEvent_t event = static_cast<topsEvent_t>(pstream->rt_resource.start_event);
        ENFLAME_RUNTIME_CHECK(topsEventDestroy(event));
    }
    if (pstream->rt_resource.stream) {
        topsStream_t stream = static_cast<topsStream_t>(pstream->rt_resource.stream);
        ENFLAME_RUNTIME_CHECK(topsStreamDestroy(stream));
    }
}

std::shared_ptr<E2EImage> FFmpegGCU::get_task() {
    auto res = m_decoded_images->get();
    if (res.first) {
        return res.second;
    }
    else {
        return nullptr;
    }
}

}
