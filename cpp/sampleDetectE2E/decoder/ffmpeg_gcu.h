#ifndef __FFMPEG_GCU_H__
#define __FFMPEG_GCU_H__

#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/avassert.h>
#include <libavutil/imgutils.h>
}

#include "include/config.h"
#include "include/error_code.h"
#include "include/module.h"
#include "pool/buffer_pool.h"
#include "queue/thread_safe_queue.hpp"

namespace e2e_sample {

struct TopsAVResource {
    AVBufferRef* hw_device_ctx = nullptr;
    AVFormatContext* input_ctx = nullptr;
    AVCodecContext* avctx = nullptr;
};

struct StreamInfo {
    FFmpegGCUConfig config;
    size_t stream_handle = 0;
    size_t video_stream = 0;
    std::atomic_bool is_working{true};
    std::thread decode_thread;
    size_t total_frames = 0;
    TopsAVResource av_resource;
    TopsRuntimeResource rt_resource;
};

class FFmpegGCU : public ModuleBase {
public:
    FFmpegGCU();
    ~FFmpegGCU();
    size_t open_stream(const FFmpegGCUConfig& config);
    E2E_Status close_stream(size_t stream_handle);
    E2E_Status stop_all();

public:
    virtual std::shared_ptr<E2EImage> get_task() override;

protected:
    void work_func(StreamInfo* stream_info);

private:
    E2E_Status open(StreamInfo* stream_info);
    E2E_Status reopen(StreamInfo* stream_info);
    void clear_av_resource(StreamInfo* stream_info);
    void clear_rt_resource(StreamInfo* stream_info);

private:
    std::atomic_size_t m_stream_handle{0};
    std::mutex m_streams_mutex;
    std::unordered_map<size_t, StreamInfo*> m_streams;
    std::unique_ptr<ImageBufferPool> m_images_pool;
    std::unique_ptr<ThreadSafeQueue<std::shared_ptr<E2EImage>>> m_decoded_images{nullptr};
};

} //! e2e_sample_gcu_ffmpeg

#endif