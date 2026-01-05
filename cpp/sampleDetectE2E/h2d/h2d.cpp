#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/avassert.h>
#include <libavutil/imgutils.h>
}

#include "h2d.h"
#include "pool/buffer_pool.h"
#include "utils/timer.h"

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

H2D::H2D(const MultiThreadsModuleConfig& config) : MultiThreadsWorker(config.nr_threads), m_config(config) {}

H2D::~H2D() {
    if (!m_stopped) {
        stop();
    }
    MultiThreadsWorker::join_all();
    for (auto&& stream : m_streams) {
        ENFLAME_RUNTIME_CHECK(topsStreamSynchronize(stream));
        ENFLAME_RUNTIME_CHECK(topsStreamDestroy(stream));
    }
    fprintf(stderr, "H2D module Deconstructor End.\n");
}

void H2D::start() {
    for (int i = 0; i < m_config.nr_threads; ++i) {
        topsStream_t stream = nullptr;
        ENFLAME_RUNTIME_CHECK(topsStreamCreate(&stream));
        m_streams.push_back(stream);
    }
    MultiThreadsWorker::start();
}

void H2D::stop() {
    fprintf(stderr, "Try Stop H2D Module.\n");
    m_stopped.store(true);
}

void H2D::work_func(int thread_id) {
    auto queue_in = this->m_config.task_queue_in;
    auto queue_out = this->m_config.task_queue_out;
    auto stream = this->m_streams[thread_id];
    size_t nr_tasks = 0;
    float total_time = 0;
    // Timer timer;
    while(!this->m_stopped) {
        auto task = queue_in->get();
        if (task) {
            nr_tasks++;
            AVFrame *sw_frame = (AVFrame*)(task->cpu_frame_decode->ptr);
            //! rgb copy
            if (sw_frame->format == AV_PIX_FMT_RGB24) {
                int size = task->cpu_frame_decode->size;
                task->gcu_image_decode = BufferPool::get_instance().get_cv_image_gcu(sw_frame->width, sw_frame->height, av_format_2_e2e_format((AVPixelFormat)(sw_frame->format)));
                ENFLAME_RUNTIME_CHECK(topsEventRecord(task->start_event.get(), stream));
                ENFLAME_RUNTIME_CHECK(topsMemcpyHtoDAsync(task->gcu_image_decode->extend_data, sw_frame->data[0], size, stream));
                ENFLAME_RUNTIME_CHECK(topsEventRecord(task->stop_event.get(), stream));
                ENFLAME_RUNTIME_CHECK(topsEventSynchronize(task->stop_event.get()));
                float duration = 0;
                ENFLAME_RUNTIME_CHECK(topsEventElapsedTime(&duration, task->start_event.get(), task->stop_event.get()));
                total_time += duration;
                float avg_fps = nr_tasks * 1000.0 / total_time;
                fprintf(stderr, "{\"message\": \"e2e_metric\", \"ops\": \"H2D\", \"avg_fps\": %.2f, \"thread_id\": %d}\n", avg_fps, thread_id);
                task->cpu_frame_decode = nullptr;
            }
            queue_out->insert(task);
        }
        else {
            usleep(100);
        }
    }
    fprintf(stderr, "H2D Worker %d Stop.\n", thread_id);
}

}