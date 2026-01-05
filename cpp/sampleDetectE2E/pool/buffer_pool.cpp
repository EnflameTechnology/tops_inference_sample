#if DEBUG
#include <iostream>
#endif
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/avassert.h>
#include <libavutil/imgutils.h>
}

#include "topscv/core/image.h"

#include "buffer_pool.h"

#define ENFLAME_RUNTIME_CHECK(_expr)                                           \
  do {                                                                         \
    if (topsError_t::topsSuccess != _expr) {                                   \
      fprintf(stderr, "EnFlame Runtime ERROR : %s @ %s:%d\n", topsGetErrorString(_expr), __FILE__, __LINE__);\
      raise(SIGUSR1);                                                                \
    }                                                                          \
  } while (0)

#define TOPS_CV_CHECK(_expr)\
    do { \
        if(topscvStatus_t::TOPSCV_STATUS_SUCCESS != _expr) { \
            const char* info = nullptr;\
            topscvErrCode2String(_expr, &info);\
            fprintf(stderr, "TopsCV ERROR : %s @ %s:%d\n", info, __FILE__, __LINE__);\
            raise(SIGUSR1);\
        }\
    } while(0)

namespace {
using ImageFormat = e2e_sample::ImageFormat;
topscvImageFormat_t e2e_format_2_gcu_cv_format(ImageFormat e2e_format) {
    switch (e2e_format) {
        case ImageFormat::RGB24 : return topscvImageFormat_t::TOPSCV_IMAGE_FORMAT_RGB_PACKED_U8;
        case ImageFormat::BGR24 : return topscvImageFormat_t::TOPSCV_IMAGE_FORMAT_BGR_PACKED_U8;
        default : return topscvImageFormat_t::TOPSCV_IMAGE_FORMAT_UNKNOWN;
    }
}

// topscvImageFormat_t av_format_2_gcu_cv_format(AVPixelFormat av_format) {
//     switch (av_format) {
//         case AV_PIX_FMT_GRAY8 : return topscvImageFormat_t::TOPSCV_IMAGE_FORMAT_GRAY_PACKED_U8;
//         case AV_PIX_FMT_YUVJ444P : return topscvImageFormat_t::TOPSCV_IMAGE_FORMAT_YUV444_PLANAR_U8;
//         case AV_PIX_FMT_RGB24 : return topscvImageFormat_t::TOPSCV_IMAGE_FORMAT_RGB_PACKED_U8;
//         case AV_PIX_FMT_BGR24 : return topscvImageFormat_t::TOPSCV_IMAGE_FORMAT_BGR_PACKED_U8;
//         default : return topscvImageFormat_t::TOPSCV_IMAGE_FORMAT_UNKNOWN;
//     }
// }

const char* ImageFormat2Str(ImageFormat format) {
    switch (format) {
        case ImageFormat::RGB24 : return "RGB24";
        case ImageFormat::BGR24 : return "BGR24";
        default : return "UNKNOW";
    }
}

const char* AVPixelFormat2Str(int format) {
    switch (format) {
        case AVPixelFormat::AV_PIX_FMT_YUV420P : return "YUV420P";
        case AVPixelFormat::AV_PIX_FMT_RGB24 : return "RGB24";
        case AVPixelFormat::AV_PIX_FMT_BGR24 : return "BGR24";
        default : return "UNKNOW";
    }
}

} 

namespace e2e_sample {

ImageBufferPool::ImageBufferPool() {}

ImageBufferPool::~ImageBufferPool() {
    //! delete gcu cv images
    std::unique_lock<std::mutex> lck(m_cv_image_pool_mutex);
    for (auto&& iter : m_cv_image_pool) {
        BufferQueue* queue = iter.second;
        while (1) {
            if (queue->chunks.size() < queue->nr_total_new) {
                fprintf(stderr, "BufferPool Waiting for all GCU CV Images Back. Total Chunks : %zu, Chucks Already Back : %zu\n", queue->nr_total_new, queue->chunks.size());
                usleep(1000 * 3000);
            }
            else {
                break;
            }
        }
        while (!queue->chunks.empty()) {
            E2EImage *raw = (E2EImage*)(queue->chunks.front());
            queue->chunks.pop();
            topscvImage_t gcu_cv_image = (topscvImage_t)(raw->buf->ptr);
            TOPS_CV_CHECK(topscvImageDestroy(&gcu_cv_image));
            gcu_cv_image = nullptr;
            raw->buf->ptr = nullptr;
            raw->buf->extend_data = nullptr;
            delete raw->buf;
            raw->buf = nullptr;
            delete raw;
            raw = nullptr;
        }
        fprintf(stderr, "Release All GCU CV Images. Total Chunks : %zu, Format is : %s\n", queue->nr_total_new, iter.first.c_str());
        delete queue;
        queue = nullptr;
        iter.second = nullptr;
    }
}

std::shared_ptr<E2EImage> ImageBufferPool::get_gcu_image(size_t width, size_t height, ImageFormat format) {
    std::string key = std::to_string(width) + "x" + std::to_string(height) + "@" + ImageFormat2Str(format);
    // fprintf(stderr, "key : %s\n", key.c_str());
    BufferQueue* queue = nullptr;
    {
        std::unique_lock<std::mutex> lck(m_cv_image_pool_mutex);
        auto res = m_cv_image_pool.emplace(key, nullptr);
        if (!res.second) {
            queue = res.first->second;
        }
        else {
            queue = new BufferQueue;
            queue->nr_total_new = 0;
            res.first->second = queue;
        }
    }
    //! get E2EImage cached or new
    E2EImage* raw = nullptr;
    {
        std::unique_lock<std::mutex> lck(queue->mutex);
        if (queue->chunks.empty()) {
            //! new E2EImage
            queue->nr_total_new++;
            topscvImage_t gcu_cv_image = nullptr;
            TOPS_CV_CHECK(topscvImageCreate(width, height, e2e_format_2_gcu_cv_format(format), &gcu_cv_image));
            topscvImageData_t gcu_cv_image_data;
            TOPS_CV_CHECK(topscvImageExportData(gcu_cv_image, &gcu_cv_image_data));
            raw = new E2EImage(width, height, format);
            raw->buf = new MemoryObj(MemoryType::GCU_CV_IMAGE, gcu_cv_image);
            raw->buf->size = gcu_cv_image_data.total_length;
            raw->buf->extend_data = gcu_cv_image_data.data;
        }
        if (!queue->chunks.empty()) {
            raw = (E2EImage*)(queue->chunks.front());
            queue->chunks.pop();
        }
    }
    auto deleter = [queue](E2EImage* content) {
        std::unique_lock<std::mutex> lck(queue->mutex);
        queue->chunks.push(content);
    };
    return std::shared_ptr<E2EImage>(raw, deleter);
}

GCUMemoryBufferPool::GCUMemoryBufferPool() {}

GCUMemoryBufferPool::~GCUMemoryBufferPool() {
    // delete gcu memory
    std::unique_lock<std::mutex> lck(m_gcu_buffer_pool_mutex);
    for (auto&& iter : m_gcu_buffer_pool) {
        BufferQueue* queue = iter.second;
        while (1) {
            if (queue->chunks.size() < queue->nr_total_new) {
                fprintf(stderr, "BufferPool Waiting for all GCU Buffer Back. Total Chunks : %zu, Chucks Already Back : %zu\n", queue->nr_total_new, queue->chunks.size());
                usleep(1000 * 3000);
            }
            else {
                break;
            }
        }
        while (!queue->chunks.empty()) {
            MemoryObj *raw = (MemoryObj*)(queue->chunks.front());
            queue->chunks.pop();
            void* gcu_ptr = raw->ptr;
            ENFLAME_RUNTIME_CHECK(topsFree(gcu_ptr));
            gcu_ptr = nullptr;
            raw->ptr = nullptr;
            raw->extend_data = nullptr;
            delete raw;
            raw = nullptr;
        }
        fprintf(stderr, "Release All GCU Buffer. Total Chunks : %zu, Size is : %zu\n", queue->nr_total_new, iter.first);
        delete queue;
        iter.second = nullptr;
    }
}

std::shared_ptr<MemoryObj> GCUMemoryBufferPool::get_gcu_memory(size_t size_in_bytes) {
    BufferQueue* queue = nullptr;
    {
        std::unique_lock<std::mutex> lck(m_gcu_buffer_pool_mutex);
        auto res = m_gcu_buffer_pool.emplace(size_in_bytes, nullptr);
        if (!res.second) {
            queue = res.first->second;
        }
        else {
            queue = new BufferQueue;
            queue->nr_total_new = 0;
            res.first->second = queue;
        }
    }
    //! get gcu_buffer cached or new
    MemoryObj* raw = nullptr;
    {
        std::unique_lock<std::mutex> lck(queue->mutex);
        if (queue->chunks.empty()) {
            //! new gcu buffer
            queue->nr_total_new++;
            void *ptr = nullptr;
            ENFLAME_RUNTIME_CHECK(topsMalloc(&ptr, size_in_bytes));
            raw = new MemoryObj(MemoryType::GCU_MEM, ptr, size_in_bytes);
        }
        if (!queue->chunks.empty()) {
            raw = (MemoryObj*)(queue->chunks.front());
            queue->chunks.pop();
        }
    }
    auto deleter = [queue](MemoryObj* content) {
        std::lock_guard lck(queue->mutex);
        queue->chunks.push(content);
    };
    return std::shared_ptr<MemoryObj>(raw, deleter);
}

}