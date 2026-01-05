#ifndef __BUFFER_POOL_H__
#define __BUFFER_POOL_H__

#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>
#include <string>

#include "include/basic_type.h"

namespace e2e_sample {

struct BufferQueue {
    size_t nr_total_new = 0;
    std::mutex mutex;
    std::queue<void*> chunks;
    BufferQueue() {}
};

class ImageBufferPool {
public:
    std::shared_ptr<E2EImage> get_gcu_image(size_t width, size_t height, ImageFormat format);

public:
    ImageBufferPool();
    ~ImageBufferPool();

public:
    ImageBufferPool(const ImageBufferPool &) = delete;
    ImageBufferPool &operator=(const ImageBufferPool &) = delete;

private:
    std::unordered_map<std::string, BufferQueue*> m_cv_image_pool;
    std::mutex m_cv_image_pool_mutex;
};

class GCUMemoryBufferPool {
public:
    std::shared_ptr<MemoryObj> get_gcu_memory(size_t size_in_bytes);

public:
    GCUMemoryBufferPool();
    ~GCUMemoryBufferPool();

public:
    GCUMemoryBufferPool(const GCUMemoryBufferPool &) = delete;
    GCUMemoryBufferPool &operator=(const GCUMemoryBufferPool &) = delete;

private:
    std::unordered_map<size_t, BufferQueue*> m_gcu_buffer_pool;
    std::mutex m_gcu_buffer_pool_mutex;
};

}

#endif
