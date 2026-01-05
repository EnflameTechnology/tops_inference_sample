#ifndef __RESIZE_H__
#define __RESIZE_H__

#include <atomic>
#include <vector>

#include "tops/tops_runtime_api.h"

#include "include/config.h"
#include "include/module.h"

#include "pool/buffer_pool.h"
#include "multi_threads/multi_threads_worker.h"
#include "queue/thread_safe_queue.hpp"

namespace e2e_sample {

class TopsResize : public ModuleBase, public MultiThreadsWorker {
public:
    explicit TopsResize(const MultiThreadsModuleConfig& config);
    ~TopsResize();
    void start();
    void stop();

public:
    virtual std::shared_ptr<E2EImage> get_task() override;

protected:
    virtual void work_func(int thread_id) override;

private:
    MultiThreadsModuleConfig m_config;
    std::atomic_bool m_is_working{true};
    std::vector<TopsRuntimeResource> m_rt_resource;
    std::vector<void*> m_ops;
    std::unique_ptr<ThreadSafeQueue<std::shared_ptr<E2EImage>>> m_resized_images{nullptr};
    std::unique_ptr<ImageBufferPool> m_images_pool{nullptr};
};

}


#endif