#ifndef __NN_H__
#define __NN_H__

#include <atomic>
#include <future>
#include <mutex>
#include <vector>

#include "include/config.h"
#include "pool/buffer_pool.h"
#include "multi_threads/multi_threads_worker.h"
#include "queue/thread_safe_queue.hpp"

namespace e2e_sample {

struct E2ENNOutput {
    std::vector<std::shared_ptr<MemoryObj>> outputs;
    size_t frame_id = 0;
};

class TopsDetect : public MultiThreadsWorker {
public:
    explicit TopsDetect(const NNConfig& config);
    ~TopsDetect();
    void start();
    void stop();

public:
    E2ENNOutput get_output();

protected:
    virtual void work_func(int thread_id);

private:
    NNConfig m_config;
    std::atomic_bool m_is_working{true};
    std::vector<std::promise<void>> m_build_engine_done;
    std::mutex m_topsinfer_init_mutex;
    void* m_error_manager;
    std::unique_ptr<GCUMemoryBufferPool> m_gcu_mem_pool;
    std::unique_ptr<ThreadSafeQueue<E2ENNOutput>> m_nn_outputs{nullptr};
    //! maybe a queue to store output
};

}


#endif