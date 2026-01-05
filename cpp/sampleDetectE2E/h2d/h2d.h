#ifndef __H2D_H__
#define __H2D_H__

#include <atomic>
#include <vector>

#include "tops/tops_runtime_api.h"

#include "include/config.h"
#include "multi_threads/multi_threads_worker.h"

namespace e2e_sample {

//! a module to do H2D copy
class H2D : public MultiThreadsWorker {
public:
    explicit H2D(const MultiThreadsModuleConfig& config);
    ~H2D();
    void start();
    void stop();

protected:
    virtual void work_func(int thread_id) override;

private:
    MultiThreadsModuleConfig m_config;
    std::atomic_bool m_stopped{false};
    std::vector<topsStream_t> m_streams;
};

}


#endif