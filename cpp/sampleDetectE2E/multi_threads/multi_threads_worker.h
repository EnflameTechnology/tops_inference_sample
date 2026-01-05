#ifndef __MULTI_THREADS_WORKER_H__
#define __MULTI_THREADS_WORKER_H__

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

namespace e2e_sample {

class MultiThreadsWorker {
public:
    explicit MultiThreadsWorker(int nr_threads);
    virtual ~MultiThreadsWorker();
    size_t add_nr_task();

protected:
    void start();
    void join_all();
    virtual void work_func(int thread_id) = 0;

private:
    std::atomic_size_t m_nr_task{0};
    int m_nr_threads = 0;
    std::vector<std::thread> m_threads;
};

}

#endif
