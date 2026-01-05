#include <iostream>

#include "multi_threads_worker.h"

namespace e2e_sample {

MultiThreadsWorker::MultiThreadsWorker(int nr_threads) : m_nr_threads(nr_threads) {
    m_threads.resize(nr_threads);
}

MultiThreadsWorker::~MultiThreadsWorker() {
    join_all();
    fprintf(stderr, "MultiThreadsWorker Deconstructor.\n");
    m_threads.clear();
}

void MultiThreadsWorker::start() {
    for (int i = 0; i < m_nr_threads; ++i) {
        m_threads.emplace_back(&MultiThreadsWorker::work_func, this, i);
    }
}

size_t MultiThreadsWorker::add_nr_task() {
    return m_nr_task.fetch_add(1) + 1;
}

void MultiThreadsWorker::join_all() {
    for (auto&& pthread : m_threads) {
        if (pthread.joinable()) {
            pthread.join();
        }
    }
}

}