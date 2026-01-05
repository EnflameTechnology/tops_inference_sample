#ifndef __THREAD_SAFE_QUEUE_HPP__
#define __THREAD_SAFE_QUEUE_HPP__

#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

#include "include/basic_type.h"

namespace e2e_sample {

template<typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(std::string name) : m_name(name) {}

    ~ThreadSafeQueue() {
        std::unique_lock<std::mutex> lck(m_queue_mutex);
        m_is_working = false;
        fprintf(stderr, "Queue : %s Deconstructor. Release %zu Tasks\n", m_name.c_str(), m_queue.size());
        std::queue<T> empty;
        m_queue.swap(empty);
    }

    void insert(T elem) {
        std::unique_lock<std::mutex> lck(m_queue_mutex);
        if (m_is_working && m_queue.size() <= 100) {
            m_queue.push(elem);
    #if DEBUG
            fprintf(stderr, "Queue : %s Size is : %zu\n",m_name.c_str(), m_queue.size());
    #endif
        }
    }

    std::pair<bool, T> get() {
        std::unique_lock<std::mutex> lck(m_queue_mutex);
        if (m_is_working) {
            if (!m_queue.empty()) {
                auto elem = m_queue.front();
                m_queue.pop();
                return {true, elem};
            }
        }
        return {false, T()};
    }

private:
    bool m_is_working{true};
    std::queue<T> m_queue;
    std::mutex m_queue_mutex;
    std::string m_name;
};

}

#endif