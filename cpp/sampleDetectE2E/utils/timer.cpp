#include <iostream>

#include "timer.h"

namespace e2e_sample {

//! =============== Timer ======================
Timer::Timer() {}

void Timer::start() {
    m_started = true;
    m_start_time = std::chrono::steady_clock::now();
}

double Timer::elapsed() {
    if (!m_started) {
        fprintf(stderr, "Timer Not Start.\n");
        return 0;
    }
    m_started = false;
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - m_start_time).count();
    return duration / 1000.0;
}

}