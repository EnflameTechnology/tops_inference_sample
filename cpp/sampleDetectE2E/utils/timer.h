#ifndef __TIMER_H__
#define __TIMER_H__

#include <chrono>

namespace e2e_sample {
class Timer {
public:
    Timer();
    void start();
    double elapsed();
private:
    bool m_started{false};
    std::chrono::steady_clock::time_point m_start_time;
};

}

#endif
