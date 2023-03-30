/*=======================================================================
 * Copyright 2020-2023 Enflame. All Rights Reserved.
 *
 *Licensed under the Apache License, Version 2.0 (the "License");
 *you may not use this file except in compliance with the License.
 *You may obtain a copy of the License at
 *
 *http://www.apache.org/licenses/LICENSE-2.0
 *
 *Unless required by applicable law or agreed to in writing, software
 *distributed under the License is distributed on an "AS IS" BASIS,
 *WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *See the License for the specific language governing permissions and
 *limitations under the License.
 *=======================================================================
 */

#ifndef __CHANNEL_HPP__
#define __CHANNEL_HPP__

#include <cassert>
#include <condition_variable>
#include <mutex>
#include <vector>

template <typename T> class Queue {
public:
  explicit Queue(int tsize) {
    capacity_ = tsize;
    rear_ = capacity_ - 1;
    front_ = size_ = 0;
    array_.resize(tsize);
  }
  bool full() {
    std::unique_lock<std::mutex> the_lock(m_mutex_);
    return size_ == capacity_;
  }
  bool empty() {
    std::unique_lock<std::mutex> the_lock(m_mutex_);
    return size_ == 0;
  }
  void push(T &&v) {
    std::unique_lock<std::mutex> the_lock(m_mutex_);
    assert(size_ < capacity_ && "queue empty!");
    rear_ = (rear_ + 1) % capacity_;
    array_[rear_] = std::move(v);
    size_++;
  }
  T pop() {
    std::unique_lock<std::mutex> the_lock(m_mutex_);
    assert(size_ > 0 && "queue empty!");
    auto item = std::move(array_[front_]);
    front_ = (front_ + 1) % capacity_;
    size_--;
    return item;
  }

private:
  int front_, rear_, size_;
  unsigned capacity_;
  std::vector<T> array_;
  std::mutex m_mutex_;
};

// This class implements the gorounte channel in golang:
// https://gobyexample.com/channels
template <class T> class Chan {
public:
  using type = T;
  explicit Chan(unsigned int queue_size = 32) : m_vals_(queue_size) {}
  virtual ~Chan() = default;
  Chan &operator=(const Chan &other) = delete;
  Chan(const Chan &other) = delete;

  T receive() {
    std::unique_lock<std::mutex> the_lock(m_mutex_);
    m_cv_.wait(the_lock, [this] { return !m_vals_.empty() || m_isclose_; });
    if (m_vals_.empty()) {
      return T();
    }
    auto a = std::move(m_vals_.pop());
    m_cv_.notify_all();
    return a;
  };
  void send(T &&val) {
    std::unique_lock<std::mutex> the_lock(m_mutex_);
    m_cv_.wait(the_lock, [this] { return !(m_vals_.full()); });
    m_vals_.push(std::move(val));
    m_cv_.notify_all();
  }

  void close() {
    std::unique_lock<std::mutex> the_lock(m_check_close_mutex_);
    m_isclose_ = true;
    m_cv_.notify_all();
  }
  bool closed() {
    std::unique_lock<std::mutex> the_lock(m_check_close_mutex_);
    return m_vals_.empty() && m_isclose_;
  }

protected:
  Queue<T> m_vals_;
  std::mutex m_mutex_;
  std::mutex m_check_close_mutex_;
  std::condition_variable m_cv_;
  bool m_isclose_{false};
};
#endif