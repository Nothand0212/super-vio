#pragma once
#include <chrono>

class Timer
{
private:
  std::chrono::high_resolution_clock::time_point start_, end_;

  double time_consumed_ms_double_;

public:
  Timer()  = default;
  ~Timer() = default;
  void tic()
  {
    start_ = std::chrono::high_resolution_clock::now();
  }

  void toc()
  {
    end_                     = std::chrono::high_resolution_clock::now();
    time_consumed_ms_double_ = std::chrono::duration<double, std::milli>( end_ - start_ ).count();
  }

  double getDuration() const
  {
    return time_consumed_ms_double_;
  }

  double tocGetDuration()
  {
    end_                     = std::chrono::high_resolution_clock::now();
    time_consumed_ms_double_ = std::chrono::duration<double, std::milli>( end_ - start_ ).count();
    return time_consumed_ms_double_;
  }
};