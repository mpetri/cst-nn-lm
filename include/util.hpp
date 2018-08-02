#pragma once

#include <string>
#include <chrono>

#include "logging.hpp"

using namespace std::chrono;
using watch = std::chrono::high_resolution_clock;


struct cstnn_timer {
    watch::time_point start;
    std::string name;
    cstnn_timer(const std::string& _n)
        : name(_n)
    {
        CNLOG << "START(" << name << ")";
        start = watch::now();
    }
    ~cstnn_timer()
    {
        auto stop = watch::now();
        CNLOG << "STOP(" << name << ") - "
                  << duration_cast<milliseconds>(stop - start).count() / 1000.0f
                  << " sec";
    }
};