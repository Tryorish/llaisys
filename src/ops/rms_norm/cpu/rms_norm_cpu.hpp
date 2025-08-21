#pragma once
#include "llaisys.h"

namespace llaisys::ops::cpu {
    void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, int64_t rows, int64_t cols, float eps);
}