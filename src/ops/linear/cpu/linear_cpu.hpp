#pragma once
#include "llaisys.h"

namespace llaisys::ops::cpu {
    void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, int64_t batch_size, int64_t in_features, int64_t out_features);
}