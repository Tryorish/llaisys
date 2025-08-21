#pragma once
#include "llaisys.h"

namespace llaisys::ops::cpu {
    void self_attention(std::byte *atten_val, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t type, int64_t seqlen, int64_t total_len, int64_t nhead, int64_t nkvhead, int64_t d, int64_t dv, float scale);
}