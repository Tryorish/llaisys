#pragma once 
#include "llaisys.h"

namespace llaisys::ops::cpu {
    void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, int64_t seqlen, int64_t nhead, int64_t d, float theta);
}