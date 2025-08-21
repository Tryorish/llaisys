#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
    void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, int64_t num_indices, int64_t embedding_dim);
}