#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, T *vals, size_t size) {
    float cur_val_ = llaisys::utils::cast<float>(vals[0]);
    int64_t cur_idx_ = 0;
    for (size_t i = 1; i < size; i++) {
        float val = llaisys::utils::cast<float>(vals[i]);
        if (val > cur_val_) {
            cur_val_ = val;
            cur_idx_ = i;
        }
    }
    *max_idx = cur_idx_;
    *max_val = llaisys::utils::cast<T>(cur_val_);
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, std::byte *vals, llaisysDataType_t type, size_t size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val), reinterpret_cast<float *>(vals), size);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val), reinterpret_cast<llaisys::bf16_t *>(vals), size);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val), reinterpret_cast<llaisys::fp16_t *>(vals), size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu