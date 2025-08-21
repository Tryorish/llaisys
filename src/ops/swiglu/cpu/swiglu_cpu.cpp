#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
template <typename T>
void swiglu_(T *out, const T *gate, const T *up, int64_t num) {
    for(int64_t i = 0; i < num; i++) {
        float gate_val = llaisys::utils::cast<float>(gate[i]);
        float up_val = llaisys::utils::cast<float>(up[i]);
        float sigmoid_val = gate_val / (1.0f + std::exp(-gate_val));
        out[i] = llaisys::utils::cast<T>(up_val * sigmoid_val);
    }
}

namespace llaisys::ops::cpu {
    void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, int64_t num) {
        switch(type) {
            case LLAISYS_DTYPE_F32:
                return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate), reinterpret_cast<const float *>(up), num);
            case LLAISYS_DTYPE_BF16:
                return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(gate), reinterpret_cast<const llaisys::bf16_t *>(up), num);
            case LLAISYS_DTYPE_F16:
                return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(gate), reinterpret_cast<const llaisys::fp16_t *>(up), num);
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
}