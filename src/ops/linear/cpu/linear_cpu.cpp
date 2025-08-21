#include "linear_cpu.hpp"
#include "../../../utils.hpp"

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, int64_t batch_size, int64_t in_features, int64_t out_features) {
    // out = in * weight^T + bias
    // in   [batch_size * in_features]
    // weight [out_features * in_features]
    // bias [out_features,] or nullptr
    // out [batch_size * out_features]
    for(int64_t b = 0; b < batch_size; b++) {
        for(int64_t o = 0; o < out_features; o++) {
            float sum = 0;
            for(int64_t i = 0; i < in_features; i++) {
                sum += llaisys::utils::cast<float>(in[b * in_features + i]) * llaisys::utils::cast<float>(weight[o * in_features + i]);
            }
            if(bias) {
                sum += llaisys::utils::cast<float>(bias[o]);
            }

            out[b * out_features + o] = llaisys::utils::cast<T>(sum);
        }
    }
}

namespace llaisys::ops::cpu {
    void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, int64_t batch_size, int64_t in_features, int64_t out_features) {
        switch(type) {
            case LLAISYS_DTYPE_F32:
                return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), batch_size, in_features, out_features);
            case LLAISYS_DTYPE_BF16:
                return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), batch_size, in_features, out_features);
            case LLAISYS_DTYPE_F16:
                return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), batch_size, in_features, out_features);
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
}