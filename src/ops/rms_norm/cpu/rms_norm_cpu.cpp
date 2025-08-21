#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, int64_t rows, int64_t cols, float eps) {
    for(int r = 0; r < rows; r++) {
        // 计算每行的平方和
        float sum = 0;
        for(int c = 0; c < cols; c++) {
            float val = llaisys::utils::cast<float>(in[r * cols + c]); 
            sum += val * val; 
        }
        // 计算均方根
        float rms = std::sqrt(sum / cols + eps);
        // 归一化
        for(int c = 0; c < cols; c++) {
            float norm = llaisys::utils::cast<float>(in[r * cols + c]) / rms;
            out[r * cols + c] = llaisys::utils::cast<T>(norm * llaisys::utils::cast<float>(weight[c]));
        }
    }
}

namespace llaisys::ops::cpu {
    void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, int64_t rows, int64_t cols, float eps) {
        switch(type) {
            case LLAISYS_DTYPE_F32:
                return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), rows, cols, eps);
            case LLAISYS_DTYPE_BF16:
                return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), rows, cols, eps);
            case LLAISYS_DTYPE_F16:
                return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), rows, cols, eps);
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(type);     
        }
    }
}