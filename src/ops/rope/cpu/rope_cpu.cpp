#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, int64_t seqlen, int64_t nhead, int64_t d, float theta) {
    // 将维度分割为两半
    const int64_t half_d = d / 2;
    
    for(int64_t seq = 0; seq < seqlen; seq++) { // 遍历序列
        int pos = pos_ids[seq]; // 当前位置id
        for(int64_t head = 0; head < nhead; head++) { // 遍历注意力头
            for(int64_t j = 0; j < half_d; j++) { // 遍历每组
                // 计算角度 angle = pos / (theta^(2j/d))
                float angle = pos / std::pow(theta, 2.0f * j / d);
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);
                // 计算输入索引
                int64_t base_idx = (seq * nhead + head) * d;
                int64_t a_idx = base_idx + j; // 前半段 
                int64_t b_idx = base_idx + j + half_d; // 后半段 
                // 取值
                float a = llaisys::utils::cast<float>(in[a_idx]);
                float b = llaisys::utils::cast<float>(in[b_idx]);
                // 旋转变换
                out[a_idx] = llaisys::utils::cast<T>(a * cos_val - b * sin_val);
                out[b_idx] = llaisys::utils::cast<T>(a * sin_val + b * cos_val);
            }
        }
    }
}

namespace llaisys::ops::cpu {
    void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, int64_t seqlen, int64_t nhead, int64_t d, float theta) {
        switch(type) {
            case LLAISYS_DTYPE_F32:
                return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const int64_t *>(pos_ids), seqlen, nhead, d, theta);
            case LLAISYS_DTYPE_BF16:
                return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), seqlen, nhead, d, theta);
            case LLAISYS_DTYPE_F16:
                return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), seqlen, nhead, d, theta);
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
}