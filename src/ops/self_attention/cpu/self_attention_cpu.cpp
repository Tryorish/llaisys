#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>

template <typename T>
void causal_softmax(T *data, int64_t total_len, int64_t row) {
    // 遍历寻找当前行最大值
    float max_val = llaisys::utils::cast<float>(data[row * total_len]);
    for(int64_t col = 1; col < total_len; col++) {
        // 因果掩码
        if(col > row) break;
        max_val = std::max(max_val, llaisys::utils::cast<float>(data[row * total_len + col])); 
    }
    // 计算指数和
    float sum_exp = 0;
    for(int64_t col = 0; col < total_len; col++) {
        // 因果掩码
        if(col > row) {
            data[row * total_len + col] = llaisys::utils::cast<T>(0);
            continue;
        }
        float exp_val = std::exp(llaisys::utils::cast<float>(data[row * total_len + col]) - max_val);
        data[row * total_len + col] = llaisys::utils::cast<T>(exp_val);
        sum_exp += exp_val;
    }
    // 归一化
    for(int64_t col = 0; col < total_len; col++) {
        data[row * total_len + col] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(data[row * total_len + col]) / sum_exp);
    }
}


template <typename T>
void self_attention_(T *atten_val, const T *q, const T  *k, const T *v, int64_t seqlen, int64_t total_len, int64_t nhead, int64_t nkvhead, int64_t d, int64_t dv, float scale) {
    // q [seqlen, nhead, d]
    // k [total_len, nkvhead, d]
    // v [total_len, nkvhead, dv]
    // 取单个头
    // q * k^T [seqlen, total_len]
    // softmax(scale *(q * k^T)) * V [seqlen, dv]
    // atten_val [seqlen, nhead, dv]

    // 遍历每个查询头
    for(int64_t head = 0; head < nhead; head++) {
        // 存储每个头的注意力分数
        std::vector<T> atten_score(seqlen * total_len);
        // MHA: nhead == nkvhead q_head 和 kv_head 一对一
        // GQA: nhead > nkvhead 多个q_head共享一个kv_head 
        int64_t kv_head = head / (nhead / nkvhead);
        // 计算q * k^T
        for(int64_t row = 0; row < seqlen; row++) {
            for(int64_t col = 0; col < total_len; col++) {
                float dot_product = 0;
                for(int64_t i = 0; i < d; i++) {
                    int64_t q_idx = (row * nhead + head) * d + i;
                    int64_t k_idx = (col * nkvhead + kv_head) * d + i;
                    dot_product += llaisys::utils::cast<float>(q[q_idx]) * llaisys::utils::cast<float>(k[k_idx]);
                }
                atten_score[row * total_len + col] = llaisys::utils::cast<T>(dot_product * scale);
            }
            // 计算当前行的softmax
            causal_softmax(atten_score.data(), total_len, row);
        }
        // 注意力加权
        for(int64_t row = 0; row < seqlen; row++) {
            for(int64_t j = 0; j < dv; j++) {
                float sum = 0;
                for(int col = 0; col < total_len; col++) {
                    if(col > row) break;
                    int64_t v_idx = (col * nkvhead + kv_head) * dv + j;
                    sum += llaisys::utils::cast<float>(atten_score[row * total_len + col]) * llaisys::utils::cast<float>(v[v_idx]);
                }
                int64_t out_idx = (row * nhead + head) * dv + j;
                atten_val[out_idx] = llaisys::utils::cast<T>(sum);
            }
        }  
    }
}

namespace llaisys::ops::cpu {
    void self_attention(std::byte *atten_val, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t type, int64_t seqlen, int64_t total_len, int64_t nhead, int64_t nkvhead, int64_t d, int64_t dv, float scale) {
        switch(type) {
            case LLAISYS_DTYPE_F32:
                return self_attention_(reinterpret_cast<float *>(atten_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v), seqlen, total_len, nhead, nkvhead, d, dv, scale);
            case LLAISYS_DTYPE_BF16:
                return self_attention_(reinterpret_cast<llaisys::bf16_t *>(atten_val), reinterpret_cast<const llaisys::bf16_t *>(q), reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v), seqlen, total_len, nhead, nkvhead, d, dv, scale);
            case LLAISYS_DTYPE_F16:
                return self_attention_(reinterpret_cast<llaisys::fp16_t *>(atten_val), reinterpret_cast<const llaisys::fp16_t *>(q), reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v), seqlen, total_len, nhead, nkvhead, d, dv, scale);
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(type); 
        }
    }
}