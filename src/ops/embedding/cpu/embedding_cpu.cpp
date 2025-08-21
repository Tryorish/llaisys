#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

template<typename T>
void embedding_(T *out, const int64_t *index, const T *weight, int64_t num_indices, int64_t embedding_dim) {
    for(int64_t i = 0; i < num_indices; i++) {
        int64_t row = index[i];
        const T *src = weight + row * embedding_dim;
        T *dst = out + i * embedding_dim;
        std::copy(src, src + embedding_dim, dst);
    }
}

namespace llaisys::ops::cpu {
    void embedding(std::byte *out,const std::byte *index,const std::byte *weight, llaisysDataType_t type, int64_t num_indices, int64_t embedding_dim) {
        switch(type) {
            case LLAISYS_DTYPE_F32:
                return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const float *>(weight), num_indices, embedding_dim);
            case LLAISYS_DTYPE_BF16:
                return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const llaisys::bf16_t *>(weight), num_indices, embedding_dim);
            case LLAISYS_DTYPE_F16:
                return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const llaisys::fp16_t *>(weight), num_indices, embedding_dim);
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
}