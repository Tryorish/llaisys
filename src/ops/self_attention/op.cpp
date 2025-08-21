#include "op.hpp"
#include "../../utils.hpp"
#include "../../core/llaisys_core.hpp"
#include "cpu/self_attention_cpu.hpp"
namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
   CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(q->ndim() == 3, "Q must be 3D [seqlen, nhead, d]");
    ASSERT(k->ndim() == 3, "K must be 3D [total_len, nkvhead, d]");
    ASSERT(v->ndim() == 3, "V must be 3D [total_len, nkvhead, dv]");
    ASSERT(attn_val->ndim() == 3, "output must be 3D [seqlen, nhead, dv]");
    ASSERT(q->shape()[2] == k->shape()[2], "Q and K must have the same d dimension");
    ASSERT(k->shape()[1] == v->shape()[1], "K and V must have the same nkvhead");
    ASSERT(q->shape()[1] == attn_val->shape()[1], "Q and output must have same nhead");
    ASSERT(v->shape()[2] == attn_val->shape()[2], "V and output must have same dv");
    int64_t seqlen = q->shape()[0];
    int64_t total_len = k->shape()[0];
    int64_t nhead = q->shape()[1];
    int64_t nkvhead = k->shape()[1];
    int64_t d = q->shape()[2];
    int64_t dv = v->shape()[2];
    
    if(attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), q->dtype(), seqlen, total_len, nhead, nkvhead, d, dv, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch(attn_val->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), q->dtype(), seqlen, total_len, nhead, nkvhead, d, dv, scale);
#ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
            TO_BE_IMPLEMENTED();
            return;
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
