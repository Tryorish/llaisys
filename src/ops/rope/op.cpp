#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"
namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    ASSERT(out->ndim() == 3 && in->ndim() == 3, "rope: inputs must be 3D [seqlen, nhead, d]");
    ASSERT(out->shape()[0] == in->shape()[0] && out->shape()[1] == in->shape()[1] && out->shape()[2] == in->shape()[2], 
        "rope: input and output shapes must match");
    ASSERT(pos_ids->ndim() == 1 && pos_ids->shape()[0] == in->shape()[0], "rope: pos_ids must be 1D with length equal to seqlen");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "rope: pos_ids must be int64");
    ASSERT(in->shape()[2] % 2 == 0, "rope: d must be even");
    if(out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), in->dtype(), in->shape()[0], in->shape()[1], in->shape()[2], theta);
    }
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch(out->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::rope(out->data(), in->data(), pos_ids->data(), in->dtype(), in->shape()[0], in->shape()[1], in->shape()[2], theta);
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
