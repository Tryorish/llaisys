#include "op.hpp"
#include "../../utils.hpp"
#include "../../core/llaisys_core.hpp"
#include "cpu/rms_norm_cpu.hpp"
namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(in->ndim() == 2 && out->ndim() == 2, "rms_norm:: inputs must be 2D.");
    ASSERT(in->shape()[0] == out->shape()[0] && in->shape()[1] == out->shape()[1], "rms_norm:: input and output shapes must match.");
    ASSERT(weight->ndim() == 1 && weight->shape()[0] == in->shape()[1], "rms_norm:: weight must be 1D and match input's last dimension.");
    ASSERT(eps > 0, "rms_norm:: eps must be positive.");
    if(out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), in->dtype(), in->shape()[0], in->shape()[1], eps);
    }
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch(out->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::rms_norm(out->data(), in->data(), weight->data(), in->dtype(), in->shape()[0], in->shape()[1], eps);
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
