#include "op.hpp"
#include "cpu/linear_cpu.hpp"
#include "../../utils.hpp"
#include "../../core/llaisys_core.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if(bias) CHECK_SAME_DEVICE(out, bias);
    ASSERT(in->ndim() == 2 && out->ndim() == 2 && weight->ndim() == 2, "Linear: inputs must be 2D.");
    ASSERT(in->shape()[1] == weight->shape()[1], "Linear: in_features mismatch.");
    ASSERT(out->shape()[0] == in->shape()[0] && out->shape()[1] == weight->shape()[0], "Linear: output shape mismatch");
    if(bias) {
        ASSERT(bias->ndim() == 1 && bias->shape()[0] == weight->shape()[0], "Linear: bias shape mismatch.");
    }
    if(out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, in->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch(out->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, in->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
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
