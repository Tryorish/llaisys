#include "llaisys.h"

namespace llaisys::ops::cpu {
    void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, int64_t num);
}