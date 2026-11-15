#include "strata/kernels/q8_block.hpp"
#include <cstdio>
#include <cmath>
int main() {
    // two blocks: scales 0.5 and -1.25, codes i-16 and 16-i
    uint8_t buf[2 * strata::kernels::q8::kBytes] = {};
    auto put = [&](int blk, float d, int base) {
        uint16_t h = sycl::bit_cast<uint16_t>(sycl::half(d));
        std::memcpy(buf + blk * 34, &h, 2);
        for (int i = 0; i < 32; ++i) buf[blk * 34 + 2 + i] = (uint8_t) (int8_t) (base + i);
    };
    put(0, 0.5f, -16); put(1, -1.25f, 16);
    int bad = 0;
    for (int i = 0; i < 32; ++i) {
        float want = 0.5f * (float) (-16 + i);
        if (std::fabs(strata::kernels::q8::value(buf, i) - want) > 1e-6f) ++bad;
        float want2 = -1.25f * (float) (16 + i);
        if (std::fabs(strata::kernels::q8::value(buf + 34, i) - want2) > 1e-6f) ++bad;
    }
    // four(): codes 1,2,3,4 at d=2 -> 2,4,6,8
    uint8_t b2[34] = {}; uint16_t h = sycl::bit_cast<uint16_t>(sycl::half(2.0f)); std::memcpy(b2, &h, 2);
    b2[2] = 1; b2[3] = 2; b2[4] = 3; b2[5] = 4;
    uint32_t q; std::memcpy(&q, b2 + 2, 4);
    auto f = strata::kernels::q8::four(q, strata::kernels::q8::scale(b2));
    if (f.x() != 2.f || f.y() != 4.f || f.z() != 6.f || f.w() != 8.f) ++bad;
    if (strata::kernels::q8::row_bytes(2560) != 80 * 34) ++bad;
    std::printf("q8 decode: %s (%d mismatches)\n", bad ? "FAIL" : "ok", bad);
    return bad ? 1 : 0;
}
