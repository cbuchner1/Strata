// Q8_0 block decode for the hyper-connection read that STRATA_HC_Q8 turns on.
//
// A Q8_0 block is 34 bytes: an fp16 scale `d`, then 32 int8 codes.  The CUDA engine reads these bytes
// directly for the hyper-connection projections (src/kernels/cuda/fused_gr.cu, gr_down_q8_kernel); the
// SYCL port instead dequantises them to BF16 at pack time (--compat-bf16, 1.18 GiB of VRAM).  These
// helpers are the format layer the port needs: no policy, no kernel, just the decode, so it can be
// checked against ggml's own dequantize_row_q8_0 before any kernel depends on it.
#pragma once
#include <cstdint>
#include <cstring>
#include <sycl/sycl.hpp>

namespace strata::kernels::q8 {

constexpr int kBlock = 32;   ///< values per block
constexpr int kBytes = 34;   ///< fp16 d + 32 int8

/// One block's scale, as a float.  `blk` points at the block's first byte.
inline float scale(const uint8_t *blk) {
    uint16_t h;
    std::memcpy(&h, blk, sizeof h);
    return (float) sycl::bit_cast<sycl::half>(h);
}

/// Value `i` of the block (0 <= i < 32): d * (int8) blk[2 + i].
inline float value(const uint8_t *blk, int i) {
    return scale(blk) * (float) (int8_t) blk[2 + i];
}

/// Four consecutive values starting at `i` (i a multiple of 4), as the CUDA read's `q8_four` does:
/// the codes are packed into a uint32 and each byte is sign-extended.
inline sycl::float4 four(uint32_t q, float d) {
    return sycl::float4(d * (float) (int8_t) (q & 255u), d * (float) (int8_t) ((q >> 8) & 255u),
                        d * (float) (int8_t) ((q >> 16) & 255u), d * (float) (int8_t) (q >> 24));
}

/// Byte stride of a row of `cols` values (cols a multiple of 32).
inline size_t row_bytes(int cols) { return (size_t) (cols / kBlock) * kBytes; }

}  // namespace strata::kernels::q8
