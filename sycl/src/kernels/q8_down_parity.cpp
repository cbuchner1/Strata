// q8_down_parity.cpp - gr_down_q8_kernel against a CPU reference.
//
// The kernel reads the hyper-connection down projection from Q8_0 bytes (34 B blocks) and writes per-(token,
// chunk, row) partials.  A wrong lane map or block stride still produces plausible numbers, so this checks the
// partials summed over chunks against a double-precision dot product over the same dequantized weights, and the
// sum-of-squares side output against sum(R^2).
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/q8_block.hpp"
#include "strata/core/geometry.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace q8 = strata::kernels::q8;

namespace {
constexpr int N = strata::core::kCompiledGeometry.n_embd;
constexpr int HC = strata::core::kCompiledGeometry.hc;
constexpr int LR = strata::core::kCompiledGeometry.hc_lr;
constexpr int D = N * HC;
constexpr int KC = 640, NKC = D / KC, PR = LR + HC, T = 6;
}  // namespace

namespace strata::kernels {
void gr_down_q8_probe(const FusedGrArgs *args, int T, float *part, float *ssg, void *stream);
}

int main() {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> ux(-1.0f, 1.0f);

    std::vector<float> R((size_t) T * D), WN((size_t) T * D);
    for (auto &v : R) v = ux(rng);
    for (auto &v : WN) v = ux(rng);

    // Q8_0 weights [LR][D]: per 32-value block, d = amax/127, codes = round(x/d)
    const size_t rowb = q8::row_bytes(D);
    std::vector<uint8_t> q8w((size_t) LR * rowb, 0);
    std::vector<float> deq((size_t) LR * D);
    for (int p = 0; p < LR; ++p) {
        for (int b = 0; b < D / 32; ++b) {
            float x[32], amax = 0.0f;
            for (int i = 0; i < 32; ++i) { x[i] = ux(rng); amax = std::max(amax, std::fabs(x[i])); }
            const float d = amax / 127.0f;
            uint8_t *blk = q8w.data() + (size_t) p * rowb + (size_t) b * 34;
            uint16_t h = sycl::bit_cast<uint16_t>(sycl::half(d));
            std::memcpy(blk, &h, 2);
            for (int i = 0; i < 32; ++i) {
                const int c = (int) std::lround(x[i] / d);
                blk[2 + i] = (uint8_t) (int8_t) std::max(-127, std::min(127, c));
                deq[(size_t) p * D + (size_t) b * 32 + i] = d * (float) (int8_t) blk[2 + i];
            }
        }
    }

    sycl::queue q;
    float *dR = sycl::malloc_device<float>((size_t) T * D, q);
    float *dWN = sycl::malloc_device<float>((size_t) T * D, q);
    uint8_t *dQ8 = sycl::malloc_device<uint8_t>((size_t) LR * rowb, q);
    float *dPart = sycl::malloc_device<float>((size_t) T * NKC * PR, q);
    float *dSsg = sycl::malloc_device<float>((size_t) T * NKC, q);
    q.memcpy(dR, R.data(), sizeof(float) * R.size()).wait();
    q.memcpy(dWN, WN.data(), sizeof(float) * WN.size()).wait();
    q.memcpy(dQ8, q8w.data(), q8w.size()).wait();
    q.memset(dPart, 0, sizeof(float) * (size_t) T * NKC * PR).wait();
    q.memset(dSsg, 0, sizeof(float) * (size_t) T * NKC).wait();

    strata::kernels::FusedGrArgs a[T];
    for (int k = 0; k < T; ++k) {
        a[k].R = dR + (size_t) k * D;
        a[k].w_norm = dWN + (size_t) k * D;
        a[k].q8_down = dQ8;
    }
    strata::kernels::gr_down_q8_probe(a, T, dPart, dSsg, nullptr);

    std::vector<float> part((size_t) T * NKC * PR), ssg((size_t) T * NKC);
    q.memcpy(part.data(), dPart, sizeof(float) * part.size()).wait();
    q.memcpy(ssg.data(), dSsg, sizeof(float) * ssg.size()).wait();

    double worst = 0.0;
    int bad = 0;
    for (int k = 0; k < T; ++k) {
        for (int p = 0; p < LR; ++p) {
            // Reference in double, but the bound is relative to the L1 norm: the kernel sums each 640-term
            // chunk in fp32, so the meaningful error is round-off against the size of the terms, not against a
            // sum that may have cancelled.  (fp32 eps ~6e-8; ~sqrt(640) terms per chunk, 16 chunks.)
            double ref = 0.0, l1 = 0.0;
            for (int i = 0; i < D; ++i) {
                const double t = (double) deq[(size_t) p * D + i] * (double) R[(size_t) k * D + i] * (double) WN[(size_t) k * D + i];
                ref += t; l1 += std::fabs(t);
            }
            double got = 0.0;
            for (int kc = 0; kc < NKC; ++kc) got += (double) part[((size_t) k * NKC + kc) * PR + p];
            const double err = std::fabs(got - ref) / std::max(1e-12, l1);
            worst = std::max(worst, err);
            if (err > 5e-5) { if (bad < 6) std::printf("  row k=%d p=%d: ref=%.6g got=%.6g rel=%.2e\n", k, p, ref, got, err); ++bad; }
        }
        double ref_ss = 0.0;
        for (int i = 0; i < D; ++i) ref_ss += (double) R[(size_t) k * D + i] * R[(size_t) k * D + i];
        double got_ss = 0.0;
        for (int kc = 0; kc < NKC; ++kc) got_ss += (double) ssg[(size_t) k * NKC + kc];
        const double e = std::fabs(got_ss - ref_ss) / std::max(1.0, std::fabs(ref_ss));
        worst = std::max(worst, e);
        if (e > 1e-4) ++bad;
    }
    std::printf("q8 down parity: %s (%d of %d rows outside tolerance, worst |err|/L1 %.2e)\n",
                bad ? "FAIL" : "ok", bad, T * LR, worst);
    return bad ? 1 : 0;
}
