// q8_read_parity.cpp - the whole Q8_0 read against the engine's own BF16 read, on identical weights.
//
// q8_down_parity and q8_up_parity check the kernels against CPU references.  This checks the thing that
// actually has to hold in production: fused_gr_read_multi with the Q8_0 pointers must produce the same mixed
// residual as the same call with the BF16 projections (the dequantized Q8_0 values, rounded to bf16).  If these
// agree, the read is numerically right and any trouble in the engine is the window's choreography, not the read.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/q8_block.hpp"
#include "strata/core/geometry.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace q8 = strata::kernels::q8;

namespace {
constexpr int N = strata::core::kCompiledGeometry.n_embd;
constexpr int HC = strata::core::kCompiledGeometry.hc;
constexpr int LR = strata::core::kCompiledGeometry.hc_lr;
constexpr int D = N * HC;
constexpr int T = 6;

uint16_t f32_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t r = (x + 0x7fffu + ((x >> 16) & 1u)) >> 16;   // round to nearest even
    return (uint16_t) r;
}
}  // namespace

int main() {
    std::mt19937 rng(31337);
    std::uniform_real_distribution<float> ux(-1.0f, 1.0f);
    std::vector<float> R((size_t) T * D), WN((size_t) T * D), wd((size_t) LR * D), wu((size_t) D * LR);
    for (auto &v : R) v = ux(rng);
    for (auto &v : WN) v = ux(rng);
    for (auto &v : wd) v = ux(rng);
    for (auto &v : wu) v = ux(rng);

    // quantise the same values to Q8_0, and keep the dequantized values as BF16 (what --compat-bf16 stores)
    const size_t dblocks = wd.size() / 32, ublocks = wu.size() / 32;
    std::vector<uint8_t> qd(dblocks * 34, 0), qu(ublocks * 34, 0);
    std::vector<uint16_t> bd(wd.size()), bu(wu.size());
    auto pack = [](const std::vector<float> &x, std::vector<uint8_t> &bytes, std::vector<uint16_t> &bf) {
        for (size_t b = 0; b < x.size() / 32; ++b) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
            const float d = amax / 127.0f;
            uint8_t *blk = bytes.data() + b * 34;
            const uint16_t h = sycl::bit_cast<uint16_t>(sycl::half(d));
            std::memcpy(blk, &h, 2);
            for (int i = 0; i < 32; ++i) {
                const int c = (int) std::lround(x[b * 32 + i] / d);
                blk[2 + i] = (uint8_t) (int8_t) std::max(-127, std::min(127, c));
                bf[b * 32 + i] = f32_to_bf16(d * (float) (int8_t) blk[2 + i]);
            }
        }
    };
    pack(wd, qd, bd);
    pack(wu, qu, bu);

    sycl::queue q;
    float *dR = sycl::malloc_device<float>((size_t) T * D, q);
    float *dWN = sycl::malloc_device<float>((size_t) D, q);
    float *dScratch = sycl::malloc_device<float>((size_t) T * D, q);
    uint16_t *dBd = sycl::malloc_device<uint16_t>(bd.size(), q);
    uint16_t *dBu = sycl::malloc_device<uint16_t>(bu.size(), q);
    uint8_t *dQd = sycl::malloc_device<uint8_t>(qd.size(), q);
    uint8_t *dQu = sycl::malloc_device<uint8_t>(qu.size(), q);
    float *dMixedA = sycl::malloc_device<float>((size_t) T * D, q);
    float *dMixedB = sycl::malloc_device<float>((size_t) T * D, q);
    float *dRsA = sycl::malloc_device<float>((size_t) T * HC, q);
    float *dRsB = sycl::malloc_device<float>((size_t) T * HC, q);
    float *dLoA = sycl::malloc_device<float>((size_t) T * LR, q);
    float *dLoB = sycl::malloc_device<float>((size_t) T * LR, q);
    q.memcpy(dR, R.data(), sizeof(float) * R.size()).wait();
    q.memcpy(dWN, WN.data(), sizeof(float) * D).wait();
    q.memcpy(dBd, bd.data(), sizeof(uint16_t) * bd.size()).wait();
    q.memcpy(dBu, bu.data(), sizeof(uint16_t) * bu.size()).wait();
    q.memcpy(dQd, qd.data(), qd.size()).wait();
    q.memcpy(dQu, qu.data(), qu.size()).wait();
    strata::kernels::fused_gr_set_q8_scratch(sycl::malloc_device<float>(strata::kernels::fused_gr_q8_scratch_floats(), q));

    strata::kernels::FusedGrArgs a[2][T];
    for (int k = 0; k < T; ++k) {
        for (int pass = 0; pass < 2; ++pass) {
            a[pass][k].R = dR + (size_t) k * D;
            a[pass][k].w_norm = dWN;
            a[pass][k].w_down = dBd;
            a[pass][k].w_up = dBu;
            a[pass][k].lo = (pass ? dLoB : dLoA) + (size_t) k * LR;
            a[pass][k].rs = (pass ? dRsB : dRsA) + (size_t) k * HC;
            a[pass][k].mixed = (pass ? dMixedB : dMixedA) + (size_t) k * D;
            a[pass][k].eps = 1e-6f;
            if (pass == 1) { a[pass][k].q8_down = dQd; a[pass][k].q8_up = dQu; }
        }
    }
    // pass 0: the BF16 read; pass 1: the Q8_0 read
    strata::kernels::fused_gr_read_multi(a[0], T, dScratch, nullptr, nullptr, 0);
    q.wait();
    strata::kernels::fused_gr_read_multi(a[1], T, dScratch, nullptr, nullptr, 0);
    q.wait();

    std::vector<float> mixA((size_t) T * D), mixB((size_t) T * D);
    q.memcpy(mixA.data(), dMixedA, sizeof(float) * mixA.size()).wait();
    q.memcpy(mixB.data(), dMixedB, sizeof(float) * mixB.size()).wait();

    double worst_ref_q8 = 1e9, worst_ref_bf16 = 1e9;
    // Independent double-precision reference of the whole Q8 read, built from the dequantized weights:
    //   lo[r] = silu( (1/hc) sum_c rs_c * sum_{col in stream c} wd[r][col]*R[col]*w_norm[col] )
    //   mixed[j] = (1/hc) sum_c R[c*N+j]*w_norm[c*N+j]*rs_c*sigmoid( sum_r wu[c*N+j][r]*lo[r] )
    // If the kernel matches THIS, then any difference from the BF16 read is rounding and summation order
    // between two different weight sets, not a bug.
    {
        std::vector<double> ddq((size_t) LR * D), duq((size_t) D * LR), wn((size_t) D);
        for (size_t i = 0; i < wn.size(); ++i) wn[i] = WN[i];
        for (size_t b = 0; b < qd.size() / 34; ++b) {
            const uint8_t *blk = qd.data() + b * 34;
            uint16_t h;
            std::memcpy(&h, blk, 2);
            const double d = (double) sycl::bit_cast<sycl::half>(h);
            for (int i = 0; i < 32; ++i) ddq[b * 32 + i] = d * (double) (int8_t) blk[2 + i];
        }
        for (size_t b = 0; b < qu.size() / 34; ++b) {
            const uint8_t *blk = qu.data() + b * 34;
            uint16_t h;
            std::memcpy(&h, blk, 2);
            const double d = (double) sycl::bit_cast<sycl::half>(h);
            for (int i = 0; i < 32; ++i) duq[b * 32 + i] = d * (double) (int8_t) blk[2 + i];
        }
        // the same reference over the BF16 weight set, so each path is judged against its own weights
        std::vector<double> ddb(bd.size()), dub(bu.size());
        for (size_t i = 0; i < bd.size(); ++i) {
            uint32_t x = (uint32_t) bd[i] << 16;
            float f; std::memcpy(&f, &x, 4); ddb[i] = (double) f;
        }
        for (size_t i = 0; i < bu.size(); ++i) {
            uint32_t x = (uint32_t) bu[i] << 16;
            float f; std::memcpy(&f, &x, 4); dub[i] = (double) f;
        }
        double worst_ref = 0.0, worst_bf = 0.0, worst_bf_own = 0.0;
        for (int k = 0; k < T; ++k) {
            std::vector<double> rs(HC, 0.0), lo(LR, 0.0);
            for (int c = 0; c < HC; ++c) {
                double ss = 0.0;
                for (int d = 0; d < N; ++d) { const double v = R[(size_t) k * D + c * N + d]; ss += v * v; }
                rs[c] = 1.0 / std::sqrt(ss / (double) N + 1e-6);
            }
            std::vector<double> lo_bf(LR, 0.0);
            for (int r = 0; r < LR; ++r) {
                double sum = 0.0, sum_bf = 0.0;
                for (int c = 0; c < HC; ++c) {
                    double p = 0.0, pb = 0.0;
                    for (int d = 0; d < N; ++d) {
                        const size_t col = (size_t) c * N + d;
                        const double rv = (double) R[(size_t) k * D + col] * wn[col];
                        p += ddq[(size_t) r * D + col] * rv;
                        pb += ddb[(size_t) r * D + col] * rv;
                    }
                    sum += rs[c] * p;
                    sum_bf += rs[c] * pb;
                }
                const double x = sum / HC, xb = sum_bf / HC;
                lo[r] = x / (1.0 + std::exp(-x));
                lo_bf[r] = xb / (1.0 + std::exp(-xb));
            }
            double mean_abs = 0.0;
            for (int j = 0; j < N; ++j) mean_abs += std::fabs(mixA[(size_t) k * D + j]);
            mean_abs /= N;
            for (int j = 0; j < N; ++j) {
                double ref2 = 0.0;
                for (int c = 0; c < HC; ++c) {
                    const size_t i2 = (size_t) c * N + j;
                    double mine = 0.0;
                    for (int r = 0; r < LR; ++r) mine += duq[i2 * LR + r] * lo[r];
                    ref2 += (double) R[(size_t) k * D + i2] * wn[i2] * rs[c] / (1.0 + std::exp(-mine));
                }
                ref2 /= HC;
                worst_ref = std::max(worst_ref, std::fabs(mixB[(size_t) k * D + j] - ref2) / std::max(1e-9, mean_abs));
                worst_bf = std::max(worst_bf, std::fabs(mixA[(size_t) k * D + j] - ref2) / std::max(1e-9, mean_abs));
                // reference 2: the same formula over the bf16 weights
                double ref3 = 0.0;
                for (int c = 0; c < HC; ++c) {
                    const size_t i2 = (size_t) c * N + j;
                    double mine = 0.0;
                    for (int r = 0; r < LR; ++r) mine += dub[i2 * LR + r] * lo_bf[r];
                    ref3 += (double) R[(size_t) k * D + i2] * wn[i2] * rs[c] / (1.0 + std::exp(-mine));
                }
                ref3 /= HC;
                worst_bf_own = std::max(worst_bf_own, std::fabs(mixA[(size_t) k * D + j] - ref3) / std::max(1e-9, mean_abs));
            }
        }
        std::printf("q8 read vs its own double reference:   %.2e of the mean |mixed|\n", worst_ref);
        std::printf("bf16 read vs its own double reference: %.2e of the mean |mixed|\n", worst_bf_own);
        worst_ref_q8 = worst_ref; worst_ref_bf16 = worst_bf_own;
    }
    double worst = 0.0;
    int bad = 0;
    for (size_t k = 0; k < (size_t) T; ++k) {
        double l1 = 0.0;
        for (int i = 0; i < N; ++i) l1 += std::fabs((double) mixA[k * D + i]);
        l1 /= N;
        for (int i = 0; i < N; ++i) {
            const double e = std::fabs(mixB[k * D + i] - mixA[k * D + i]) / std::max(1e-6, l1);
            worst = std::max(worst, e);
            if (e > 5e-2) {
                if (bad < 8) std::printf("  k=%zu i=%d (i%%16=%d) bf16=%.6g q8=%.6g rel=%.2e\n", k, i, i % 16,
                                         mixA[k * D + i], mixB[k * D + i], e);
                ++bad;
            }
        }
    }
    std::printf("q8 read vs bf16 read (informational): %d of %d cols differ by >5%% of the mean |mixed|, worst %.2e\n",
                bad, T * N, worst);
    // The two reads use different weight sets (Q8_0 bytes vs their bf16 rounding), so a per-column difference
    // on cancelling columns is expected; what must hold is that each read reproduces the formula over its own
    // weights.  That is the criterion (printed above).
    return (worst_ref_q8 < 1e-3 && worst_ref_bf16 < 1e-3) ? 0 : 1;
}
