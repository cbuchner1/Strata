// q8_up_parity.cpp - gr_up_q8_kernel against a CPU reference.
//
// Three checks, because one would not be enough:
//   1. zero partials -> lo = 0 -> the gate is exactly 0.5, which validates rs, the R/w_norm indexing, the g
//      staging and the final mix, independently of any weight read.
//   2. one non-zero weight per row, at positions that straddle Q8_0 block and 128-value step boundaries
//      (v = 4*lane + 128*s).  Then the dot is exactly lo[r0], and a wrong block offset or a dropped step is
//      an O(1) difference rather than noise.
//   3. dense random weights: every output column against
//      mixed[j] = (1/hc) sum_c R[c*N+j]*w_norm[c*N+j]*rs_c*sigmoid(q8_up[c*N+j] . lo).
//      Bound relative to the L1 norm of the terms.  The dot is 320 fp32 products inside a sigmoid with lo in
//      the tens, so summation order alone moves this by ~1e-2; checks 1 and 2 are what pin the structure.
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
constexpr int KC = 640, NKC = D / KC, PR = LR + HC, T = 6, CPS = N / KC;

void quantise(const std::vector<float> &x, std::vector<uint8_t> &bytes, std::vector<float> &deq) {
    const size_t blocks = x.size() / 32;
    bytes.assign(blocks * 34, 0);
    deq.assign(x.size(), 0.0f);
    for (size_t b = 0; b < blocks; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
        const float d = amax / 127.0f;
        uint8_t *blk = bytes.data() + b * 34;
        uint16_t h = sycl::bit_cast<uint16_t>(sycl::half(d));
        std::memcpy(blk, &h, 2);
        for (int i = 0; i < 32; ++i) {
            const int c = (int) std::lround(x[b * 32 + i] / d);
            blk[2 + i] = (uint8_t) (int8_t) std::max(-127, std::min(127, c));
            deq[b * 32 + i] = d * (float) (int8_t) blk[2 + i];
        }
    }
}
}  // namespace

namespace strata::kernels {
void gr_down_q8_probe(const FusedGrArgs *args, int T, float *part, float *ssg, void *stream);
void gr_up_q8_probe(const FusedGrArgs *args, int T, const float *part, const float *ssg, void *stream);
}  // namespace strata::kernels

int main() {
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> ux(-1.0f, 1.0f);

    std::vector<float> R((size_t) T * D), WN((size_t) T * D), wd((size_t) LR * D), wu((size_t) D * LR);
    for (auto &v : R) v = ux(rng);
    for (auto &v : WN) v = ux(rng);
    for (auto &v : wd) v = ux(rng);
    for (auto &v : wu) v = ux(rng);
    std::vector<uint8_t> qd, qu;
    std::vector<float> deq_d, deq_u;
    quantise(wd, qd, deq_d);
    quantise(wu, qu, deq_u);

    sycl::queue q;
    float *dR = sycl::malloc_device<float>((size_t) T * D, q);
    float *dWN = sycl::malloc_device<float>((size_t) T * D, q);
    uint8_t *dQd = sycl::malloc_device<uint8_t>(qd.size(), q);
    uint8_t *dQu = sycl::malloc_device<uint8_t>(qu.size(), q);
    float *dPart = sycl::malloc_device<float>((size_t) T * NKC * PR, q);
    float *dSsg = sycl::malloc_device<float>((size_t) T * NKC, q);
    float *dMixed = sycl::malloc_device<float>((size_t) T * D, q);
    float *dRs = sycl::malloc_device<float>((size_t) T * HC, q);
    q.memcpy(dR, R.data(), sizeof(float) * R.size()).wait();
    q.memcpy(dWN, WN.data(), sizeof(float) * WN.size()).wait();
    q.memcpy(dQd, qd.data(), qd.size()).wait();
    q.memcpy(dQu, qu.data(), qu.size()).wait();
    q.memset(dPart, 0, sizeof(float) * (size_t) T * NKC * PR).wait();
    q.memset(dSsg, 0, sizeof(float) * (size_t) T * NKC).wait();

    strata::kernels::FusedGrArgs a[T];
    for (int k = 0; k < T; ++k) {
        a[k].R = dR + (size_t) k * D;
        a[k].w_norm = dWN + (size_t) k * D;
        a[k].q8_down = dQd;
        a[k].q8_up = dQu;
        a[k].mixed = dMixed + (size_t) k * D;
        a[k].rs = dRs + (size_t) k * HC;
        a[k].eps = 1e-6f;
    }

    std::vector<float> part((size_t) T * NKC * PR), ssg((size_t) T * NKC), mixed((size_t) T * D);
    auto pull = [&] {
        q.memcpy(part.data(), dPart, sizeof(float) * part.size()).wait();
        q.memcpy(ssg.data(), dSsg, sizeof(float) * ssg.size()).wait();
        q.memcpy(mixed.data(), dMixed, sizeof(float) * mixed.size()).wait();
    };
    auto lo_ref = [&](int k, std::vector<float> &lo, std::vector<double> &rs) {
        lo.assign(LR, 0.0f);
        rs.assign(HC, 0.0);
        for (int c = 0; c < HC; ++c) {
            double ss = 0.0;
            for (int j = 0; j < CPS; ++j) ss += ssg[(size_t) k * NKC + c * CPS + j];
            rs[c] = 1.0 / std::sqrt(ss / (double) N + (double) a[k].eps);
        }
        for (int r = 0; r < LR; ++r) {
            double sum = 0.0;
            for (int c = 0; c < HC; ++c) {
                double p = 0.0;
                for (int j = 0; j < CPS; ++j) p += part[((size_t) k * NKC + c * CPS + j) * PR + r];
                sum += rs[c] * p;
            }
            const double x = sum / HC;
            lo[r] = (float) (x / (1.0 + std::exp(-x)));
        }
    };

    // ---- 1. zero partials: gate = 0.5 exactly
    {
        std::vector<float> ones((size_t) T * NKC, 1.0f);
        q.memset(dPart, 0, sizeof(float) * (size_t) T * NKC * PR).wait();
        q.memcpy(dSsg, ones.data(), sizeof(float) * ones.size()).wait();
        strata::kernels::gr_up_q8_probe(a, T, dPart, dSsg, nullptr);
        pull();
        std::vector<float> rs0((size_t) T * HC);
        q.memcpy(rs0.data(), dRs, sizeof(float) * rs0.size()).wait();
        double worst = 0.0;
        for (int k = 0; k < T; ++k)
            for (int j = 0; j < N; ++j) {
                double ref = 0.0;
                for (int c = 0; c < HC; ++c) {
                    const size_t i = (size_t) c * N + j;
                    ref += (double) R[(size_t) k * D + i] * (double) WN[(size_t) k * D + i] *
                           rs0[(size_t) k * HC + c] * 0.5;
                }
                ref /= HC;
                worst = std::max(worst, std::fabs(mixed[(size_t) k * D + j] - ref) / std::max(1e-6, std::fabs(ref)));
            }
        std::printf("check 1 (zero partials, gate 0.5): %s (worst rel %.2e)\n", worst < 1e-3 ? "ok" : "FAIL", worst);
        if (worst >= 1e-3) return 1;
    }

    // ---- 2. one non-zero weight per row, at block and step boundaries
    {
        int bad = 0;
        double worst = 0.0;
        for (int r0 : {0, 31, 32, 160, 287, 288, 319}) {
            std::vector<uint8_t> qu1((size_t) D * (LR / 32) * 34, 0);
            for (int row = 0; row < D; ++row) {
                uint8_t *blk = qu1.data() + (size_t) row * (LR / 32) * 34 + (size_t) (r0 / 32) * 34;
                uint16_t h = sycl::bit_cast<uint16_t>(sycl::half(1.0f / 127.0f));
                std::memcpy(blk, &h, 2);
                blk[2 + (r0 % 32)] = 127;
            }
            q.memcpy(dQu, qu1.data(), qu1.size()).wait();
            strata::kernels::gr_down_q8_probe(a, T, dPart, dSsg, nullptr);
            strata::kernels::gr_up_q8_probe(a, T, dPart, dSsg, nullptr);
            pull();
            for (int k = 0; k < T; ++k) {
                std::vector<float> lo;
                std::vector<double> rs;
                lo_ref(k, lo, rs);
                const double sig = 1.0 / (1.0 + std::exp(-(double) lo[r0]));
                for (int j = 0; j < N; ++j) {
                    double ref = 0.0;
                    for (int c = 0; c < HC; ++c) {
                        const size_t i = (size_t) c * N + j;
                        ref += (double) R[(size_t) k * D + i] * (double) WN[(size_t) k * D + i] * rs[c] * sig;
                    }
                    ref /= HC;
                    const double e = std::fabs(mixed[(size_t) k * D + j] - ref) / std::max(1e-9, std::fabs(ref));
                    worst = std::max(worst, e);
                    if (e > 1e-3) ++bad;
                }
            }
        }
        std::printf("check 2 (one weight at r0 in {0,31,32,160,287,288,319}): %s (worst rel %.2e)\n",
                    bad ? "FAIL" : "ok", worst);
        if (bad) return 1;
    }

    // ---- 3. dense random weights (restore the real up weights first: check 2 overwrote them)
    q.memcpy(dQu, qu.data(), qu.size()).wait();
    strata::kernels::gr_down_q8_probe(a, T, dPart, dSsg, nullptr);
    strata::kernels::gr_up_q8_probe(a, T, dPart, dSsg, nullptr);
    pull();
    {
        int bad = 0;
        double worst = 0.0;
        for (int k = 0; k < T; ++k) {
            std::vector<float> lo;
            std::vector<double> rs;
            lo_ref(k, lo, rs);
            for (int j = 0; j < N; ++j) {
                double ref = 0.0, l1 = 0.0;
                for (int c = 0; c < HC; ++c) {
                    const size_t i = (size_t) c * N + j;
                    float mine = 0.0f;
                    for (int r = 0; r < LR; ++r) mine = std::fma(deq_u[i * LR + r], lo[r], mine);
                    const double term = (double) R[(size_t) k * D + i] * (double) WN[(size_t) k * D + i] * rs[c] /
                                        (1.0 + std::exp(-(double) mine));
                    ref += term;
                    l1 += std::fabs(term);
                }
                ref /= HC;
                const double err = std::fabs(mixed[(size_t) k * D + j] - ref) / std::max(1e-12, l1);
                worst = std::max(worst, err);
                if (err > 2e-2) ++bad;
            }
        }
        std::printf("check 3 (dense weights): %s (%d of %d cols, worst |err|/L1 %.2e)\n",
                    bad ? "FAIL" : "ok", bad, T * N, worst);
        if (bad) return 1;
    }
    std::printf("q8 up parity: ok\n");
    return 0;
}
