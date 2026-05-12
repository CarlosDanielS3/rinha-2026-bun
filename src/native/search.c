/*
 * AVX2-optimized IVF search for 14-dim int16 quantized vectors.
 * nprobe=1 + bounding-box repair for exact-quality search.
 * Compiled: gcc -O3 -march=haswell -shared -fPIC -o libsearch.so search.c
 */
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define DIMS 14
#define K 5
#define QUANT_SCALE 10000

/* ── Index state (set once at init) ────────────────────────────── */

static int g_nc, g_nv;
static float* g_cent;              /* numClusters × 16 floats, padded */
static const int16_t* g_bmin;     /* bbox min: nc × dims */
static const int16_t* g_bmax;     /* bbox max: nc × dims */
static const uint32_t* g_off;     /* cluster offsets */
static const int16_t* g_vec;      /* quantized vectors (14 per row) */
static const uint8_t*  g_lab;     /* labels */

/* ── SIMD helpers ──────────────────────────────────────────────── */

static inline float hsum8(__m256 v) {
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 s   = _mm_add_ps(lo, hi);
    __m128 shf = _mm_movehdup_ps(s);
    s = _mm_add_ps(s, shf);
    shf = _mm_movehl_ps(shf, s);
    return _mm_cvtss_f32(_mm_add_ss(s, shf));
}

/* ── Max-heap (k=5) ───────────────────────────────────────────── */

static inline void sift(uint64_t* hd, uint8_t* hl) {
    int i = 0;
    for (;;) {
        int lg = i, l = 2*i+1, r = l+1;
        if (l < K && hd[l] > hd[lg]) lg = l;
        if (r < K && hd[r] > hd[lg]) lg = r;
        if (lg == i) return;
        uint64_t t = hd[i]; hd[i] = hd[lg]; hd[lg] = t;
        uint8_t  u = hl[i]; hl[i] = hl[lg]; hl[lg] = u;
        i = lg;
    }
}

static inline void build5(uint64_t* d, uint8_t* l) {
    int lg = 1;
    if (d[3] > d[lg]) lg = 3;
    if (d[4] > d[lg]) lg = 4;
    if (lg != 1) {
        uint64_t t = d[1]; d[1] = d[lg]; d[lg] = t;
        uint8_t  u = l[1]; l[1] = l[lg]; l[lg] = u;
    }
    lg = 0;
    if (d[1] > d[lg]) lg = 1;
    if (d[2] > d[lg]) lg = 2;
    if (lg != 0) {
        uint64_t t = d[0]; d[0] = d[lg]; d[lg] = t;
        uint8_t  u = l[0]; l[0] = l[lg]; l[lg] = u;
        if (lg == 1) {
            int lg2 = 1;
            if (d[3] > d[lg2]) lg2 = 3;
            if (d[4] > d[lg2]) lg2 = 4;
            if (lg2 != 1) {
                t = d[1]; d[1] = d[lg2]; d[lg2] = t;
                u = l[1]; l[1] = l[lg2]; l[lg2] = u;
            }
        }
    }
}

/* ── Bbox lower bound ──────────────────────────────────────────── */

static inline uint64_t bbox_lower_bound(
    uint32_t cluster,
    const int16_t q[DIMS],
    uint64_t stop_after
) {
    const int16_t* bmin = g_bmin + (size_t)cluster * DIMS;
    const int16_t* bmax = g_bmax + (size_t)cluster * DIMS;
    uint64_t sum = 0;
    for (int d = 0; d < DIMS; d++) {
        int32_t target = q[d];
        int32_t delta = 0;
        if (target < bmin[d]) delta = target - bmin[d];
        else if (target > bmax[d]) delta = target - bmax[d];
        sum += (uint64_t)(delta * delta);
        if (sum > stop_after) return sum;
    }
    return sum;
}

/* ── Cluster scanner ───────────────────────────────────────────── */

static inline void scan_cluster(
    uint32_t cid,
    const int16_t qi[DIMS],
    uint64_t* hd, uint8_t* hl, int* hsz, uint64_t* tau
) {
    uint32_t st = g_off[cid], en = g_off[cid + 1];

    for (uint32_t i = st; i < en; i++) {
        const int16_t* vp = &g_vec[(size_t)i * DIMS];

        if (i + 4 < en)
            __builtin_prefetch(&g_vec[(size_t)(i + 4) * DIMS], 0, 0);

        /* Compute int16 L2 distance with early exit */
        int32_t d0 = (int32_t)qi[0] - vp[0], d1 = (int32_t)qi[1] - vp[1];
        int32_t d2 = (int32_t)qi[2] - vp[2], d3 = (int32_t)qi[3] - vp[3];
        int32_t d4 = (int32_t)qi[4] - vp[4], d5 = (int32_t)qi[5] - vp[5];
        int32_t d6 = (int32_t)qi[6] - vp[6], d7 = (int32_t)qi[7] - vp[7];

        uint64_t partial = (uint64_t)(d0*d0) + (uint64_t)(d1*d1) +
                           (uint64_t)(d2*d2) + (uint64_t)(d3*d3) +
                           (uint64_t)(d4*d4) + (uint64_t)(d5*d5) +
                           (uint64_t)(d6*d6) + (uint64_t)(d7*d7);

        if (*hsz == K && partial > *tau) continue;

        int32_t d8  = (int32_t)qi[8]  - vp[8],  d9  = (int32_t)qi[9]  - vp[9];
        int32_t d10 = (int32_t)qi[10] - vp[10], d11 = (int32_t)qi[11] - vp[11];
        int32_t d12 = (int32_t)qi[12] - vp[12], d13 = (int32_t)qi[13] - vp[13];

        uint64_t dist = partial + (uint64_t)(d8*d8)   + (uint64_t)(d9*d9) +
                                  (uint64_t)(d10*d10) + (uint64_t)(d11*d11) +
                                  (uint64_t)(d12*d12) + (uint64_t)(d13*d13);

        if (*hsz < K) {
            hd[*hsz] = dist;
            hl[*hsz] = g_lab[i];
            (*hsz)++;
            if (*hsz == K) { build5(hd, hl); *tau = hd[0]; }
        } else if (dist < *tau) {
            hd[0] = dist;
            hl[0] = g_lab[i];
            sift(hd, hl);
            *tau = hd[0];
        }
    }
}

/* ── Public API ────────────────────────────────────────────────── */

void ivf_init(const uint8_t* buf, uint32_t sz) {
    uint32_t off = 0;
    g_nv = *(const uint32_t*)(buf + off); off += 4;
    g_nc = *(const uint32_t*)(buf + off); off += 4;
    off += 4;  /* dims (14) */
    off += 4;  /* scale (10000) */

    /* Centroids: nc × dims × float32 */
    const float* rc = (const float*)(buf + off);
    off += (uint32_t)((size_t)g_nc * DIMS * 4);

    /* Pad centroids to 16 floats each for aligned AVX loads */
    g_cent = (float*)aligned_alloc(32, (size_t)g_nc * 16 * sizeof(float));
    for (int c = 0; c < g_nc; c++) {
        memcpy(&g_cent[c * 16], &rc[c * DIMS], DIMS * sizeof(float));
        g_cent[c * 16 + 14] = g_cent[c * 16 + 15] = 0;
    }

    /* Bounding boxes: nc × dims × int16 each */
    g_bmin = (const int16_t*)(buf + off);
    off += (uint32_t)((size_t)g_nc * DIMS * 2);
    g_bmax = (const int16_t*)(buf + off);
    off += (uint32_t)((size_t)g_nc * DIMS * 2);

    /* Cluster offsets */
    g_off = (const uint32_t*)(buf + off);
    off += ((uint32_t)g_nc + 1) * 4;

    /* Vectors and labels */
    g_vec = (const int16_t*)(buf + off);
    off += (uint32_t)((size_t)g_nv * DIMS * 2);
    g_lab = (const uint8_t*)(buf + off);

    printf("  Native IVF: %d vectors, %d clusters (nprobe=1 + bbox_repair, AVX2)\n", g_nv, g_nc);
}

int ivf_search(const float* q) {
    float qp[16] __attribute__((aligned(32)));
    memcpy(qp, q, DIMS * sizeof(float));
    qp[14] = qp[15] = 0;

    __m256 q0 = _mm256_load_ps(qp);
    __m256 q1 = _mm256_load_ps(qp + 8);

    /* ── Step 1: find nearest centroid (SIMD) ── */
    float bestDist = 1e30f;
    int bestCluster = 0;
    for (int c = 0; c < g_nc; c++) {
        const float* cp = &g_cent[c * 16];
        __m256 d0 = _mm256_sub_ps(q0, _mm256_load_ps(cp));
        __m256 d1 = _mm256_sub_ps(q1, _mm256_load_ps(cp + 8));
        float dist = hsum8(_mm256_add_ps(_mm256_mul_ps(d0, d0),
                                          _mm256_mul_ps(d1, d1)));
        if (dist < bestDist) {
            bestDist = dist;
            bestCluster = c;
        }
    }

    /* ── Step 2: quantize query to int16 ── */
    int16_t qi[DIMS];
    for (int d = 0; d < DIMS; d++) {
        long r = lroundf(q[d] * QUANT_SCALE);
        if (r < -32768) r = -32768;
        if (r > 32767) r = 32767;
        qi[d] = (int16_t)r;
    }

    /* ── Step 3: scan nearest cluster ── */
    uint64_t hd[K]; for (int i=0;i<K;i++) hd[i]=UINT64_MAX;
    uint8_t  hl[K] = {0};
    int   hsz = 0;
    uint64_t tau = UINT64_MAX;

    scan_cluster(bestCluster, qi, hd, hl, &hsz, &tau);

    /* ── Step 4: bbox repair — scan any cluster whose bbox overlaps ── */
    for (int c = 0; c < g_nc; c++) {
        if (c == bestCluster) continue;
        if (g_off[c] == g_off[c + 1]) continue; /* empty cluster */
        if (bbox_lower_bound(c, qi, tau) <= tau) {
            scan_cluster(c, qi, hd, hl, &hsz, &tau);
        }
    }

    return hl[0] + hl[1] + hl[2] + hl[3] + hl[4];
}
