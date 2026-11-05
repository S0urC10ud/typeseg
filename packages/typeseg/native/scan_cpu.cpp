#define PY_SSIZE_T_CLEAN
#define ORT_API_MANUAL_INIT
#include <Python.h>
#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>

#if defined(__SSE2__) || defined(_M_X64)
#include <immintrin.h>
#ifdef _WIN32
#include <intrin.h>
#endif
#define TYPESEG_SSE2 1
#if defined(__GNUC__) || defined(__clang__)
#define TYPESEG_WIDE __attribute__((target("avx512f"), noinline))
#define TYPESEG_WIDE_INLINE __attribute__((target("avx512f"), always_inline)) inline
#else
#define TYPESEG_WIDE __declspec(noinline)
#define TYPESEG_WIDE_INLINE __forceinline
#endif
#endif

namespace {

// Range reduction followed by a degree-seven expansion on [-log(2)/2, log(2)/2].
// Mamba's decay exponent is nonpositive. Keep subnormal values on the libm path.
float decay_exp(float x) {
    if (x < -87.0f || !std::isfinite(x) || x > 0.0f) {
        return std::exp(x);
    }
    const float scaled = x * 1.4426950408889634f + 0.5f;
    int exponent = static_cast<int>(scaled);
    exponent -= static_cast<float>(exponent) > scaled;
    float r = x - static_cast<float>(exponent) * 0.693359375f;
    r -= static_cast<float>(exponent) * -0.00021219444005469058f;
    float p = 1.0f / 5040.0f;
    p = 1.0f / 720.0f + r * p;
    p = 1.0f / 120.0f + r * p;
    p = 1.0f / 24.0f + r * p;
    p = 1.0f / 6.0f + r * p;
    p = 0.5f + r * p;
    p = 1.0f + r * p;
    p = 1.0f + r * p;
    const uint32_t bits = static_cast<uint32_t>(exponent + 127) << 23;
    float scale;
    std::memcpy(&scale, &bits, sizeof(scale));
    return p * scale;
}

#ifdef TYPESEG_SSE2
#ifdef _WIN32
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("xsave"), noinline))
#endif
uint64_t saved_cpu_state() {
    return _xgetbv(0);
}
#endif

bool wide_scan_available() {
    static const bool available = [] {
#ifndef _WIN32
        __builtin_cpu_init();
        return bool(__builtin_cpu_supports("avx512f"));
#else
        int registers[4];
        __cpuid(registers, 0);
        if (registers[0] < 7) return false;
        __cpuidex(registers, 1, 0);
        if ((registers[2] & (1 << 27)) == 0 || (saved_cpu_state() & 0xe6) != 0xe6) return false;
        __cpuidex(registers, 7, 0);
        return (registers[1] & (1 << 16)) != 0;
#endif
    }();
    return available;
}

TYPESEG_WIDE_INLINE __m512 decay_exp16(__m512 x) {
    const __m512 zero = _mm512_setzero_ps();
    const __m512 lower = _mm512_set1_ps(-87.0f);
    const __mmask16 exceptional = _mm512_cmp_ps_mask(x, lower, _CMP_LT_OQ)
        | _mm512_cmp_ps_mask(x, zero, _CMP_NLE_UQ);
    const __m512 bounded = _mm512_min_ps(_mm512_max_ps(x, lower), zero);
    const __m512 scaled = _mm512_add_ps(
        _mm512_mul_ps(bounded, _mm512_set1_ps(1.4426950408889634f)), _mm512_set1_ps(0.5f));
    __m512i exponent = _mm512_cvttps_epi32(scaled);
    const __mmask16 correction = _mm512_cmp_ps_mask(_mm512_cvtepi32_ps(exponent), scaled, _CMP_GT_OQ);
    exponent = _mm512_mask_sub_epi32(exponent, correction, exponent, _mm512_set1_epi32(1));
    const __m512 exponent_float = _mm512_cvtepi32_ps(exponent);
    __m512 r = _mm512_sub_ps(bounded, _mm512_mul_ps(exponent_float, _mm512_set1_ps(0.693359375f)));
    r = _mm512_sub_ps(r, _mm512_mul_ps(exponent_float, _mm512_set1_ps(-0.00021219444005469058f)));
    __m512 p = _mm512_set1_ps(1.0f / 5040.0f);
    p = _mm512_add_ps(_mm512_set1_ps(1.0f / 720.0f), _mm512_mul_ps(r, p));
    p = _mm512_add_ps(_mm512_set1_ps(1.0f / 120.0f), _mm512_mul_ps(r, p));
    p = _mm512_add_ps(_mm512_set1_ps(1.0f / 24.0f), _mm512_mul_ps(r, p));
    p = _mm512_add_ps(_mm512_set1_ps(1.0f / 6.0f), _mm512_mul_ps(r, p));
    p = _mm512_add_ps(_mm512_set1_ps(0.5f), _mm512_mul_ps(r, p));
    p = _mm512_add_ps(_mm512_set1_ps(1.0f), _mm512_mul_ps(r, p));
    p = _mm512_add_ps(_mm512_set1_ps(1.0f), _mm512_mul_ps(r, p));
    const __m512 scale = _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_add_epi32(exponent, _mm512_set1_epi32(127)), 23));
    __m512 result = _mm512_mul_ps(p, scale);
    if (exceptional) {
        float inputs[16], outputs[16];
        _mm512_storeu_ps(inputs, x);
        _mm512_storeu_ps(outputs, result);
        for (int lane = 0; lane < 16; ++lane) {
            if (exceptional & (1 << lane)) outputs[lane] = std::exp(inputs[lane]);
        }
        result = _mm512_loadu_ps(outputs);
    }
    return result;
}

__m128 decay_exp4(__m128 x) {
    const __m128 zero = _mm_setzero_ps();
    const __m128 lower = _mm_set1_ps(-87.0f);
    const int exceptional = _mm_movemask_ps(_mm_or_ps(
        _mm_cmplt_ps(x, lower), _mm_cmpnle_ps(x, zero)));
    const __m128 bounded = _mm_min_ps(_mm_max_ps(x, lower), zero);
    const __m128 scaled = _mm_add_ps(
        _mm_mul_ps(bounded, _mm_set1_ps(1.4426950408889634f)), _mm_set1_ps(0.5f));
    __m128i exponent = _mm_cvttps_epi32(scaled);
    const __m128i correction = _mm_and_si128(
        _mm_castps_si128(_mm_cmpgt_ps(_mm_cvtepi32_ps(exponent), scaled)),
        _mm_set1_epi32(1));
    exponent = _mm_sub_epi32(exponent, correction);
    const __m128 exponent_float = _mm_cvtepi32_ps(exponent);
    __m128 r = _mm_sub_ps(bounded, _mm_mul_ps(exponent_float, _mm_set1_ps(0.693359375f)));
    r = _mm_sub_ps(r, _mm_mul_ps(exponent_float, _mm_set1_ps(-0.00021219444005469058f)));
    __m128 p = _mm_set1_ps(1.0f / 5040.0f);
    p = _mm_add_ps(_mm_set1_ps(1.0f / 720.0f), _mm_mul_ps(r, p));
    p = _mm_add_ps(_mm_set1_ps(1.0f / 120.0f), _mm_mul_ps(r, p));
    p = _mm_add_ps(_mm_set1_ps(1.0f / 24.0f), _mm_mul_ps(r, p));
    p = _mm_add_ps(_mm_set1_ps(1.0f / 6.0f), _mm_mul_ps(r, p));
    p = _mm_add_ps(_mm_set1_ps(0.5f), _mm_mul_ps(r, p));
    p = _mm_add_ps(_mm_set1_ps(1.0f), _mm_mul_ps(r, p));
    p = _mm_add_ps(_mm_set1_ps(1.0f), _mm_mul_ps(r, p));
    const __m128 scale = _mm_castsi128_ps(_mm_slli_epi32(
        _mm_add_epi32(exponent, _mm_set1_epi32(127)), 23));
    __m128 result = _mm_mul_ps(p, scale);
    if (exceptional) {
        float inputs[4], outputs[4];
        _mm_storeu_ps(inputs, x);
        _mm_storeu_ps(outputs, result);
        for (int lane = 0; lane < 4; ++lane) {
            if (exceptional & (1 << lane)) {
                outputs[lane] = std::exp(inputs[lane]);
            }
        }
        result = _mm_loadu_ps(outputs);
    }
    return result;
}
#endif

struct ScanWork {
    const float *dt, *u, *b, *c, *a, *d;
    float* y;
    int64_t length, channels;
    bool reverse;

#ifdef TYPESEG_SSE2
    TYPESEG_WIDE void RunWide(size_t channel) const;
#endif
    void Run(size_t channel) const {
#ifdef TYPESEG_SSE2
        if (wide_scan_available()) {
            RunWide(channel);
            return;
        }
        __m128 memory[4] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()};
        __m128 decay_weights[4];
        for (size_t group = 0; group < 4; ++group) {
            decay_weights[group] = _mm_loadu_ps(a + channel * 16 + group * 4);
        }
        for (int64_t index = 0; index < length; ++index) {
            const int64_t t = reverse ? length - index - 1 : index;
            const int64_t pos = t * channels + channel;
            const __m128 step = _mm_set1_ps(dt[pos]);
            const __m128 input = _mm_set1_ps(u[pos]);
            float products[16];
            for (size_t group = 0; group < 4; ++group) {
                const __m128 decay = decay_exp4(_mm_mul_ps(step, decay_weights[group]));
                const __m128 addition = _mm_mul_ps(input, _mm_mul_ps(
                    step, _mm_loadu_ps(b + t * 16 + group * 4)));
                memory[group] = _mm_add_ps(_mm_mul_ps(decay, memory[group]), addition);
                _mm_storeu_ps(products + group * 4, _mm_mul_ps(
                    memory[group], _mm_loadu_ps(c + t * 16 + group * 4)));
            }
            // Keep the original reduction order while vectorizing the state updates.
            float result = 0.0f;
            for (float product : products) {
                result += product;
            }
            y[pos] = result + u[pos] * d[channel];
        }
#else
        std::array<float, 16> memory{};
        for (int64_t index = 0; index < length; ++index) {
            const int64_t t = reverse ? length - index - 1 : index;
            const int64_t pos = t * channels + channel;
            const float step = dt[pos];
            const float input = u[pos];
            float result = 0.0f;
            for (size_t state = 0; state < memory.size(); ++state) {
                const float decay = decay_exp(step * a[channel * 16 + state]);
                const float addition = input * (step * b[t * 16 + state]);
                memory[state] = decay * memory[state] + addition;
                result += memory[state] * c[t * 16 + state];
            }
            y[pos] = result + input * d[channel];
        }
#endif
    }
};

#ifdef TYPESEG_SSE2
TYPESEG_WIDE void ScanWork::RunWide(size_t channel) const {
    __m512 memory = _mm512_setzero_ps();
    const __m512 decay_weights = _mm512_loadu_ps(a + channel * 16);
    for (int64_t index = 0; index < length; ++index) {
        const int64_t t = reverse ? length - index - 1 : index;
        const int64_t pos = t * channels + channel;
        const __m512 step = _mm512_set1_ps(dt[pos]);
        const __m512 input = _mm512_set1_ps(u[pos]);
        const __m512 decay = decay_exp16(_mm512_mul_ps(step, decay_weights));
        const __m512 addition = _mm512_mul_ps(input, _mm512_mul_ps(step, _mm512_loadu_ps(b + t * 16)));
        memory = _mm512_add_ps(_mm512_mul_ps(decay, memory), addition);
        float products[16];
        _mm512_storeu_ps(products, _mm512_mul_ps(memory, _mm512_loadu_ps(c + t * 16)));
        float result = 0.0f;
        for (float product : products) result += product;
        y[pos] = result + u[pos] * d[channel];
    }
}
#endif

struct ScanKernel {
    ScanKernel(const OrtApi& api, const OrtKernelInfo* info) : api(api) {
        const auto direction = Ort::ConstKernelInfo(info).GetAttribute<int64_t>("reverse");
        if (direction != 0 && direction != 1) {
            throw std::invalid_argument("reverse must be zero or one");
        }
        reverse = direction != 0;
    }

    OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
        try {
            Ort::KernelContext context(raw);
            const auto dt = context.GetInput(0);
            const auto u = context.GetInput(1);
            const auto b = context.GetInput(2);
            const auto c = context.GetInput(3);
            const auto a = context.GetInput(4);
            const auto d = context.GetInput(5);
            const auto shape = u.GetTensorTypeAndShapeInfo().GetShape();
            if (
                shape.size() != 2 || shape[0] < 0 || shape[1] <= 0
                || dt.GetTensorTypeAndShapeInfo().GetShape() != shape
                || b.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{shape[0], 16}
                || c.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{shape[0], 16}
                || a.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{shape[1], 16}
                || d.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{shape[1]}
            ) {
                throw std::invalid_argument("Invalid TypeSeg selective scan tensor shapes");
            }
            auto output = context.GetOutput(0, shape);
            ScanWork work{
                dt.GetTensorData<float>(), u.GetTensorData<float>(),
                b.GetTensorData<float>(), c.GetTensorData<float>(),
                a.GetTensorData<float>(), d.GetTensorData<float>(),
                output.GetTensorMutableData<float>(), shape[0], shape[1], reverse,
            };
            context.ParallelFor([](void* data, size_t channel) {
                static_cast<ScanWork*>(data)->Run(channel);
            }, static_cast<size_t>(shape[1]), 0, &work);
            return nullptr;
        } catch (const std::exception& error) {
            return api.CreateStatus(ORT_FAIL, error.what());
        }
    }

    const OrtApi& api;
    bool reverse;
};

struct ScanOp : Ort::CustomOpBase<ScanOp, ScanKernel, true> {
    OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
        try {
            *kernel = new ScanKernel(api, info);
            return nullptr;
        } catch (const std::exception& error) {
            return api.CreateStatus(ORT_FAIL, error.what());
        }
    }
    const char* GetName() const { return "SelectiveScan"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
    size_t GetInputTypeCount() const { return 6; }
    ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
};

}  // namespace

#ifdef _WIN32
#define TYPESEG_EXPORT __declspec(dllexport)
#else
#define TYPESEG_EXPORT __attribute__((visibility("default")))
#endif

extern "C" TYPESEG_EXPORT OrtStatus* RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* base) {
    const OrtApi* api = base->GetApi(ORT_API_VERSION);
    if (!api) {
        return base->GetApi(1)->CreateStatus(ORT_FAIL, "TypeSeg CPU scan requires ONNX Runtime 1.17 or newer");
    }
    try {
        static std::once_flag initialized;
        std::call_once(initialized, [api] { Ort::InitApi(api); });
        static ScanOp operation;
        static Ort::CustomOpDomain domain = [] {
            Ort::CustomOpDomain value("typeseg.cpu");
            value.Add(&operation);
            return value;
        }();
        Ort::UnownedSessionOptions settings(options);
        settings.Add(domain);
        return nullptr;
    } catch (const std::exception& error) {
        return api->CreateStatus(ORT_FAIL, error.what());
    }
}

static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_scan_cpu", nullptr, -1, nullptr};

PyMODINIT_FUNC PyInit__scan_cpu() {
    PyObject* value = PyModule_Create(&module);
#ifdef TYPESEG_SSE2
    const char* simd = wide_scan_available() ? "avx512" : "sse2";
#else
    const char* simd = "scalar";
#endif
    if (value && PyModule_AddStringConstant(value, "simd", simd) < 0) {
        Py_DECREF(value);
        return nullptr;
    }
    return value;
}
