#ifndef PS2_SIMD_H
#define PS2_SIMD_H

#include <cstdint>
#include <cstring>

// Only include x86 intrinsics on x86/x64 platforms
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h> // For SSE/AVX instructions
// Use native SIMD types
typedef __m128i simd128i_t;
typedef __m128 simd128f_t;
#else
// Fallback for other architectures (like Apple Silicon)
// TODO: Implement ARM NEON equivalents for better performance
struct alignas(16) simd128i_t
{
    union
    {
        uint64_t data[2];
        uint32_t m128i_u32[4];
        int32_t m128i_i32[4];
        uint64_t m128i_u64[2];
        int64_t m128i_i64[2];
    };
    simd128i_t() : data{0, 0} {}
    simd128i_t(uint64_t lo, uint64_t hi) : data{lo, hi} {}
};
struct alignas(16) simd128f_t
{
    float data[4];
    simd128f_t() : data{0.0f, 0.0f, 0.0f, 0.0f} {}
    simd128f_t(float x, float y, float z, float w) : data{x, y, z, w} {}
};
#endif

// Cross-platform SIMD function wrappers
namespace simd
{
    // Load/Store operations
    inline simd128i_t load_128i(const void *ptr)
    {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        return _mm_loadu_si128(static_cast<const __m128i *>(ptr));
#else
        simd128i_t result;
        memcpy(&result, ptr, 16);
        return result;
#endif
    }

    inline void store_128i(void *ptr, simd128i_t value)
    {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        _mm_storeu_si128(static_cast<__m128i *>(ptr), value);
#else
        memcpy(ptr, &value, 16);
#endif
    }

    inline simd128i_t setzero_128i()
    {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        return _mm_setzero_si128();
#else
        return simd128i_t{0, 0};
#endif
    }

    inline simd128i_t set1_epi32(int32_t value)
    {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        return _mm_set1_epi32(value);
#else
        uint32_t v = static_cast<uint32_t>(value);
        uint64_t doubled = (static_cast<uint64_t>(v) << 32) | v;
        return simd128i_t{doubled, doubled};
#endif
    }

    inline simd128f_t setzero_ps()
    {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        return _mm_setzero_ps();
#else
        return simd128f_t{0.0f, 0.0f, 0.0f, 0.0f};
#endif
    }

    inline uint64_t extract_epi64(simd128i_t value, int index)
    {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        return _mm_extract_epi64(value, index);
#else
        return value.data[index & 1];
#endif
    }

    // Additional helper functions for generated macros compatibility
    inline simd128i_t set_epi32(int32_t w, int32_t z, int32_t y, int32_t x)
    {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        return _mm_set_epi32(w, z, y, x);
#else
        simd128i_t result;
        uint32_t *data = reinterpret_cast<uint32_t *>(&result);
        data[0] = x;
        data[1] = y;
        data[2] = z;
        data[3] = w;
        return result;
#endif
    }
    
    inline simd128i_t set_epi64x(int64_t hi, int64_t lo)
    {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        return _mm_set_epi64x(hi, lo);
#else
        simd128i_t result;
        result.data[0] = static_cast<uint64_t>(lo);
        result.data[1] = static_cast<uint64_t>(hi);
        return result;
#endif
    }
}

#endif // PS2_SIMD_H