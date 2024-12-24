#pragma once

#include <cstdint>
#include <cstdlib>

inline uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    return h;
}

inline uint64_t fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= (uint64_t) 0xff51afd7ed558ccdull;
    k ^= k >> 33;
    k *= (uint64_t) 0xc4ceb9fe1a85ec53ull;
    k ^= k >> 33;
    return k;
}