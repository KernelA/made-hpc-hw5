#pragma once
#include "stdafx.h"

namespace comb {

template <typename T>
void random_permutation(T* data, size_t size,
                        std::default_random_engine& engine) {
  for (size_t i = 0; i < size; i++) {
    std::uniform_int_distribution<size_t> gen(0, i);
    size_t j = gen(engine);
    std::swap(data[i], data[j]);
  }
}
}  // namespace comb
