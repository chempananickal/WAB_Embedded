#pragma once

#include <cstddef>

namespace logp_inference {

constexpr std::size_t kInputSize = 2048;

bool init_model();
bool run_inference(const float* input, std::size_t input_len, float* output);

}  // namespace logp_inference