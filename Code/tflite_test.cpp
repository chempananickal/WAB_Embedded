
#define TFLITE_MICRO_UNIT_TEST
#include "tflite_micro_skeleton.h"

#include <cstdio>

int main() {
	if (!logp_inference::init_model()) {
		std::fprintf(stderr, "Model init failed\n");
		return 1;
	}

	static float input_data[logp_inference::kInputSize] = {0.0f};
	float result = 0.0f;
	if (!logp_inference::run_inference(
			input_data, logp_inference::kInputSize, &result)) {
		std::fprintf(stderr, "Inference failed\n");
		return 1;
	}

	std::printf("Inference OK, logP=%f\n", result);
	return 0;
}
