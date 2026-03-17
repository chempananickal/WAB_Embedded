// Minimal TFLite Micro inference skeleton for ESP32.
// You must provide model_data.h (C array) and adjust tensor arena size.

#include "tflite_micro_skeleton.h"

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "model_data.h"

#if defined(ARDUINO)
#include <Arduino.h>
#endif

namespace {
constexpr int kTensorArenaSize = 200 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

inline int8_t quantize_float(float value, float scale, int zero_point) {
    if (scale <= 0.0f) {
        return static_cast<int8_t>(zero_point);
    }
    const float scaled = value / scale;
    const int32_t shifted = static_cast<int32_t>(std::round(scaled)) + zero_point;
    if (shifted < -128) {
        return -128;
    }
    if (shifted > 127) {
        return 127;
    }
    return static_cast<int8_t>(shifted);
}

inline float dequantize_int8(int8_t value, float scale, int zero_point) {
    return (static_cast<int>(value) - zero_point) * scale;
}
}

static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;

namespace logp_inference {

bool init_model() {
    model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model schema mismatch");
        return false;
    }

    static tflite::MicroMutableOpResolver<1> resolver;
    const TfLiteStatus add_status = resolver.AddFullyConnected();
    if (add_status != kTfLiteOk) {
        MicroPrintf("Failed to register FULLY_CONNECTED op");
        return false;
    }

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors failed");
        interpreter = nullptr;
        return false;
    }

    return true;
}

bool run_inference(const float* input, std::size_t input_len, float* output) {
    if (!interpreter) {
        return false;
    }
    if (input == nullptr || output == nullptr) {
        return false;
    }

    TfLiteTensor* input_tensor = interpreter->input(0);
    if (input_len != kInputSize) {
        return false;
    }

    if (input_tensor->type == kTfLiteFloat32) {
        if (static_cast<size_t>(input_tensor->bytes) != input_len * sizeof(float)) {
            return false;
        }
        std::memcpy(input_tensor->data.f, input, input_len * sizeof(float));
    } else if (input_tensor->type == kTfLiteInt8) {
        if (static_cast<size_t>(input_tensor->bytes) != input_len * sizeof(int8_t)) {
            return false;
        }
        const float scale = input_tensor->params.scale;
        const int zero_point = input_tensor->params.zero_point;
        for (size_t i = 0; i < input_len; ++i) {
            input_tensor->data.int8[i] = quantize_float(input[i], scale, zero_point);
        }
    } else {
        return false;
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        return false;
    }

    TfLiteTensor* output_tensor = interpreter->output(0);
    if (output_tensor->type == kTfLiteFloat32) {
        *output = output_tensor->data.f[0];
    } else if (output_tensor->type == kTfLiteInt8) {
        const float scale = output_tensor->params.scale;
        const int zero_point = output_tensor->params.zero_point;
        *output = dequantize_int8(output_tensor->data.int8[0], scale, zero_point);
    } else {
        return false;
    }
    return true;
}

}  // namespace logp_inference

#if !defined(TFLITE_MICRO_UNIT_TEST) && defined(ARDUINO)

void setup() {
    Serial.begin(115200);
    while (!Serial) {
        delay(10);
    }
    if (!logp_inference::init_model()) {
        Serial.println("Model init failed");
    } else {
        Serial.println("Model init OK");
    }
}

void loop() {
    static float input_data[logp_inference::kInputSize] = {0.0f};
    float result = 0.0f;

    // Fill input_data with 2048 floats (Morgan fingerprint) from host or sensor.
    // This example uses zeros to prove the pipeline runs.
    const bool ok = logp_inference::run_inference(
        input_data, logp_inference::kInputSize, &result);
    if (ok) {
        Serial.print("logP: ");
        Serial.println(result, 6);
    } else {
        Serial.println("Inference failed");
    }

    delay(2000);
}

#elif !defined(TFLITE_MICRO_UNIT_TEST)

extern "C" void app_main(void) {
    if (!logp_inference::init_model()) {
        return;
    }
    static float input_data[logp_inference::kInputSize] = {0.0f};
    float result = 0.0f;
    const bool ok = logp_inference::run_inference(
        input_data, logp_inference::kInputSize, &result);
    (void)ok;
    // Add your input acquisition and inference loop here.
}

#endif
