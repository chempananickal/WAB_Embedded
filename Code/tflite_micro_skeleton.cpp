// Minimal TFLite Micro inference skeleton for ESP32.
// You must provide model_data.h (C array) and adjust tensor arena size.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "model_data.h"

#if defined(ARDUINO)
#include <Arduino.h>
#endif

namespace {
constexpr int kTensorArenaSize = 200 * 1024;
constexpr size_t kInputSize = 2048;
static uint8_t tensor_arena[kTensorArenaSize];

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

static tflite::ErrorReporter* reporter = nullptr;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;

static bool init_model() {
    static tflite::MicroErrorReporter micro_reporter;
    reporter = &micro_reporter;

    model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        reporter->Report("Model schema mismatch");
        return false;
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        reporter->Report("AllocateTensors failed");
        interpreter = nullptr;
        return false;
    }

    return true;
}

static bool run_inference(const float* input, size_t input_len, float* output) {
    if (!interpreter) {
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

#if defined(ARDUINO)

void setup() {
    Serial.begin(115200);
    while (!Serial) {
        delay(10);
    }
    if (!init_model()) {
        Serial.println("Model init failed");
    } else {
        Serial.println("Model init OK");
    }
}

void loop() {
    static float input_data[kInputSize] = {0.0f};
    float result = 0.0f;

    // Fill input_data with 2048 floats (Morgan fingerprint) from host or sensor.
    // This example uses zeros to prove the pipeline runs.
    const bool ok = run_inference(input_data, kInputSize, &result);
    if (ok) {
        Serial.print("logP: ");
        Serial.println(result, 6);
    } else {
        Serial.println("Inference failed");
    }

    delay(2000);
}

#else

extern "C" void app_main(void) {
    if (!init_model()) {
        return;
    }
    static float input_data[kInputSize] = {0.0f};
    float result = 0.0f;
    const bool ok = run_inference(input_data, kInputSize, &result);
    (void)ok;
    // Add your input acquisition and inference loop here.
}

#endif
