#include <cstdint>
#include <cstdio>
#include <cstring>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "logp_model_tflite.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr size_t kFingerprintBits = 2048;
constexpr size_t kCommandPrefixLength = 3;
constexpr size_t kLineBufferSize = kCommandPrefixLength + kFingerprintBits + 4;
constexpr size_t kTensorArenaSize = 64 * 1024;

alignas(16) uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

bool InitializeModel() {
    tflite::InitializeTarget();

    const tflite::Model* model = tflite::GetModel(logp_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        std::printf(
            "ERROR model_schema_mismatch got=%lu expected=%d\n",
            static_cast<unsigned long>(model->version()),
            TFLITE_SCHEMA_VERSION);
        std::fflush(stdout);
        return false;
    }

    static tflite::MicroMutableOpResolver<1> resolver;
    if (resolver.AddFullyConnected() != kTfLiteOk) {
        std::printf("ERROR add_fully_connected_failed\n");
        std::fflush(stdout);
        return false;
    }

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::printf("ERROR allocate_tensors_failed arena=%u\n", static_cast<unsigned>(kTensorArenaSize));
        std::fflush(stdout);
        return false;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    if (input_tensor == nullptr || output_tensor == nullptr) {
        std::printf("ERROR tensor_lookup_failed\n");
        std::fflush(stdout);
        return false;
    }

    if (input_tensor->type != kTfLiteInt8 || output_tensor->type != kTfLiteInt8) {
        std::printf("ERROR expected_int8_io input=%d output=%d\n", input_tensor->type, output_tensor->type);
        std::fflush(stdout);
        return false;
    }

    if (input_tensor->dims == nullptr || input_tensor->dims->size != 2 ||
        input_tensor->dims->data[0] != 1 || input_tensor->dims->data[1] != static_cast<int>(kFingerprintBits)) {
        std::printf("ERROR unexpected_input_shape\n");
        std::fflush(stdout);
        return false;
    }

    if (output_tensor->dims == nullptr || output_tensor->dims->size != 2 ||
        output_tensor->dims->data[0] != 1 || output_tensor->dims->data[1] != 1) {
        std::printf("ERROR unexpected_output_shape\n");
        std::fflush(stdout);
        return false;
    }

    std::printf(
        "READY input_scale=%.9f input_zero_point=%ld output_scale=%.9f output_zero_point=%ld\n",
        static_cast<double>(input_tensor->params.scale),
        static_cast<long>(input_tensor->params.zero_point),
        static_cast<double>(output_tensor->params.scale),
        static_cast<long>(output_tensor->params.zero_point));
    std::fflush(stdout);
    return true;
}

bool ReadLine(char* buffer, size_t buffer_size) {
    size_t index = 0;
    while (true) {
        int value = std::getchar();
        if (value == EOF) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        char ch = static_cast<char>(value);
        if (ch == '\r') {
            continue;
        }

        if (ch == '\n') {
            if (index == 0) {
                continue;
            }
            buffer[index] = '\0';
            return true;
        }

        if (index + 1 >= buffer_size) {
            buffer[0] = '\0';
            std::printf("ERROR command_too_long\n");
            std::fflush(stdout);
            index = 0;
            continue;
        }

        buffer[index++] = ch;
    }
}

bool LoadFingerprint(const char* payload) {
    if (std::strlen(payload) != kFingerprintBits) {
        std::printf("ERROR fingerprint_length expected=%u got=%u\n",
                    static_cast<unsigned>(kFingerprintBits),
                    static_cast<unsigned>(std::strlen(payload)));
        std::fflush(stdout);
        return false;
    }

    for (size_t index = 0; index < kFingerprintBits; ++index) {
        char bit = payload[index];
        if (bit == '0') {
            input_tensor->data.int8[index] = -128;
        } else if (bit == '1') {
            input_tensor->data.int8[index] = 127;
        } else {
            std::printf("ERROR invalid_bit index=%u value=%c\n",
                        static_cast<unsigned>(index), bit);
            std::fflush(stdout);
            return false;
        }
    }

    return true;
}

void HandleFingerprintCommand(const char* payload) {
    if (!LoadFingerprint(payload)) {
        return;
    }

    const int64_t start_us = esp_timer_get_time();
    if (interpreter->Invoke() != kTfLiteOk) {
        std::printf("ERROR invoke_failed\n");
        std::fflush(stdout);
        return;
    }
    const int64_t elapsed_us = esp_timer_get_time() - start_us;

    const int8_t quantized_output = output_tensor->data.int8[0];
    const float prediction =
        (static_cast<float>(quantized_output) - static_cast<float>(output_tensor->params.zero_point)) *
        output_tensor->params.scale;

    std::printf("RESULT %.6f %lld\n", static_cast<double>(prediction), static_cast<long long>(elapsed_us));
    std::fflush(stdout);
}

}  // namespace

extern "C" void app_main(void) {
    std::setvbuf(stdin, nullptr, _IONBF, 0);
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    if (!InitializeModel()) {
        return;
    }

    char line[kLineBufferSize] = {};
    while (ReadLine(line, sizeof(line))) {
        if (std::strcmp(line, "PING") == 0) {
            std::printf("PONG\n");
            std::fflush(stdout);
            continue;
        }

        if (std::strcmp(line, "HELP") == 0) {
            std::printf("INFO send: FP <2048 bits of 0/1>\n");
            std::fflush(stdout);
            continue;
        }

        if (std::strncmp(line, "FP ", kCommandPrefixLength) == 0) {
            HandleFingerprintCommand(line + kCommandPrefixLength);
            continue;
        }

        std::printf("ERROR unknown_command\n");
        std::fflush(stdout);
    }
}