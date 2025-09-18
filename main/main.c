#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "simple_model.h"  // Include our model

// Try to include TFLM headers
#if __has_include("tensorflow/lite/micro/micro_interpreter.h")
#define TFLM_AVAILABLE 1
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/model.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#else
#define TFLM_AVAILABLE 0
#endif

static const char *TAG = "TFLM_MODEL_TEST";

void app_main() {
    ESP_LOGI(TAG, "Starting TensorFlow Lite Micro model test...");
    ESP_LOGI(TAG, "Model size: %d bytes", simple_model_tflite_len);
    ESP_LOGI(TAG, "Free heap at start: %d bytes", esp_get_free_heap_size());
    
#if TFLM_AVAILABLE
    ESP_LOGI(TAG, "TFLM component found!");
    
    // Initialize TensorFlow Lite
    tflite::InitializeTarget();
    
    // Set up the model
    const tflite::Model* model = tflite::GetModel(simple_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version %d not supported. Supported: %d", 
                 model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    ESP_LOGI(TAG, "Model version check passed");
    
    // Allocate memory for operations
    static tflite::AllOpsResolver resolver;
    
    // Set up tensor arena (increased size for better stability)
    const int tensor_arena_size = 8 * 1024; // 8KB
    uint8_t *tensor_arena = (uint8_t*)malloc(tensor_arena_size);
    if (tensor_arena == NULL) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena!");
        return;
    }
    ESP_LOGI(TAG, "Tensor arena allocated: %d bytes", tensor_arena_size);
    
    // Set up interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, tensor_arena_size);
    tflite::MicroInterpreter* interpreter = &static_interpreter;
    
    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed with status: %d", allocate_status);
        free(tensor_arena);
        return;
    }
    ESP_LOGI(TAG, "Tensors allocated successfully");
    
    // Get input and output tensors
    TfLiteTensor* input = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);
    
    if (input == nullptr || output == nullptr) {
        ESP_LOGE(TAG, "Failed to get input/output tensors");
        free(tensor_arena);
        return;
    }
    
    ESP_LOGI(TAG, "Model loaded successfully!");
    ESP_LOGI(TAG, "Input shape: %d dimensions", input->dims->size);
    for (int i = 0; i < input->dims->size; i++) {
        ESP_LOGI(TAG, "  Dimension %d: %d", i, input->dims->data[i]);
    }
    ESP_LOGI(TAG, "Output shape: %d dimensions", output->dims->size);
    for (int i = 0; i < output->dims->size; i++) {
        ESP_LOGI(TAG, "  Dimension %d: %d", i, output->dims->data[i]);
    }
    
    // Test the model with a few values
    float test_values[] = {1.0f, 5.0f, -2.0f, 10.0f};
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);
    
    ESP_LOGI(TAG, "Running inference tests...");
    for (int i = 0; i < num_tests; i++) {
        // Set input value
        input->data.f[0] = test_values[i];
        
        ESP_LOGI(TAG, "Test %d: Setting input to %.2f", i + 1, test_values[i]);
        
        // Run inference
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke failed for input %.2f with status: %d", 
                     test_values[i], invoke_status);
            continue;
        }
        
        // Get output value
        float result = output->data.f[0];
        float expected = test_values[i] + 1.0f;  // Assuming y = x + 1 model
        ESP_LOGI(TAG, "  Input: %.2f -> Output: %.2f (Expected: %.2f)", 
                 test_values[i], result, expected);
        
        // Check if result is reasonable
        float diff = result - expected;
        if (diff < -0.1f || diff > 0.1f) {
            ESP_LOGW(TAG, "  Large difference from expected: %.3f", diff);
        }
    }
    
    free(tensor_arena);
    ESP_LOGI(TAG, "Model test completed successfully!");
    
#else
    ESP_LOGE(TAG, "TFLM component not available!");
    ESP_LOGE(TAG, "Please install esp-tflite-micro component:");
    ESP_LOGE(TAG, "  idf.py add-dependency \"espressif/esp-tflite-micro\"");
    ESP_LOGE(TAG, "Or manually clone to components/esp-tflite-micro");
#endif
    
    // Main loop
    while(1) {
        ESP_LOGI(TAG, "Free heap: %d bytes", esp_get_free_heap_size());
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}