#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "simple_model.h"  // Include our model

// TFLM headers - include after confirming component works
#ifdef CONFIG_ESP_TFLITE_MICRO_ENABLED
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h" 
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#endif

static const char *TAG = "TFLM_MODEL_TEST";

void app_main() {
    ESP_LOGI(TAG, "Starting TensorFlow Lite Micro model test...");
    ESP_LOGI(TAG, "Model size: %d bytes", simple_model_tflite_len);
    ESP_LOGI(TAG, "Free heap at start: %d bytes", esp_get_free_heap_size());
    
#ifdef CONFIG_ESP_TFLITE_MICRO_ENABLED
    // Initialize TensorFlow Lite
    tflite::InitializeTarget();
    
    // Set up the model
    const tflite::Model* model = tflite::GetModel(simple_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version %d not supported. Supported: %d", 
                 model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    
    // Allocate memory for operations
    static tflite::AllOpsResolver resolver;
    
    // Set up tensor arena
    const int tensor_arena_size = 4 * 1024; // 4KB
    uint8_t *tensor_arena = (uint8_t*)malloc(tensor_arena_size);
    if (tensor_arena == NULL) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena!");
        return;
    }
    
    // Set up interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, tensor_arena_size, nullptr);
    tflite::MicroInterpreter* interpreter = &static_interpreter;
    
    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        free(tensor_arena);
        return;
    }
    
    // Get input and output tensors
    TfLiteTensor* input = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);
    
    ESP_LOGI(TAG, "Model loaded successfully!");
    ESP_LOGI(TAG, "Input shape: %d dimensions", input->dims->size);
    ESP_LOGI(TAG, "Output shape: %d dimensions", output->dims->size);
    
    // Test the model with a few values
    float test_values[] = {1.0f, 5.0f, -2.0f, 10.0f};
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);
    
    for (int i = 0; i < num_tests; i++) {
        // Set input value
        input->data.f[0] = test_values[i];
        
        // Run inference
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke failed for input %f", test_values[i]);
            continue;
        }
        
        // Get output value
        float result = output->data.f[0];
        ESP_LOGI(TAG, "Input: %.2f -> Output: %.2f (Expected: %.2f)", 
                 test_values[i], result, test_values[i] + 1.0f);
    }
    
    free(tensor_arena);
    ESP_LOGI(TAG, "Model test completed successfully!");
    
#else
    ESP_LOGE(TAG, "TFLM component not found! Install esp-tflite-micro first.");
#endif
    
    while(1) {
        ESP_LOGI(TAG, "Model test running... Free heap: %d bytes", esp_get_free_heap_size());
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}