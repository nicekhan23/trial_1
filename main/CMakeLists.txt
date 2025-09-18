idf_component_register(SRCS "main.c"
                       INCLUDE_DIRS "."
                       REQUIRES freertos esp_system
                       PRIV_REQUIRES esp-tflite-micro)
