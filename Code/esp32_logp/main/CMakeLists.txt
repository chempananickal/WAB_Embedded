idf_component_register(
    SRCS
        "main.cc"
        "../../artifacts/logp_model_tflite.cc"
    PRIV_REQUIRES spi_flash esp_timer esp-tflite-micro
    INCLUDE_DIRS
        "."
        "../../artifacts")