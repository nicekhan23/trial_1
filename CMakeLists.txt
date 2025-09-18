cmake_minimum_required(VERSION 3.16)

# Register extra components BEFORE including ESP-IDF build system
set(EXTRA_COMPONENT_DIRS components/esp-dl)

# Include ESP-IDF project system
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

idf_build_set_property(MINIMAL_BUILD OFF)
project(trial_1)
