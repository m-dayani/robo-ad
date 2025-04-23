cmake_minimum_required(VERSION 3.10)
project(ImageProcessing)

# Find Python with Interpreter component
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(DBoW2 CONFIG REQUIRED)

add_library(imageproc MODULE image_processing.cpp)

target_link_libraries(imageproc PRIVATE pybind11::module ${OpenCV_LIBS} ${DBoW2_LIBRARIES})

set_target_properties(imageproc PROPERTIES
    PREFIX ""
    SUFFIX ".so"  # Use ".pyd" on Windows
)

target_include_directories(imageproc PRIVATE ${OpenCV_INCLUDE_DIRS} ${DBoW2_INCLUDE_DIRS})

