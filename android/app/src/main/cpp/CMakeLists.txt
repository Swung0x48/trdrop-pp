# The analysis of trdrop++ as a shared library for the Android app: the sources of the command line program
# (FFmpeg decoding and encoding with the MediaCodec codecs of the device, analysis on the CPU) and a JNI entry.
#
#   TRDROP_OPENCV_ANDROID_SDK  the unpacked OpenCV Android SDK (the folder with sdk/)
#   TRDROP_FFMPEG_ANDROID      FFmpeg built by android/build-ffmpeg.sh (the folder with <abi>/include, <abi>/lib)
cmake_minimum_required(VERSION 3.22)
project(trdrop-android LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(TRDROP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)

if(NOT EXISTS "${TRDROP_OPENCV_ANDROID_SDK}/sdk/native/jni/OpenCVConfig.cmake")
    message(FATAL_ERROR "OpenCV Android SDK not found in '${TRDROP_OPENCV_ANDROID_SDK}' (see android/README.md)")
endif()
set(OpenCV_DIR "${TRDROP_OPENCV_ANDROID_SDK}/sdk/native/jni")
find_package(OpenCV REQUIRED)

set(_ffmpeg "${TRDROP_FFMPEG_ANDROID}/${ANDROID_ABI}")
if(NOT EXISTS "${_ffmpeg}/lib/libavcodec.so")
    message(FATAL_ERROR "FFmpeg for ${ANDROID_ABI} not found in '${_ffmpeg}', build it with android/build-ffmpeg.sh (see android/README.md)")
endif()
foreach(_lib avformat avcodec avutil)
    add_library(ffmpeg_${_lib} SHARED IMPORTED)
    set_target_properties(ffmpeg_${_lib} PROPERTIES
        IMPORTED_LOCATION "${_ffmpeg}/lib/lib${_lib}.so"
        INTERFACE_INCLUDE_DIRECTORIES "${_ffmpeg}/include")
endforeach()

# the compute shaders of the GPU pipeline, embedded as SPIR-V (<name>.spv.inc), compiled with the NDK's glslc
set(_shader_dir "${CMAKE_CURRENT_BINARY_DIR}/shaders")
find_program(TRDROP_GLSLC glslc HINTS "${ANDROID_NDK}/shader-tools/${ANDROID_HOST_TAG}" NO_DEFAULT_PATH)
if(NOT TRDROP_GLSLC)
    message(FATAL_ERROR "glslc not found in ${ANDROID_NDK}/shader-tools/${ANDROID_HOST_TAG}")
endif()
set(_shaders "")
foreach(_name yuv_diff motion_pyramid motion_lk overlay_base overlay_draw ahb_import nv12_to_rgba)
    set(_src "${TRDROP_ROOT}/src/vulkan/${_name}.comp")
    set(_inc "${_shader_dir}/${_name}.spv.inc")
    add_custom_command(
        OUTPUT "${_inc}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_shader_dir}"
        COMMAND ${TRDROP_GLSLC} -O --target-env=vulkan1.1 -mfmt=num -o "${_inc}" "${_src}"
        DEPENDS "${_src}"
        COMMENT "Compiling ${_name}.comp"
        VERBATIM)
    list(APPEND _shaders "${_inc}")
endforeach()

add_library(trdrop SHARED
    trdrop_jni.cpp
    ${TRDROP_ROOT}/src/android/vk_mediacodec_pipeline.cpp
    ${_shaders}
    ${TRDROP_ROOT}/src/main.cpp
    ${TRDROP_ROOT}/src/video_sources.cpp
    ${TRDROP_ROOT}/src/frame_analyzer.cpp
    ${TRDROP_ROOT}/src/diff_engine_cpu.cpp
    ${TRDROP_ROOT}/src/opencv_pipeline.cpp
    ${TRDROP_ROOT}/src/motion.cpp
    ${TRDROP_ROOT}/src/motion_video.cpp
    ${TRDROP_ROOT}/src/video_writer.cpp
    ${TRDROP_ROOT}/src/ffmpeg/ffmpeg_decoder.cpp
    ${TRDROP_ROOT}/src/ffmpeg/ffmpeg_cpu_pipeline.cpp
    ${TRDROP_ROOT}/src/ffmpeg/ffmpeg_writer.cpp
    ${TRDROP_ROOT}/src/ffmpeg/yuv_picture.cpp
)
target_include_directories(trdrop PRIVATE ${TRDROP_ROOT}/src ${OpenCV_INCLUDE_DIRS} "${_shader_dir}")
target_compile_definitions(trdrop PRIVATE TRDROP_LIBRARY=1 USE_FFMPEG=1 TRDROP_MEDIACODEC_VULKAN=1)
target_compile_options(trdrop PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
# the OpenCV of the SDK as one shared library (libopencv_java4.so), FFmpeg as shared libraries
target_link_libraries(trdrop PRIVATE opencv_java ffmpeg_avformat ffmpeg_avcodec ffmpeg_avutil vulkan mediandk android log)
