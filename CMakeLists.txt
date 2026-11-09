cmake_minimum_required(VERSION 3.21)
project(trdrop-cli VERSION 1.1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(TRDROP_BUILD_TOOLS "Build the test fixture generator" ON)
option(USE_VULKAN "Build the Vulkan (GPU) accelerated backend (needs the Vulkan headers and glslc)" OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)

function(trdrop_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /utf-8 /permissive-)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra)
    endif()
endfunction()

add_executable(trdrop-pp
    src/main.cpp
    src/video_sources.cpp
    src/frame_analyzer.cpp
    src/diff_engine_cpu.cpp
)
target_include_directories(trdrop-pp PRIVATE src)
target_link_libraries(trdrop-pp PRIVATE ${OpenCV_LIBS})
trdrop_warnings(trdrop-pp)

if(USE_VULKAN)
    # only the headers and the shader compiler are needed at build time,
    # the Vulkan loader is loaded at runtime (the program falls back to the CPU without it)
    find_package(Vulkan REQUIRED)
    if(NOT Vulkan_GLSLC_EXECUTABLE)
        find_program(Vulkan_GLSLC_EXECUTABLE NAMES glslc HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin")
    endif()
    if(NOT Vulkan_GLSLC_EXECUTABLE)
        message(FATAL_ERROR "USE_VULKAN: glslc not found (install the Vulkan SDK or set Vulkan_GLSLC_EXECUTABLE)")
    endif()

    set(_shader_dir "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    set(_shader_src "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/row_diff.comp")
    set(_shader_inc "${_shader_dir}/row_diff.spv.inc")
    add_custom_command(
        OUTPUT "${_shader_inc}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_shader_dir}"
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.0 -mfmt=num -o "${_shader_inc}" "${_shader_src}"
        DEPENDS "${_shader_src}"
        COMMENT "Compiling row_diff.comp"
        VERBATIM)

    target_sources(trdrop-pp PRIVATE src/vulkan/vk_diff_engine.cpp "${_shader_inc}")
    target_include_directories(trdrop-pp PRIVATE "${_shader_dir}")
    target_compile_definitions(trdrop-pp PRIVATE USE_VULKAN=1)
    if(TARGET Vulkan::Headers)
        target_link_libraries(trdrop-pp PRIVATE Vulkan::Headers)
    else()
        target_include_directories(trdrop-pp PRIVATE ${Vulkan_INCLUDE_DIRS})
    endif()
    target_link_libraries(trdrop-pp PRIVATE ${CMAKE_DL_LIBS})
    message(STATUS "trdrop-pp: Vulkan backend enabled (glslc: ${Vulkan_GLSLC_EXECUTABLE})")
endif()

if(TRDROP_BUILD_TOOLS)
    add_executable(trdrop_make_fixtures tools/make_fixtures.cpp)
    target_link_libraries(trdrop_make_fixtures PRIVATE ${OpenCV_LIBS})
    trdrop_warnings(trdrop_make_fixtures)
endif()

# Windows: copy the OpenCV DLLs (incl. the dynamically loaded ffmpeg plugin) next to the executable
if(WIN32)
    set(_ocv_dll_dirs "")
    foreach(_lib IN LISTS OpenCV_LIBS)
        if(TARGET ${_lib})
            foreach(_cfg RELEASE RELWITHDEBINFO DEBUG NOCONFIG)
                get_target_property(_loc ${_lib} IMPORTED_LOCATION_${_cfg})
                if(_loc)
                    get_filename_component(_dir "${_loc}" DIRECTORY)
                    list(APPEND _ocv_dll_dirs "${_dir}")
                endif()
            endforeach()
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _ocv_dll_dirs)
    set(_ocv_dlls "")
    foreach(_dir IN LISTS _ocv_dll_dirs)
        file(GLOB _found "${_dir}/opencv_*.dll")
        # skip the debug runtime of the prebuilt Windows pack (e.g. opencv_world4120d.dll)
        list(FILTER _found EXCLUDE REGEX "opencv_world[0-9]+d\\.dll$")
        list(APPEND _ocv_dlls ${_found})
    endforeach()
    if(_ocv_dlls)
        add_custom_command(TARGET trdrop-pp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_ocv_dlls} "$<TARGET_FILE_DIR:trdrop-pp>"
            COMMENT "Copying OpenCV runtime DLLs")
    endif()
endif()

install(TARGETS trdrop-pp RUNTIME DESTINATION bin)
