cmake_minimum_required(VERSION 3.21)
project(trdrop-cli VERSION 1.1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(TRDROP_BUILD_TOOLS "Build the test fixture generator" ON)

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
)
target_include_directories(trdrop-pp PRIVATE src)
target_link_libraries(trdrop-pp PRIVATE ${OpenCV_LIBS})
trdrop_warnings(trdrop-pp)

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
