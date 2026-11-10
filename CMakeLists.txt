cmake_minimum_required(VERSION 3.21)
# trdrop++, spelled trdrop-pp where a "+" is unwelcome: project, target and file names
project(trdrop-pp VERSION 1.1.1 DESCRIPTION "trdrop++ - raw video analysis (framerate, frametime and tear detection)" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(TRDROP_BUILD_TOOLS "Build the test fixture generator" ON)
option(TRDROP_INSTALL_RUNTIME "Windows: install the DLLs the program needs (OpenCV, FFmpeg, MSVC runtime) next to it" ON)
option(USE_VULKAN "Build the Vulkan (GPU) accelerated backend (needs the Vulkan headers and glslc)" OFF)
option(USE_FFMPEG "Decode with FFmpeg directly, with USE_VULKAN on the GPU via Vulkan Video (needs FFmpeg >= 8.0)" OFF)
set(FFMPEG_DIR "" CACHE PATH "FFmpeg development files (include/, lib/, bin/), otherwise pkg-config is used")

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio video)

# Where `cmake --install` puts things. Windows: a folder that can be zipped and used anywhere
# (bin/, scripts/, skills/, licenses/, README.md); elsewhere the usual prefix layout.
include(GNUInstallDirs)
if(WIN32)
    set(TRDROP_INSTALL_BINDIR bin)
    set(TRDROP_INSTALL_SCRIPTDIR scripts)
    set(TRDROP_INSTALL_SKILLDIR skills)
    set(TRDROP_INSTALL_DOCDIR .)
    set(TRDROP_INSTALL_LICENSEDIR licenses)
else()
    set(TRDROP_INSTALL_BINDIR "${CMAKE_INSTALL_BINDIR}")
    set(TRDROP_INSTALL_SCRIPTDIR "${CMAKE_INSTALL_DATADIR}/trdrop-pp/scripts")
    set(TRDROP_INSTALL_SKILLDIR "${CMAKE_INSTALL_DATADIR}/trdrop-pp/skills")
    set(TRDROP_INSTALL_DOCDIR "${CMAKE_INSTALL_DOCDIR}")
    set(TRDROP_INSTALL_LICENSEDIR "${CMAKE_INSTALL_DOCDIR}/licenses")
endif()

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
    src/opencv_pipeline.cpp
    src/motion.cpp
    src/motion_video.cpp
    src/video_writer.cpp
)
target_include_directories(trdrop-pp PRIVATE src)
target_link_libraries(trdrop-pp PRIVATE ${OpenCV_LIBS})
trdrop_warnings(trdrop-pp)
if(MSVC)
    # UTF-8 process code page: non-ASCII file names on the command line
    target_sources(trdrop-pp PRIVATE src/windows/trdrop-pp.manifest)
endif()

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

    # compute shaders are embedded as SPIR-V (<name>.spv.inc)
    set(_shader_dir "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    function(trdrop_shader name out_var)
        set(_src "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/${name}.comp")
        set(_inc "${_shader_dir}/${name}.spv.inc")
        add_custom_command(
            OUTPUT "${_inc}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_shader_dir}"
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.0 -mfmt=num -o "${_inc}" "${_src}"
            DEPENDS "${_src}"
            COMMENT "Compiling ${name}.comp"
            VERBATIM)
        set(${out_var} "${_inc}" PARENT_SCOPE)
    endfunction()
    trdrop_shader(row_diff _shader_inc)

    target_sources(trdrop-pp PRIVATE src/vulkan/vk_diff_engine.cpp "${_shader_inc}")
    target_include_directories(trdrop-pp PRIVATE "${_shader_dir}")
    target_compile_definitions(trdrop-pp PRIVATE USE_VULKAN=1)
    if(TARGET Vulkan::Headers)
        target_link_libraries(trdrop-pp PRIVATE Vulkan::Headers)
    else()
        target_include_directories(trdrop-pp PRIVATE ${Vulkan_INCLUDE_DIRS})
    endif()
    target_link_libraries(trdrop-pp PRIVATE ${CMAKE_DL_LIBS})
    message(STATUS "trdrop++: Vulkan backend enabled (glslc: ${Vulkan_GLSLC_EXECUTABLE})")
endif()

if(USE_FFMPEG)
    # libavformat / libavcodec / libavutil, linked dynamically (an LGPL build is sufficient)
    set(_ffmpeg_libs avformat avcodec avutil)
    if(FFMPEG_DIR)
        find_path(FFMPEG_INCLUDE_DIR libavcodec/avcodec.h HINTS "${FFMPEG_DIR}/include" NO_DEFAULT_PATH)
        set(FFMPEG_LIBRARIES "")
        foreach(_lib IN LISTS _ffmpeg_libs)
            find_library(FFMPEG_${_lib}_LIBRARY NAMES ${_lib} HINTS "${FFMPEG_DIR}/lib" NO_DEFAULT_PATH)
            if(NOT FFMPEG_${_lib}_LIBRARY)
                message(FATAL_ERROR "USE_FFMPEG: lib${_lib} not found in ${FFMPEG_DIR}/lib")
            endif()
            list(APPEND FFMPEG_LIBRARIES "${FFMPEG_${_lib}_LIBRARY}")
        endforeach()
        if(NOT FFMPEG_INCLUDE_DIR)
            message(FATAL_ERROR "USE_FFMPEG: FFmpeg headers not found in ${FFMPEG_DIR}/include")
        endif()
        target_include_directories(trdrop-pp SYSTEM PRIVATE "${FFMPEG_INCLUDE_DIR}")
        target_link_libraries(trdrop-pp PRIVATE ${FFMPEG_LIBRARIES})
    else()
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavformat>=62 libavcodec>=62 libavutil>=60)
        target_link_libraries(trdrop-pp PRIVATE PkgConfig::FFMPEG)
    endif()

    target_sources(trdrop-pp PRIVATE src/ffmpeg/ffmpeg_decoder.cpp src/ffmpeg/ffmpeg_cpu_pipeline.cpp src/ffmpeg/ffmpeg_writer.cpp
                                     src/ffmpeg/yuv_picture.cpp)
    target_compile_definitions(trdrop-pp PRIVATE USE_FFMPEG=1)
    if(USE_VULKAN)
        trdrop_shader(yuv_diff _yuv_shader_inc)
        trdrop_shader(motion_pyramid _pyramid_shader_inc)
        trdrop_shader(motion_lk _lk_shader_inc)
        trdrop_shader(overlay_base _overlay_base_shader_inc)
        trdrop_shader(overlay_draw _overlay_draw_shader_inc)
        target_sources(trdrop-pp PRIVATE src/vulkan/vk_video_pipeline.cpp "${_yuv_shader_inc}" "${_pyramid_shader_inc}" "${_lk_shader_inc}"
                                         "${_overlay_base_shader_inc}" "${_overlay_draw_shader_inc}")
        message(STATUS "trdrop++: FFmpeg decoding enabled (with Vulkan Video)")
    else()
        message(STATUS "trdrop++: FFmpeg decoding enabled (CPU only, Vulkan Video needs USE_VULKAN)")
    endif()

    # Windows: copy the FFmpeg DLLs next to the executable
    if(WIN32 AND FFMPEG_DIR)
        # avformat -> avcodec -> avutil, swresample
        file(GLOB _ffmpeg_dlls "${FFMPEG_DIR}/bin/avformat*.dll" "${FFMPEG_DIR}/bin/avcodec*.dll"
                               "${FFMPEG_DIR}/bin/avutil*.dll" "${FFMPEG_DIR}/bin/swresample*.dll")
        if(_ffmpeg_dlls)
            add_custom_command(TARGET trdrop-pp POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_ffmpeg_dlls} "$<TARGET_FILE_DIR:trdrop-pp>"
                COMMENT "Copying FFmpeg runtime DLLs")
            if(TRDROP_INSTALL_RUNTIME)
                install(FILES ${_ffmpeg_dlls} DESTINATION "${TRDROP_INSTALL_BINDIR}")
                # the DLLs are LGPL: their license goes with them
                if(EXISTS "${FFMPEG_DIR}/LICENSE.txt")
                    install(FILES "${FFMPEG_DIR}/LICENSE.txt" DESTINATION "${TRDROP_INSTALL_LICENSEDIR}" RENAME FFmpeg-LICENSE.txt)
                endif()
            endif()
        endif()
    endif()
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
        if(TRDROP_INSTALL_RUNTIME)
            install(FILES ${_ocv_dlls} DESTINATION "${TRDROP_INSTALL_BINDIR}")
            # the prebuilt Windows pack: OpenCV's license and the one of its FFmpeg plugin (opencv_videoio_ffmpeg*.dll)
            foreach(_license IN ITEMS "LICENSE|OpenCV-LICENSE.txt" "../LICENSE_FFMPEG.txt|OpenCV-FFmpeg-plugin-LICENSE.txt")
                string(REPLACE "|" ";" _license "${_license}")
                list(GET _license 0 _from)
                list(GET _license 1 _to)
                if(OpenCV_INSTALL_PATH AND EXISTS "${OpenCV_INSTALL_PATH}/${_from}")
                    install(FILES "${OpenCV_INSTALL_PATH}/${_from}" DESTINATION "${TRDROP_INSTALL_LICENSEDIR}" RENAME "${_to}")
                endif()
            endforeach()
        endif()
    endif()
endif()

# ---- install: the program with what it needs to run, the Python scripts, the skills, README and licenses
#   cmake --install <build dir> --prefix <folder>
install(TARGETS trdrop-pp RUNTIME DESTINATION "${TRDROP_INSTALL_BINDIR}")
if(WIN32 AND TRDROP_INSTALL_RUNTIME)
    # the MSVC runtime, so the program also starts without the Visual C++ redistributable
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION "${TRDROP_INSTALL_BINDIR}")
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_NO_WARNINGS ON)
    include(InstallRequiredSystemLibraries)
    if(MSVC AND NOT CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS)
        # a toolset this CMake version doesn't know yet: take the DLLs from the redist folder of the toolset in use
        set(_msvc_runtime "")
        if(DEFINED ENV{VCToolsRedistDir})
            file(TO_CMAKE_PATH "$ENV{VCToolsRedistDir}" _redist)
            string(TOLOWER "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" _arch)
            file(GLOB _msvc_runtime "${_redist}/${_arch}/Microsoft.VC*.CRT/msvcp140*.dll"
                                    "${_redist}/${_arch}/Microsoft.VC*.CRT/vcruntime140*.dll"
                                    "${_redist}/${_arch}/Microsoft.VC*.CRT/concrt140.dll")
        endif()
        if(_msvc_runtime)
            install(FILES ${_msvc_runtime} DESTINATION "${TRDROP_INSTALL_BINDIR}")
        else()
            message(STATUS "trdrop++: MSVC runtime DLLs not found, the installed program needs the Visual C++ redistributable")
        endif()
    endif()
endif()
install(FILES
        scripts/framerate.py
        scripts/frametime_histogram.py
        scripts/smoothness.py
        scripts/trdrop_csv.py
        scripts/requirements.txt
        tools/make_ball_videos.py
    DESTINATION "${TRDROP_INSTALL_SCRIPTDIR}")
# agent skills (skills/<name>/SKILL.md); they expect the scripts in ../../scripts relative to themselves
install(DIRECTORY skills/ DESTINATION "${TRDROP_INSTALL_SKILLDIR}")
install(FILES README.md DESTINATION "${TRDROP_INSTALL_DOCDIR}")
install(FILES LICENSE DESTINATION "${TRDROP_INSTALL_LICENSEDIR}" RENAME trdrop++-LICENSE.txt)

# ---- package: the installed folder as an archive
#   cmake --build <build dir> --target package
set(CPACK_PACKAGE_NAME "trdrop-pp")
set(CPACK_PACKAGE_VENDOR "trdrop++")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
if(WIN32)
    set(CPACK_GENERATOR ZIP)
else()
    set(CPACK_GENERATOR TGZ)
endif()
include(CPack)
