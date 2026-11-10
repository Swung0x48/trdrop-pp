#pragma once

#include <cstddef>
#include <cstdint>

#include <opencv2/core.hpp>

#include "ffmpeg/ffmpeg_decoder.h"

namespace trdrop::ffmpeg {

//! the planes of an 8-bit YUV frame in memory
struct YuvPlanes {
    int width = 0;
    int height = 0;
    YuvLayout layout;
    //! interleaved chroma: the V sample comes first (NV21, NV42)
    bool v_first = false;
    //! Y, then UV pairs (interleaved) or U and V (planar)
    const uint8_t * data[3] = {};
    size_t stride[3] = {};
};

//! the planes of a frame decoded on the CPU
YuvPlanes yuv_planes(const AVFrame * frame);

//! the frame as I420 at the resolution the motion is measured at (MotionDetail::picture)
void motion_picture(const YuvPlanes & planes, cv::Mat & picture, cv::Mat & scratch);

} // namespace trdrop::ffmpeg
