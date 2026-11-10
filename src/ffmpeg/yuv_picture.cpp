#include "ffmpeg/yuv_picture.h"

#include <opencv2/imgproc.hpp>

#include "motion.h"

namespace trdrop::ffmpeg {

YuvPlanes yuv_planes(const AVFrame * frame)
{
    YuvPlanes planes;
    planes.width = frame->width;
    planes.height = frame->height;
    planes.layout = yuv_layout(static_cast<AVPixelFormat>(frame->format));
    planes.v_first = frame->format == AV_PIX_FMT_NV21 || frame->format == AV_PIX_FMT_NV42;
    for (int plane = 0; plane < 3; ++plane)
    {
        planes.data[plane] = frame->data[plane];
        planes.stride[plane] = static_cast<size_t>(frame->linesize[plane]);
    }
    return planes;
}

void motion_picture(const YuvPlanes & planes, cv::Mat & picture, cv::Mat & scratch)
{
    const YuvLayout & layout = planes.layout;
    const int scale = motion_scale(planes.width, planes.height);
    const cv::Size size = motion_picture_size(planes.width, planes.height);
    const cv::Size chroma_size(size.width / 2, size.height / 2);
    picture.create(size.height * 3 / 2, size.width, CV_8UC1);
    cv::Mat y_plane = picture(cv::Rect(0, 0, size.width, size.height));
    // the U and V planes follow the Y plane without padding
    cv::Mat u_plane(chroma_size.height, chroma_size.width, CV_8UC1, picture.data + size.area());
    cv::Mat v_plane(chroma_size.height, chroma_size.width, CV_8UC1, picture.data + size.area() + chroma_size.area());

    // the part of the frame the picture shows (an odd row / column of the measured resolution is cut off)
    const int width = size.width * scale;
    const int height = size.height * scale;
    const cv::Mat luma(height, width, CV_8UC1, const_cast<uint8_t *>(planes.data[0]), planes.stride[0]);
    cv::resize(luma, y_plane, size, 0, 0, cv::INTER_AREA);

    if (layout.chroma == YuvLayout::None)
    {
        u_plane.setTo(128);
        v_plane.setTo(128);
        return;
    }
    const int chroma_width = width >> layout.shift_x;
    const int chroma_height = height >> layout.shift_y;
    const int interpolation = chroma_width >= chroma_size.width && chroma_height >= chroma_size.height ? cv::INTER_AREA : cv::INTER_LINEAR;
    if (layout.chroma == YuvLayout::Interleaved)
    {
        const cv::Mat pairs(chroma_height, chroma_width, CV_8UC2, const_cast<uint8_t *>(planes.data[1]), planes.stride[1]);
        cv::resize(pairs, scratch, chroma_size, 0, 0, interpolation);
        cv::extractChannel(scratch, u_plane, planes.v_first ? 1 : 0);
        cv::extractChannel(scratch, v_plane, planes.v_first ? 0 : 1);
    }
    else
    {
        const cv::Mat u(chroma_height, chroma_width, CV_8UC1, const_cast<uint8_t *>(planes.data[1]), planes.stride[1]);
        const cv::Mat v(chroma_height, chroma_width, CV_8UC1, const_cast<uint8_t *>(planes.data[2]), planes.stride[2]);
        cv::resize(u, u_plane, chroma_size, 0, 0, interpolation);
        cv::resize(v, v_plane, chroma_size, 0, 0, interpolation);
    }
}

} // namespace trdrop::ffmpeg
