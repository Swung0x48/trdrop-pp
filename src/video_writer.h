#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

namespace trdrop {

//! encodes pictures into a video file
class VideoWriter {
public:
    virtual ~VideoWriter() = default;

    //! encoder used, e.g. h264_nvenc
    virtual std::string description() const = 0;
    //! the next picture: BGR, 8 bit, of the size the writer was made for
    virtual void write(const cv::Mat & picture) = 0;
    //! writes what is left, the file is complete afterwards
    virtual void finish() = 0;
};

//! The container follows from the extension of the path (.mp4, .mkv, .mov, ...), width and height have to be
//! even. Builds with FFmpeg encode H.264 with the first encoder that works (the GPU's, the operating system's,
//! OpenH264), the others write what OpenCV can (MPEG-4, Motion JPEG for .avi).
//! throws std::runtime_error if the file can't be written
std::unique_ptr<VideoWriter> make_video_writer(const std::string & path, int width, int height, double fps);

#ifdef USE_FFMPEG
std::unique_ptr<VideoWriter> make_ffmpeg_video_writer(const std::string & path, int width, int height, double fps);
#endif

} // namespace trdrop
