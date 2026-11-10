#include "video_writer.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include <opencv2/videoio.hpp>

namespace trdrop {

namespace {

//! OpenCV's writer: every build has it, the codecs depend on its backends
class OpenCvVideoWriter final : public VideoWriter {
public:
    OpenCvVideoWriter(const std::string & path, int width, int height, double fps)
        : _size(width, height)
    {
        std::string extension = path.substr(std::min(path.size(), path.find_last_of('.')));
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const bool avi = extension == ".avi";
        _name = avi ? "Motion JPEG" : "MPEG-4";
        const int fourcc = avi ? cv::VideoWriter::fourcc('M', 'J', 'P', 'G') : cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        if (!_writer.open(path, fourcc, fps, _size, true))
        {
            throw std::runtime_error("OpenCV can't write '" + path + "' as " + _name);
        }
    }

    std::string description() const override { return _name + " (OpenCV)"; }

    void write(const cv::Mat & picture) override
    {
        if (picture.size() != _size || picture.type() != CV_8UC3) throw std::runtime_error("video writer: unexpected picture");
        _writer.write(picture);
    }

    void finish() override { _writer.release(); }

private:
    cv::VideoWriter _writer;
    cv::Size _size;
    std::string _name;
};

} // namespace

std::unique_ptr<VideoWriter> make_video_writer(const std::string & path, int width, int height, double fps)
{
    if (width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0) throw std::runtime_error("video writer: the size has to be even");
#ifdef USE_FFMPEG
    std::string reason;
    try
    {
        return make_ffmpeg_video_writer(path, width, height, fps);
    }
    catch (const std::exception & e)
    {
        reason = e.what();
    }
    try
    {
        return std::make_unique<OpenCvVideoWriter>(path, width, height, fps);
    }
    catch (const std::exception & e)
    {
        throw std::runtime_error(reason + "; " + e.what());
    }
#else
    return std::make_unique<OpenCvVideoWriter>(path, width, height, fps);
#endif
}

} // namespace trdrop
