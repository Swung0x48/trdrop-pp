#include "video_sources.h"

#include <cctype>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace trdrop {

VideoSources::VideoSources(const std::vector<std::string> & paths)
{
    for (const std::string & path : paths)
    {
        cv::VideoCapture capture(path);
        if (!capture.isOpened()) throw std::runtime_error("could not open video '" + path + "'");
        _captures.push_back(std::move(capture));
    }
}

bool VideoSources::read_next_frames(std::vector<cv::Mat> & frames)
{
    if (_captures.empty()) return false;
    frames.resize(_captures.size());
    for (size_t i = 0; i < _captures.size(); ++i)
    {
        if (!_captures[i].read(frames[i]) || frames[i].empty()) return false;
        // the analysis works on 8-bit BGR frames
        if (frames[i].type() != CV_8UC3)
        {
            cv::Mat converted;
            if (frames[i].channels() == 1) cv::cvtColor(frames[i], converted, cv::COLOR_GRAY2BGR);
            else if (frames[i].channels() == 4) cv::cvtColor(frames[i], converted, cv::COLOR_BGRA2BGR);
            else converted = frames[i];
            if (converted.depth() != CV_8U) converted.convertTo(converted, CV_8U);
            frames[i] = converted;
        }
    }
    ++_frames_read;
    return true;
}

double VideoSources::recorded_framerate(size_t index) const
{
    return _captures.at(index).get(cv::CAP_PROP_FPS);
}

long long VideoSources::reported_frame_count(size_t index) const
{
    return static_cast<long long>(_captures.at(index).get(cv::CAP_PROP_FRAME_COUNT));
}

int VideoSources::frame_width(size_t index) const
{
    return static_cast<int>(_captures.at(index).get(cv::CAP_PROP_FRAME_WIDTH));
}

int VideoSources::frame_height(size_t index) const
{
    return static_cast<int>(_captures.at(index).get(cv::CAP_PROP_FRAME_HEIGHT));
}

std::string VideoSources::codec_name(size_t index) const
{
    const unsigned fourcc = static_cast<unsigned>(static_cast<long long>(_captures.at(index).get(cv::CAP_PROP_FOURCC)));
    std::string name;
    for (int shift = 0; shift < 32; shift += 8)
    {
        const char c = static_cast<char>((fourcc >> shift) & 0xFF);
        // raw formats use numeric tags (BI_RGB = 0, 'BGR\x18', ...)
        if (c < 32 || c > 126) return "raw";
        name += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    while (!name.empty() && name.back() == ' ') name.pop_back();
    return name.empty() ? "raw" : name;
}

bool VideoSources::lossy_codec(size_t index) const
{
    // FOURCCs and the first 4 characters of FFmpeg's codec names (used by containers without a FOURCC)
    static const char * const lossless[] = {
        "RAW", "RAWV", "DIB", "FFV1", "HFYU", "HUFF", "FFVH", "MAGY", "MAGI", "UTVI", "LAGS", "LAGA", "ZMBV",
        "PNG", "MPNG", "BMP", "I420", "IYUV", "YV12", "NV12", "NV21", "YUY2", "YUYV", "UYVY", "Y42B", "Y444",
        "V210", "R210", "V308", "V408", "V410", "BGRA", "RGBA", "BGR0", "RGB0", "24BG",
    };
    const std::string name = codec_name(index);
    for (const char * candidate : lossless)
    {
        if (name == candidate) return false;
    }
    // Ut Video: ULRG, ULY0, ULH2, UQY2, ...
    if (name.size() == 4 && name[0] == 'U' && (name[1] == 'L' || name[1] == 'Q' || name[1] == 'M')) return false;
    return true;
}

} // namespace trdrop
