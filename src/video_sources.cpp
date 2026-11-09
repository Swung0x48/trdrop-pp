#include "video_sources.h"

#include <algorithm>
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

double VideoSources::progress() const
{
    double progress = -1.0;
    for (size_t i = 0; i < _captures.size(); ++i)
    {
        const long long total = reported_frame_count(i);
        if (total <= 0) continue;
        const double p = static_cast<double>(_frames_read) / static_cast<double>(total);
        progress = std::max(progress, p);
    }
    return std::min(progress, 1.0);
}

} // namespace trdrop
