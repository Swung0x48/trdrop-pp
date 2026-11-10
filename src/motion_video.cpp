#include "motion_video.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace trdrop {

namespace {

//! brightness of the picture under the arrows
constexpr double kDim = 0.6;
//! motion in pixels (of the measured resolution) that gets the last color
constexpr double kLongestMotion = 45.0;
//! the dark end of the color map is left out
constexpr int kFirstColor = 48;
//! arrows shorter than this many pixels are drawn as a dot
constexpr double kShortestArrow = 1.5;
//! positions are passed to OpenCV with this many fractional bits: the arrows follow sub pixel motion
constexpr int kShift = 3;

constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;
constexpr int kMargin = 8;
constexpr int kPadding = 10;
constexpr int kLines = 4;

cv::Point fixed_point(double x, double y)
{
    return cv::Point(static_cast<int>(std::lround(x * (1 << kShift))), static_cast<int>(std::lround(y * (1 << kShift))));
}

std::string format(const char * pattern, double value)
{
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), pattern, value);
    return buffer;
}

} // namespace

MotionVideo::MotionVideo(const std::string & path, int frame_width, int frame_height, double fps, double arrow_scale)
    : _path(path)
    , _size(motion_picture_size(frame_width, frame_height))
    , _arrow_scale(arrow_scale)
{
    if (_size.width < 2 || _size.height < 2) throw std::runtime_error("the video is too small for a motion video");
    _writer = make_video_writer(path, _size.width, _size.height, fps);

    cv::Mat ramp(256, 1, CV_8UC1), colors;
    for (int i = 0; i < 256; ++i) ramp.at<uchar>(i) = static_cast<uchar>(i);
    cv::applyColorMap(ramp, colors, cv::COLORMAP_TURBO);
    for (int i = 0; i < 256; ++i)
    {
        const cv::Vec3b color = colors.at<cv::Vec3b>(i);
        _colors.emplace_back(color[0], color[1], color[2]);
    }

    // the text: smaller for small pictures, every digit gets the width of the widest one
    _font_scale = std::clamp(0.6 * _size.width / 1280.0, 0.4, 0.6);
    int baseline = 0;
    for (char digit = '0'; digit <= '9'; ++digit)
    {
        _digit_width = std::max(_digit_width, cv::getTextSize(std::string(1, digit), kFont, _font_scale, 1, &baseline).width);
    }
    _space = _digit_width;
    _line_height = static_cast<int>(std::lround(42.0 * _font_scale));
    for (const char * state : { "new", "duplicate", "torn", "first frame" })
    {
        _state_width = std::max(_state_width, cv::getTextSize(state, kFont, _font_scale, 1, &baseline).width);
    }
    // the box holds the longest text there can be
    MotionVideoFrame widest;
    widest.compared = true;
    widest.sample.measurable = true;
    widest.sample.has_error = true;
    widest.sample.has_perceived_fps = true;
    widest.sample.lost = 1;
    _box_width = _text(nullptr, widest, 0, 0, true) - kMargin + kPadding;
    // no text on pictures it doesn't fit on
    if (_box_width > _size.width - 2 * kMargin || kLines * _line_height + kPadding > _size.height - 2 * kMargin) _box_width = 0;
}

int MotionVideo::_label(cv::Mat * target, int x, int line, const std::string & text) const
{
    int baseline = 0;
    const int width = cv::getTextSize(text, kFont, _font_scale, 1, &baseline).width;
    if (target)
    {
        const int y = kMargin + kPadding / 2 + _line_height * (line + 1) - _line_height / 4;
        cv::putText(*target, text, cv::Point(x, y), kFont, _font_scale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    return x + width;
}

int MotionVideo::_number(cv::Mat * target, int x, int line, int cells, const std::string & text) const
{
    const int end = x + cells * _digit_width;
    if (target)
    {
        const int y = kMargin + kPadding / 2 + _line_height * (line + 1) - _line_height / 4;
        int baseline = 0;
        for (size_t k = 0; k < text.size(); ++k)
        {
            // from the right: the last character is in the last cell
            const std::string character(1, text[text.size() - 1 - k]);
            const int width = cv::getTextSize(character, kFont, _font_scale, 1, &baseline).width;
            const int cell = end - static_cast<int>(k + 1) * _digit_width;
            cv::putText(*target, character, cv::Point(cell + (_digit_width - width) / 2, y), kFont, _font_scale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }
    return end;
}

int MotionVideo::_text(cv::Mat * target, const MotionVideoFrame & values, int used, int rejected, bool counted) const
{
    const int left = kMargin + kPadding;
    int widest = 0;

    // frame 1234   new   2104 points used   37 rejected   (arrows x8)
    int x = _label(target, left, 0, "frame");
    x = _number(target, x, 0, 8, std::to_string(values.frame)) + 2 * _space;
    _label(target, x, 0, !values.compared ? "first frame" : values.duplicate ? "duplicate" : values.torn ? "torn" : "new");
    x += _state_width + _space;
    x = _number(target, x, 0, 5, counted ? std::to_string(used) : "") + _space;
    x = _label(counted ? target : nullptr, x, 0, "points used") + _space;
    x = _number(target, x, 0, 5, counted ? std::to_string(rejected) : "") + _space;
    x = _label(counted ? target : nullptr, x, 0, "rejected");
    if (_arrow_scale != 1.0) x = _label(target, x + 2 * _space, 0, "arrows x" + format("%g", _arrow_scale));
    widest = std::max(widest, x);

    // animation step 1.05 intervals   votes spread 0.26
    x = _label(target, left, 1, "animation step");
    if (values.sample.measurable)
    {
        x = _number(target, x, 1, 7, format("%.2f", values.sample.step)) + _space;
        x = _label(target, x, 1, "intervals") + 2 * _space;
        x = _label(target, x, 1, "votes spread");
        x = _number(target, x, 1, 6, format("%.2f", values.sample.step_spread));
    }
    else
    {
        x = _label(target, x + 2 * _space, 1, "not measurable");
    }
    widest = std::max(widest, x);

    // animation error +0.4 ms   1 picture lost by the capture
    x = _label(target, left, 2, "animation error");
    x = _number(target, x, 2, 7, values.sample.has_error ? format("%+.1f", values.sample.error_ms) : "") + _space;
    x = _label(values.sample.has_error ? target : nullptr, x, 2, "ms");
    if (values.sample.lost > 0) x = _label(target, x + 2 * _space, 2, "1 picture lost by the capture");
    widest = std::max(widest, x);

    // framerate 119.0 fps   perceived smoothness 118.6 fps
    x = _label(target, left, 3, "framerate");
    x = _number(target, x, 3, 6, format("%.1f", values.framerate)) + _space;
    x = _label(target, x, 3, "fps") + 2 * _space;
    x = _label(target, x, 3, "perceived smoothness");
    x = _number(target, x, 3, 6, values.sample.has_perceived_fps ? format("%.1f", values.sample.perceived_fps) : "") + _space;
    x = _label(values.sample.has_perceived_fps ? target : nullptr, x, 3, "fps");
    widest = std::max(widest, x);
    return widest;
}

void MotionVideo::_draw_points(cv::Mat & picture, const std::vector<MotionPoint> & points, bool dimmed) const
{
    const int thickness = picture.cols >= 1280 ? 2 : 1;
    for (const MotionPoint & point : points)
    {
        if (point.state == MotionPoint::Rejected)
        {
            cv::drawMarker(picture, cv::Point(static_cast<int>(std::lround(point.x)), static_cast<int>(std::lround(point.y))),
                           cv::Scalar(40, 40, 230), cv::MARKER_TILTED_CROSS, 5, 1, cv::LINE_AA);
        }
        if (point.state != MotionPoint::Used) continue;
        const double length = std::hypot(point.dx, point.dy);
        cv::Scalar color = _colors[static_cast<size_t>(std::min(255.0, kFirstColor + (255 - kFirstColor) * length / kLongestMotion))];
        if (dimmed) color = cv::Scalar(0.45 * color[0] + 60.0, 0.45 * color[1] + 60.0, 0.45 * color[2] + 60.0);
        const double drawn = length * _arrow_scale;
        if (drawn < kShortestArrow)
        {
            cv::circle(picture, fixed_point(point.x, point.y), thickness << kShift, color, cv::FILLED, cv::LINE_AA, kShift);
        }
        else
        {
            cv::arrowedLine(picture, fixed_point(point.x, point.y), fixed_point(point.x + point.dx * _arrow_scale, point.y + point.dy * _arrow_scale),
                            color, thickness, cv::LINE_AA, kShift, std::min(0.5, 6.0 / drawn));
        }
    }
}

void MotionVideo::write(const MotionVideoFrame & values, const MotionDetail & detail)
{
    cv::cvtColor(detail.picture, _picture, cv::COLOR_YUV2BGR_I420);
    // the size of the frames changed within the video
    if (_picture.size() != _size) cv::resize(_picture, _picture, _size, 0, 0, cv::INTER_AREA);
    _picture.convertTo(_picture, -1, kDim);

    int used = 0, rejected = 0;
    const bool counted = !detail.points.empty() && !(values.compared && values.duplicate);
    if (!detail.points.empty())
    {
        if (values.compared && values.duplicate && !_held.empty())
        {
            _draw_points(_picture, _held, true);
        }
        else
        {
            _draw_points(_picture, detail.points, false);
            _held = detail.points;
        }
        for (const MotionPoint & point : detail.points)
        {
            used += point.state == MotionPoint::Used;
            rejected += point.state == MotionPoint::Rejected;
        }
    }
    else
    {
        _held.clear();
    }

    if (_box_width > 0)
    {
        cv::Mat box = _picture(cv::Rect(kMargin, kMargin, _box_width, kLines * _line_height + kPadding));
        box.convertTo(box, -1, 0.25);
        _text(&_picture, values, used, rejected, counted);
    }

    _writer->write(_picture);
    ++_frames;
}

} // namespace trdrop
