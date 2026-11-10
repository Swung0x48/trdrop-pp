#include "motion_video.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "video_writer.h"

namespace trdrop {

namespace {

//! brightness of the picture under the arrows, and of the text box relative to that
constexpr float kDim = 0.6f;
constexpr float kBoxDim = 0.25f;
//! motion in pixels (of the measured resolution) that gets the last color
constexpr double kLongestMotion = 45.0;
//! the dark end of the color map is left out
constexpr int kFirstColor = 48;
//! arrows shorter than this many pixels are drawn as a dot
constexpr double kShortestArrow = 1.5;
//! the tips of an arrow: this long (at most half the arrow), at 45 degrees
constexpr double kTipLength = 6.0;
//! half the size of the x of a rejected point
constexpr float kCrossSize = 2.0f;
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

//! table for cv::LUT: 8-bit values times `factor`
cv::Mat scale_table(float factor)
{
    cv::Mat table(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i) table.at<uchar>(i) = cv::saturate_cast<uchar>(i * factor);
    return table;
}

//! draws the scene with OpenCV and encodes the pictures with a VideoWriter
class CpuMotionVideoTarget final : public MotionVideoTarget {
public:
    CpuMotionVideoTarget(const std::string & path, cv::Size size, double fps)
        : _size(size), _writer(make_video_writer(path, size.width, size.height, fps))
    { }

    std::string description() const override { return _writer->description() + ", drawn on the CPU"; }

    void write(const MotionDetail & detail, const MotionVideoScene & scene) override
    {
        if (detail.picture.empty()) throw std::runtime_error("motion video: the pipeline didn't keep the picture of the frame");
        cv::cvtColor(detail.picture, _picture, cv::COLOR_YUV2BGR_I420);
        // the size of the frames changed within the video
        if (_picture.size() != _size) cv::resize(_picture, _picture, _size, 0, 0, cv::INTER_AREA);
        if (_dim != scene.dim)
        {
            _dim = scene.dim;
            _dim_table = scale_table(_dim);
        }
        cv::LUT(_picture, _dim_table, _picture);

        for (const MotionVideoSegment & segment : scene.segments)
        {
            const cv::Scalar color(segment.blue, segment.green, segment.red);
            if (segment.x0 == segment.x1 && segment.y0 == segment.y1)
            {
                cv::circle(_picture, fixed_point(segment.x0, segment.y0), static_cast<int>(std::lround(segment.radius * (1 << kShift))),
                           color, cv::FILLED, cv::LINE_AA, kShift);
            }
            else
            {
                cv::line(_picture, fixed_point(segment.x0, segment.y0), fixed_point(segment.x1, segment.y1), color,
                         std::max(1, static_cast<int>(std::lround(2.0f * segment.radius))), cv::LINE_AA, kShift);
            }
        }

        const cv::Rect box = scene.box & cv::Rect(cv::Point(0, 0), _size);
        if (!box.empty() && scene.text.size() == scene.box.size())
        {
            if (_box_dim != scene.box_dim)
            {
                _box_dim = scene.box_dim;
                _box_table = scale_table(_box_dim);
            }
            cv::Mat region = _picture(box);
            cv::LUT(region, _box_table, region);
            // white text, by its coverage
            for (int y = 0; y < box.height; ++y)
            {
                const uchar * coverage = scene.text.ptr<uchar>(y + box.y - scene.box.y) + (box.x - scene.box.x);
                cv::Vec3b * pixels = region.ptr<cv::Vec3b>(y);
                for (int x = 0; x < box.width; ++x)
                {
                    const int a = coverage[x];
                    if (a == 0) continue;
                    for (int c = 0; c < 3; ++c) pixels[x][c] = static_cast<uchar>(pixels[x][c] + (255 - pixels[x][c]) * a / 255);
                }
            }
        }
        _writer->write(_picture);
    }

    void finish() override { _writer->finish(); }

private:
    cv::Size _size;
    std::unique_ptr<VideoWriter> _writer;
    cv::Mat _picture;
    float _dim = -1.0f;
    float _box_dim = -1.0f;
    cv::Mat _dim_table;
    cv::Mat _box_table;
};

} // namespace

std::unique_ptr<MotionVideoTarget> make_cpu_motion_video_target(const std::string & path, cv::Size size, double fps)
{
    return std::make_unique<CpuMotionVideoTarget>(path, size, fps);
}

MotionVideo::MotionVideo(const std::string & path, int frame_width, int frame_height, double fps, double arrow_scale,
                         std::unique_ptr<MotionVideoTarget> target)
    : _path(path)
    , _size(motion_picture_size(frame_width, frame_height))
    , _arrow_scale(arrow_scale)
    , _target(std::move(target))
{
    if (_size.width < 2 || _size.height < 2) throw std::runtime_error("the video is too small for a motion video");
    if (!_target) _target = make_cpu_motion_video_target(path, _size, fps);

    cv::Mat ramp(256, 1, CV_8UC1), colors;
    for (int i = 0; i < 256; ++i) ramp.at<uchar>(i) = static_cast<uchar>(i);
    cv::applyColorMap(ramp, colors, cv::COLORMAP_TURBO);
    for (int i = 0; i < 256; ++i) _colors.push_back(colors.at<cv::Vec3b>(i));

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
    const cv::Size box(_text(nullptr, widest, 0, 0, true) + kPadding, kLines * _line_height + kPadding);
    _scene.dim = kDim;
    _scene.box_dim = kBoxDim;
    // no text on pictures it doesn't fit on
    if (box.width <= _size.width - 2 * kMargin && box.height <= _size.height - 2 * kMargin)
    {
        _scene.box = cv::Rect(cv::Point(kMargin, kMargin), box);
        _scene.text.create(box, CV_8UC1);
    }
}

int MotionVideo::_label(cv::Mat * target, int x, int line, const std::string & text) const
{
    int baseline = 0;
    const int width = cv::getTextSize(text, kFont, _font_scale, 1, &baseline).width;
    if (target)
    {
        const int y = kPadding / 2 + _line_height * (line + 1) - _line_height / 4;
        cv::putText(*target, text, cv::Point(x, y), kFont, _font_scale, cv::Scalar(255), 1, cv::LINE_AA);
    }
    return x + width;
}

int MotionVideo::_number(cv::Mat * target, int x, int line, int cells, const std::string & text) const
{
    const int end = x + cells * _digit_width;
    if (target)
    {
        const int y = kPadding / 2 + _line_height * (line + 1) - _line_height / 4;
        int baseline = 0;
        for (size_t k = 0; k < text.size(); ++k)
        {
            // from the right: the last character is in the last cell
            const std::string character(1, text[text.size() - 1 - k]);
            const int width = cv::getTextSize(character, kFont, _font_scale, 1, &baseline).width;
            const int cell = end - static_cast<int>(k + 1) * _digit_width;
            cv::putText(*target, character, cv::Point(cell + (_digit_width - width) / 2, y), kFont, _font_scale, cv::Scalar(255), 1, cv::LINE_AA);
        }
    }
    return end;
}

int MotionVideo::_text(cv::Mat * target, const MotionVideoFrame & values, int used, int rejected, bool counted) const
{
    const int left = kPadding;
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

void MotionVideo::_add_points(const std::vector<MotionPoint> & points, bool dimmed)
{
    const float thickness = _size.width >= 1280 ? 2.0f : 1.0f;
    for (const MotionPoint & point : points)
    {
        if (point.state == MotionPoint::Rejected)
        {
            MotionVideoSegment arm;
            arm.radius = 0.5f;
            arm.blue = 40;
            arm.green = 40;
            arm.red = 230;
            arm.x0 = point.x - kCrossSize;
            arm.y0 = point.y - kCrossSize;
            arm.x1 = point.x + kCrossSize;
            arm.y1 = point.y + kCrossSize;
            _scene.segments.push_back(arm);
            std::swap(arm.y0, arm.y1);
            _scene.segments.push_back(arm);
        }
        if (point.state != MotionPoint::Used) continue;
        const double length = std::hypot(point.dx, point.dy);
        cv::Vec3b color = _colors[static_cast<size_t>(std::min(255.0, kFirstColor + (255 - kFirstColor) * length / kLongestMotion))];
        if (dimmed)
        {
            for (int c = 0; c < 3; ++c) color[c] = cv::saturate_cast<uchar>(0.45 * color[c] + 60.0);
        }
        MotionVideoSegment segment;
        segment.blue = color[0];
        segment.green = color[1];
        segment.red = color[2];
        segment.x0 = segment.x1 = point.x;
        segment.y0 = segment.y1 = point.y;
        const double drawn = length * _arrow_scale;
        if (drawn < kShortestArrow)
        {
            segment.radius = thickness;
            _scene.segments.push_back(segment);
            continue;
        }
        segment.radius = 0.5f * thickness;
        segment.x1 = static_cast<float>(point.x + point.dx * _arrow_scale);
        segment.y1 = static_cast<float>(point.y + point.dy * _arrow_scale);
        _scene.segments.push_back(segment);
        // the two lines of the tip, back from the end of the arrow
        const double tip = std::min(0.5 * drawn, kTipLength);
        const double angle = std::atan2(segment.y0 - segment.y1, segment.x0 - segment.x1);
        for (const double side : { CV_PI / 4.0, -CV_PI / 4.0 })
        {
            MotionVideoSegment line = segment;
            line.x0 = static_cast<float>(segment.x1 + tip * std::cos(angle + side));
            line.y0 = static_cast<float>(segment.y1 + tip * std::sin(angle + side));
            _scene.segments.push_back(line);
        }
    }
}

void MotionVideo::write(const MotionVideoFrame & values, const MotionDetail & detail)
{
    _scene.segments.clear();
    int used = 0, rejected = 0;
    const bool counted = !detail.points.empty() && !(values.compared && values.duplicate);
    if (!detail.points.empty())
    {
        if (values.compared && values.duplicate && !_held.empty())
        {
            _add_points(_held, true);
        }
        else
        {
            _add_points(detail.points, false);
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
    if (!_scene.box.empty())
    {
        _scene.text.setTo(0);
        _text(&_scene.text, values, used, rejected, counted);
    }
    _target->write(detail, _scene);
    ++_frames;
}

} // namespace trdrop
