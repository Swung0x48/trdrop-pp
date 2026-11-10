#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace trdrop {

//! the motion of a frame is measured on a grid of cells
constexpr int kMotionGridX = 8;
constexpr int kMotionGridY = 5;
constexpr int kMotionCells = kMotionGridX * kMotionGridY;

//! motion from the previous frame to a frame: one vector per grid cell
struct MotionField {
    //! false: there was no previous frame of the same size
    bool estimated = false;
    //! source pixels per pixel of the resolution the motion was measured at
    float pixel_scale = 1.0f;
    //! per cell: motion in source pixels and the amount of tracked points it is the median of
    std::array<float, kMotionCells> dx {};
    std::array<float, kMotionCells> dy {};
    std::array<uint16_t, kMotionCells> points {};
};

//! Measures the motion between consecutive frames on the CPU: corners of the previous frame are
//! tracked into the current one (pyramidal Lucas-Kanade, sub pixel accurate) at up to 1920x1080
class MotionEstimator {
public:
    //! luma: 8-bit single channel frame in full resolution
    MotionField estimate(const cv::Mat & luma);

private:
    cv::Mat _current;
    cv::Size _previous_size;
    std::vector<cv::Mat> _previous_pyramid;
    std::vector<cv::Mat> _current_pyramid;
    std::vector<cv::Point2f> _corners;
    std::vector<cv::Point2f> _tracked;
    std::vector<unsigned char> _status;
    std::vector<float> _error;
    std::array<std::vector<float>, kMotionCells> _cell_dx;
    std::array<std::vector<float>, kMotionCells> _cell_dy;
};

//! what the motion says about one frame
struct MotionSample {
    //! the frame has enough coherent motion to measure its animation step
    bool measurable = false;
    //! animation step: how far the animation advanced since the previous frame, in display intervals
    //! (1 = as much as it advances per captured frame on average around this frame)
    double step = 0.0;
    //! animation error of a new frame: animation step since the previous new frame minus the time
    //! that frame was displayed
    bool has_error = false;
    double error_ms = 0.0;
    //! perceived smoothness over the last second
    bool has_perceived_fps = false;
    double perceived_fps = 0.0;
};

//! Turns the motion fields of a video into animation steps, the animation error and the perceived smoothness.
//!
//! The animation step of a frame is the common scale of its motion field relative to the average motion
//! field around it: m_cell(frame) = step * v_cell, v being the motion of the cell per display interval in a
//! window around the frame. It is the weighted median over the cells of m.v / |v|^2, so it works for pans,
//! forward motion and frames with static overlays.
//!
//! The window grows (up to 1 second) until it holds enough new frames.
//! The results are delayed by half the longest reference window (look ahead).
class MotionAnalyzer {
public:
    struct Summary {
        long long frames = 0;
        long long measurable_frames = 0;
        //! frames the pixel comparison counts as new, and how many of them have a usable motion
        long long new_frames = 0;
        long long measurable_new_frames = 0;
        //! new frames with an animation step
        long long animation_frames = 0;
        //! new frames per second while the motion was measurable
        double motion_fps = 0.0;
        //! framerate of evenly paced motion that would look as smooth: the new frames per second divided by
        //! 1 + (coefficient of variation of the steps)^2, over the whole video (a single long step lowers it a lot)
        double perceived_fps = 0.0;
        //! ... of the values over one second each: median and the 10th percentile (0 without a full second)
        double perceived_fps_median = 0.0;
        double perceived_fps_low = 0.0;
        double mean_abs_error_ms = 0.0;
        double median_abs_error_ms = 0.0;
        double p95_abs_error_ms = 0.0;
        double max_abs_error_ms = 0.0;
    };

    //! fps: recorded framerate, window_ms: length of the reference window (0: automatic)
    MotionAnalyzer(double fps, double window_ms);

    //! next frame. duplicate: the pixel comparison counts the frame as a duplicate,
    //! changed_share: share (0 - 1) of its pixels which are different
    void push(const MotionField & field, bool duplicate, double changed_share);
    //! no more frames follow, the remaining samples become available
    void finish();
    //! the next sample in frame order, false if it isn't available yet
    bool pop(MotionSample & sample);
    bool ready() const { return !_ready.empty(); }

    Summary summary() const;

private:
    struct Entry {
        MotionField field;
        bool duplicate = false;
        double changed_share = 0.0;
    };

    void _finalize_available(bool flush);
    MotionSample _finalize(size_t position);
    void _reset_chain();

    double _fps;
    size_t _half_window;
    size_t _max_half_window;
    size_t _rolling_window;

    //! frames [_first_index, _first_index + _entries.size())
    std::deque<Entry> _entries;
    long long _first_index = 0;
    long long _finalized = 0;
    std::deque<MotionSample> _ready;

    // animation chain: steps since the last new frame
    bool _chain = false;
    double _accumulated_step = 0.0;
    long long _shown = 0;

    // perceived smoothness over the last second
    struct RollingStep {
        long long frame;
        double step;
        double displayed;
    };
    std::deque<RollingStep> _rolling_steps;
    std::deque<unsigned char> _rolling_measurable;
    long long _rolling_measurable_count = 0;

    // summary
    long long _measurable_frames = 0;
    long long _new_frames = 0;
    long long _measurable_new_frames = 0;
    long long _animation_frames = 0;
    long long _shown_total = 0;
    double _step_sum = 0.0;
    double _step_square_sum = 0.0;
    std::vector<float> _errors;
    std::vector<float> _rolling_values;
    // scratch of _finalize
    std::vector<double> _group_weight;
    std::vector<double> _group_scale;
};

} // namespace trdrop
