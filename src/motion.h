#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace trdrop {

//! The motion of a frame is kept per cell of a fine grid. Every cell with motion later has one vote on the
//! animation step of the frame, so a moving part of the picture is enough (a cell is 1 / 640 of it).
constexpr int kMotionGridX = 32;
constexpr int kMotionGridY = 20;
constexpr int kMotionCells = kMotionGridX * kMotionGridY;
//! The motion is measured at a regular grid of points, this many per cell in both directions. A point is used if
//! its surroundings have texture in both directions, the frames match there after the tracking and tracking
//! it back ends at most kMotionMaxBackError pixels (of the measured resolution) from the point: wrong matches -
//! repeating texture, motion too large - don't come back. A cell needs kMotionMinCellPoints of its points.
constexpr int kMotionCellPoints = 2;
constexpr int kMotionPointsX = kMotionCellPoints * kMotionGridX;
constexpr int kMotionPointsY = kMotionCellPoints * kMotionGridY;
constexpr float kMotionMinTexture = 1.0f; // smaller eigenvalue of the gradient matrix per pixel of the window
constexpr float kMotionMaxBackError = 1.0f;
constexpr size_t kMotionMinCellPoints = 2;

//! motion from the previous frame to a frame: one vector per grid cell
struct MotionField {
    //! false: there was no previous frame of the same size
    bool estimated = false;
    //! source pixels per pixel of the resolution the motion was measured at
    float pixel_scale = 1.0f;
    //! per cell: motion in source pixels and the amount of tracked points it is the median of (0: none)
    std::array<float, kMotionCells> dx {};
    std::array<float, kMotionCells> dy {};
    std::array<uint8_t, kMotionCells> points {};
};

//! collects the motion of tracked points and reduces it to the median per cell
class MotionCellBinner {
public:
    void clear() { _points.clear(); }
    //! x, y: position of the point relative to the frame (0 - 1), dx, dy: its motion
    void add(float x, float y, float dx, float dy);
    //! sets the cells with at least min_points points, their motion multiplied by scale
    void finish(MotionField & field, size_t min_points, float scale);

private:
    struct Point {
        uint16_t cell;
        float dx;
        float dy;
    };
    std::vector<Point> _points;
    std::vector<float> _values;
};

//! a point of the grid as it was tracked, in pixels of the resolution the motion is measured at
struct MotionPoint {
    enum State : uint8_t {
        NoTexture = 0, //!< nothing to track at the point
        Rejected = 1,  //!< tracked, but tracking it back doesn't end at the point
        Used = 2,
    };
    float x = 0.0f;
    float y = 0.0f;
    float dx = 0.0f;
    float dy = 0.0f;
    State state = NoTexture;
};

//! what the motion video shows of a frame
struct MotionDetail {
    //! the frame at the resolution the motion is measured at (see motion_picture_size), as I420: the Y plane
    //! followed by the U and V planes of half the size, rows * 3 / 2 rows of 8-bit values
    cv::Mat picture;
    //! instead of `picture` if the pipeline draws the motion video on the GPU: its handle of the frame's picture
    int gpu_picture = -1;
    //! the points of the grid, empty if there was no previous frame of the same size
    std::vector<MotionPoint> points;
};

//! the motion of a frame is measured at 1 / this of its resolution (at up to 1920x1080)
int motion_scale(int width, int height);
//! size of MotionDetail::picture for a frame: the resolution the motion is measured at, made even
cv::Size motion_picture_size(int width, int height);

//! Measures the motion between consecutive frames on the CPU: the points of the grid are tracked from the
//! previous frame into the current one and back (pyramidal Lucas-Kanade, sub pixel accurate) at up to 1920x1080.
class MotionEstimator {
public:
    //! luma: 8-bit single channel frame in full resolution, points: if given, receives every point of the grid
    MotionField estimate(const cv::Mat & luma, std::vector<MotionPoint> * points = nullptr);

private:
    cv::Mat _current;
    cv::Size _previous_size;
    std::vector<cv::Mat> _previous_pyramid;
    std::vector<cv::Mat> _current_pyramid;
    std::vector<cv::Point2f> _points;
    cv::Size _points_size;
    std::vector<cv::Point2f> _tracked;
    std::vector<cv::Point2f> _back;
    std::vector<unsigned char> _status;
    std::vector<unsigned char> _back_status;
    std::vector<float> _error;
    MotionCellBinner _binner;
};

//! what the motion says about one frame
struct MotionSample {
    //! the frame has enough coherent motion to measure its animation step
    bool measurable = false;
    //! animation step: how far the animation advanced since the previous frame, in display intervals
    //! (1 = as much as it advances per captured frame on average around this frame)
    double step = 0.0;
    //! how much the cells disagree about the step: scaled median absolute deviation of their votes, in display
    //! intervals. 0 for rigid motion; on real footage the error of a step has about this size
    double step_spread = 0.0;
    //! animation error of a new frame: animation step since the previous new frame minus the time
    //! that frame was displayed
    bool has_error = false;
    double error_ms = 0.0;
    //! asynchronous capture: pictures the capture lost between the previous new frame and this one
    int lost = 0;
    //! perceived smoothness over the last second
    bool has_perceived_fps = false;
    double perceived_fps = 0.0;
};

//! Turns the motion fields of a video into animation steps, the animation error and the perceived smoothness.
//!
//! The animation step of a frame is the common scale of its motion field relative to the average motion
//! field around it: m_cell(frame) = step * v_cell, v being the motion of the cell per display interval in a
//! window around the frame. Every cell with motion votes with m.v / |v|^2 and the weighted median wins, so it
//! works for pans, forward motion, static overlays and if only a part of the picture moves. The steps are scaled
//! so that the animation advances by one interval per displayed interval on average over the window. The cells
//! never agree exactly in real footage (the depth of what a cell shows changes); how much they disagree is
//! reported with the step instead of rejecting frames.
//!
//! The window grows (up to 1 second) until it holds enough new frames.
//! The results are delayed by half the longest reference window (look ahead).
//!
//! Asynchronous capture: if the capture doesn't run in step with what it records (variable refresh rate, a
//! capture program with its own clock), a picture shown at time t is first recorded at the next tick of the
//! capture, up to one interval later, and a picture shown for less than an interval may not be recorded at
//! all. A picture that is recorded an interval late looks like a repeated frame, the one that makes up for it
//! like a jump: animation errors of -1 and +1 interval which are not in what was recorded. In this mode
//!  - the time a picture was shown is only known to one interval: the animation error is what is left if
//!    every picture may have been shown up to one interval before it was recorded. The drift (animation time
//!    minus recorded time) may move freely within a band of one interval, only what pushes the band counts.
//!  - a step twice as long as the steps around it, after a picture which was recorded once, stands for two
//!    pictures of which the capture lost one: it counts as two steps. The same step after a
//!    picture that was recorded twice (the capture repeated one picture and lost the next, or the game showed
//!    it twice) counts as two as well while the capture demonstrably loses pictures.
//! A game that really shows a picture an interval late and makes up for it later looks exactly the same, so this
//! is an assumption about the capture and not something the video can tell.
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
        //! median spread of the votes on the steps of the new frames, as time
        double median_step_spread_ms = 0.0;
        //! share (0 - 1) of the animation errors which are about one whole interval too short / too long:
        //! many of both is what an asynchronous capture looks like without async_capture
        double interval_errors_short = 0.0;
        double interval_errors_long = 0.0;
        //! asynchronous capture: new frames whose display time was moved by more than half an interval,
        //! and pictures the capture lost (they are counted in animation_frames, motion_fps and the smoothness)
        long long retimed_frames = 0;
        long long lost_frames = 0;
        //! ... of them after a picture that was recorded twice (attributed to the capture, not proven)
        long long lost_after_repeat = 0;
    };

    //! fps: recorded framerate, window_ms: length of the reference window (0: automatic),
    //! async_capture: the capture didn't run in step with what it recorded (see above)
    MotionAnalyzer(double fps, double window_ms, bool async_capture = false);

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

    // asynchronous capture: the drift of the current new frame is 0, the lower edge of the band of one
    // interval in which it may move lies somewhere in [_band_low, _band_high]
    bool _async_capture;
    double _band_low = -1.0;
    double _band_high = 0.0;
    long long _retimed_frames = 0;
    long long _lost_frames = 0;
    long long _lost_after_repeat = 0;
    //! frames at which a picture was missing, of the last second
    std::deque<long long> _lost_evidence;

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
    std::vector<float> _step_spread;
    // scratch of _finalize
    //! a new frame of the window with the intervals since the previous new frame
    struct Group {
        const MotionField * field;
        //! sum of the weights of its intervals, and divided by their number
        double weight;
        double scale;
    };
    //! reference motion of a cell: v / |v|^2 and |v|
    struct Reference {
        bool usable = false;
        double x = 0.0;
        double y = 0.0;
        double speed = 0.0;
    };
    std::vector<Group> _groups;
    std::array<Reference, kMotionCells> _reference;
    std::vector<float> _recent_steps;
};

} // namespace trdrop
