#include "motion.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace trdrop {

namespace {

//! the motion is measured at up to this resolution
constexpr double kMaxWidth = 1920.0;
constexpr double kMaxHeight = 1080.0;
constexpr int kMaxCorners = 800;
//! a cell needs this many tracked points
constexpr int kMinPoints = 3;

//! reference motion of a cell: at least this speed (pixels per frame at the measured resolution) ...
constexpr double kMinSpeed = 0.5;
//! ... and tracked in this share of the frames of the window
constexpr double kMinTrackedShare = 0.6;
//! a frame needs this many usable cells
constexpr int kMinCells = 6;
//! a frame whose animation step is smaller didn't advance the animation
constexpr double kMinStep = 0.3;
//! ... it is a duplicate if less than this share of its pixels changed, otherwise the scene changed without moving
constexpr double kDuplicateShare = 0.02;
//! automatic reference window: this long, growing up to kMaxWindowSeconds until it holds kMinWindowFrames new frames
constexpr double kDefaultWindowMs = 400.0;
constexpr double kMaxWindowSeconds = 2.0;
constexpr size_t kMinWindowFrames = 16;
//! the cells have to agree on the step of a frame: their median deviation from it, relative to max(1, step),
//! may be at most this. They disagree if the motion isn't the average motion scaled (turns, cuts, moving objects)
constexpr double kMaxStepSpread = 0.2;
//! the perceived smoothness needs this many steps and this share of measurable frames in the last second
constexpr size_t kMinRollingSteps = 6;
constexpr double kMinRollingMeasurable = 0.8;

float median(std::vector<float> & values)
{
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

} // namespace

MotionField MotionEstimator::estimate(const cv::Mat & luma)
{
    MotionField field;
    const int scale = std::max(1, static_cast<int>(std::ceil(std::max(luma.cols / kMaxWidth, luma.rows / kMaxHeight) - 1e-9)));
    field.pixel_scale = static_cast<float>(scale);
    if (scale > 1) cv::resize(luma, _current, cv::Size(luma.cols / scale, luma.rows / scale), 0, 0, cv::INTER_AREA);
    else luma.copyTo(_current);
    // the pyramid (with derivatives: image and derivative per level) is reused as the previous one of the next frame
    const cv::Size window(21, 21);
    const int levels = cv::buildOpticalFlowPyramid(_current, _current_pyramid, window, 3, true);

    if (!_previous_pyramid.empty() && _previous_size == _current.size())
    {
        field.estimated = true;
        // corners are detected at half the resolution (level 1 of the pyramid) for large frames, they only seed the tracking
        const int corner_level = _current.cols > 960 && levels >= 1 ? 1 : 0;
        const cv::Mat & corner_image = _previous_pyramid[static_cast<size_t>(2 * corner_level)];
        const double min_distance = std::max(4.0, corner_image.cols / 80.0);
        cv::goodFeaturesToTrack(corner_image, _corners, kMaxCorners, 0.01, min_distance, cv::noArray(), corner_level == 1 ? 5 : 7);
        if (corner_level == 1)
        {
            for (cv::Point2f & corner : _corners) corner = cv::Point2f(corner.x * 2.0f + 0.5f, corner.y * 2.0f + 0.5f);
        }
        if (_corners.size() >= 8)
        {
            cv::calcOpticalFlowPyrLK(_previous_pyramid, _current_pyramid, _corners, _tracked, _status, _error, window, levels,
                                     cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.01));
            for (int cell = 0; cell < kMotionCells; ++cell)
            {
                _cell_dx[static_cast<size_t>(cell)].clear();
                _cell_dy[static_cast<size_t>(cell)].clear();
            }
            for (size_t p = 0; p < _corners.size(); ++p)
            {
                if (!_status[p]) continue;
                const int cell_x = std::min(kMotionGridX - 1, std::max(0, static_cast<int>(_corners[p].x) * kMotionGridX / _current.cols));
                const int cell_y = std::min(kMotionGridY - 1, std::max(0, static_cast<int>(_corners[p].y) * kMotionGridY / _current.rows));
                const size_t cell = static_cast<size_t>(cell_y * kMotionGridX + cell_x);
                _cell_dx[cell].push_back(_tracked[p].x - _corners[p].x);
                _cell_dy[cell].push_back(_tracked[p].y - _corners[p].y);
            }
            for (size_t cell = 0; cell < static_cast<size_t>(kMotionCells); ++cell)
            {
                if (_cell_dx[cell].size() < static_cast<size_t>(kMinPoints)) continue;
                field.points[cell] = static_cast<uint16_t>(std::min<size_t>(_cell_dx[cell].size(), 65535));
                field.dx[cell] = median(_cell_dx[cell]) * field.pixel_scale;
                field.dy[cell] = median(_cell_dy[cell]) * field.pixel_scale;
            }
        }
    }
    _previous_size = _current.size();
    std::swap(_previous_pyramid, _current_pyramid);
    return field;
}

MotionAnalyzer::MotionAnalyzer(double fps, double window_ms)
    : _fps(fps)
    , _half_window(static_cast<size_t>(std::max(2.0, std::round((window_ms > 0.0 ? window_ms : kDefaultWindowMs) / 2000.0 * fps))))
    // a given window is used as it is, the automatic one may grow
    , _max_half_window(window_ms > 0.0 ? _half_window
                                       : std::max(_half_window, static_cast<size_t>(std::round(kMaxWindowSeconds / 2.0 * fps))))
    , _rolling_window(static_cast<size_t>(std::max(1.0, std::ceil(fps - 1e-9))))
{ }

void MotionAnalyzer::push(const MotionField & field, bool duplicate, double changed_share)
{
    Entry entry;
    entry.field = field;
    entry.duplicate = duplicate;
    entry.changed_share = changed_share;
    _entries.push_back(entry);
    _finalize_available(false);
}

void MotionAnalyzer::finish()
{
    _finalize_available(true);
}

bool MotionAnalyzer::pop(MotionSample & sample)
{
    if (_ready.empty()) return false;
    sample = _ready.front();
    _ready.pop_front();
    return true;
}

void MotionAnalyzer::_finalize_available(bool flush)
{
    const long long end = _first_index + static_cast<long long>(_entries.size());
    // a frame needs the frames of the half window after it
    while (_finalized < end && (flush || _finalized + static_cast<long long>(_max_half_window) < end))
    {
        _ready.push_back(_finalize(static_cast<size_t>(_finalized - _first_index)));
        ++_finalized;
        // frames before the window of the next frame are not needed anymore
        while (_first_index + static_cast<long long>(_max_half_window) < _finalized)
        {
            _entries.pop_front();
            ++_first_index;
        }
    }
}

void MotionAnalyzer::_reset_chain()
{
    _chain = false;
    _accumulated_step = 0.0;
    _shown = 0;
}

MotionSample MotionAnalyzer::_finalize(size_t position)
{
    MotionSample sample;
    const Entry & entry = _entries[position];
    // the window grows until it holds enough new frames: with a low framerate (e.g. 15 fps in 60) the average
    // of a short window is dominated by the few steps in it and hides their differences
    size_t half_window = _half_window;
    size_t first = 0, last = 0;
    while (true)
    {
        first = position >= half_window ? position - half_window : 0;
        last = std::min(_entries.size() - 1, position + half_window);
        if (half_window >= _max_half_window) break;
        size_t new_frames = 0;
        for (size_t k = first; k <= last; ++k) new_frames += _entries[k].duplicate ? 0 : 1;
        if (new_frames >= kMinWindowFrames) break;
        ++half_window;
    }
    // triangular weights: the mean of a periodic pattern (e.g. a new frame every 2nd frame) then hardly
    // depends on where the window starts and ends
    auto weight = [&](size_t k) {
        const size_t distance = k > position ? k - position : position - k;
        return static_cast<double>(half_window + 1 - distance);
    };
    double window = 0.0;
    for (size_t k = first; k <= last; ++k) window += weight(k);

    // animation step: weighted median over the cells of m.v / |v|^2 (v: mean motion of the cell in the window)
    std::array<std::pair<double, double>, kMotionCells> ratios; // (ratio, weight)
    size_t usable = 0;
    double weight_sum = 0.0;
    if (entry.field.estimated)
    {
        const double min_speed = kMinSpeed * entry.field.pixel_scale;
        for (size_t cell = 0; cell < static_cast<size_t>(kMotionCells); ++cell)
        {
            if (entry.field.points[cell] < kMinPoints) continue;
            double vx = 0.0, vy = 0.0, tracked = 0.0;
            for (size_t k = first; k <= last; ++k)
            {
                const MotionField & field = _entries[k].field;
                if (!field.estimated || field.points[cell] < kMinPoints) continue;
                const double w = weight(k);
                vx += w * field.dx[cell];
                vy += w * field.dy[cell];
                tracked += w;
            }
            vx /= window;
            vy /= window;
            const double speed_squared = vx * vx + vy * vy;
            if (speed_squared < min_speed * min_speed || tracked / window <= kMinTrackedShare) continue;
            const double speed = std::sqrt(speed_squared);
            ratios[usable++] = { (entry.field.dx[cell] * vx + entry.field.dy[cell] * vy) / speed_squared, speed };
            weight_sum += speed;
        }
    }
    if (usable >= static_cast<size_t>(kMinCells))
    {
        std::sort(ratios.begin(), ratios.begin() + static_cast<std::ptrdiff_t>(usable));
        double cumulative = 0.0;
        for (size_t k = 0; k < usable; ++k)
        {
            cumulative += ratios[k].second;
            if (cumulative >= 0.5 * weight_sum)
            {
                sample.step = ratios[k].first;
                break;
            }
        }
        // weighted median of the deviations from the step
        for (size_t k = 0; k < usable; ++k) ratios[k].first = std::fabs(ratios[k].first - sample.step);
        std::sort(ratios.begin(), ratios.begin() + static_cast<std::ptrdiff_t>(usable));
        double spread = 0.0;
        cumulative = 0.0;
        for (size_t k = 0; k < usable; ++k)
        {
            cumulative += ratios[k].second;
            if (cumulative >= 0.5 * weight_sum)
            {
                spread = ratios[k].first;
                break;
            }
        }
        sample.measurable = spread <= kMaxStepSpread * std::max(1.0, std::fabs(sample.step));
        if (!sample.measurable) sample.step = 0.0;
    }

    // animation chain: a new frame ends the step which started at the previous new frame
    if (!sample.measurable)
    {
        _reset_chain();
    }
    else
    {
        ++_measurable_frames;
        const bool small = sample.step < kMinStep;
        if (small && (entry.duplicate || entry.changed_share < kDuplicateShare))
        {
            // duplicate frame: the previous frame stays on screen
            if (_chain)
            {
                _accumulated_step += sample.step;
                ++_shown;
            }
        }
        else if (small)
        {
            // the scene changed but didn't move with the motion around it (e.g. the camera stopped)
            _reset_chain();
        }
        else
        {
            if (_chain)
            {
                _accumulated_step += sample.step;
                ++_shown;
                sample.has_error = true;
                sample.error_ms = (_accumulated_step - static_cast<double>(_shown)) * 1000.0 / _fps;
                _errors.push_back(static_cast<float>(sample.error_ms));
                ++_animation_frames;
                _shown_total += _shown;
                _step_sum += _accumulated_step;
                _step_square_sum += _accumulated_step * _accumulated_step;
                _rolling_steps.emplace_back(_finalized, _accumulated_step);
            }
            _chain = true;
            _accumulated_step = 0.0;
            _shown = 0;
        }
    }

    // perceived smoothness over the last second
    _rolling_measurable.push_back(sample.measurable ? 1 : 0);
    _rolling_measurable_count += sample.measurable ? 1 : 0;
    if (_rolling_measurable.size() > _rolling_window)
    {
        _rolling_measurable_count -= _rolling_measurable.front();
        _rolling_measurable.pop_front();
    }
    while (!_rolling_steps.empty() && _rolling_steps.front().first + static_cast<long long>(_rolling_window) <= _finalized)
    {
        _rolling_steps.pop_front();
    }
    if (_rolling_steps.size() >= kMinRollingSteps
        && static_cast<double>(_rolling_measurable_count) > kMinRollingMeasurable * static_cast<double>(_rolling_window))
    {
        double sum = 0.0, square_sum = 0.0;
        for (const auto & step : _rolling_steps)
        {
            sum += step.second;
            square_sum += step.second * step.second;
        }
        if (square_sum > 0.0)
        {
            sample.has_perceived_fps = true;
            sample.perceived_fps = _fps * sum / square_sum;
        }
    }
    return sample;
}

MotionAnalyzer::Summary MotionAnalyzer::summary() const
{
    Summary result;
    result.frames = _finalized;
    result.measurable_frames = _measurable_frames;
    result.animation_frames = _animation_frames;
    if (_shown_total > 0) result.motion_fps = _fps * static_cast<double>(_animation_frames) / static_cast<double>(_shown_total);
    if (_step_square_sum > 0.0) result.perceived_fps = _fps * _step_sum / _step_square_sum;
    if (!_errors.empty())
    {
        std::vector<float> absolute(_errors.size());
        double sum = 0.0;
        for (size_t i = 0; i < _errors.size(); ++i)
        {
            absolute[i] = std::fabs(_errors[i]);
            sum += absolute[i];
        }
        std::sort(absolute.begin(), absolute.end());
        result.mean_abs_error_ms = sum / static_cast<double>(absolute.size());
        result.p95_abs_error_ms = absolute[std::min(absolute.size() - 1, static_cast<size_t>(0.95 * static_cast<double>(absolute.size())))];
        result.max_abs_error_ms = absolute.back();
    }
    return result;
}

} // namespace trdrop
