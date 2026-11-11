#include "motion.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace trdrop {

namespace {

//! the motion is measured at up to this resolution
constexpr double kMaxLongSide = 1920.0;
constexpr double kMaxShortSide = 1080.0;
//! reference motion of a cell: at least this speed (pixels per frame at the measured resolution) ...
constexpr double kMinSpeed = 0.5;
//! ... and tracked in this share of the new frames of the window
constexpr double kMinTrackedShare = 0.3;
//! a frame needs this many cells with motion (2% of the picture)
constexpr int kMinCells = kMotionCells / 50;
//! a frame whose animation step is smaller didn't advance the animation
constexpr double kMinStep = 0.3;
//! ... it is a duplicate if less than this share of its pixels changed, otherwise the scene changed without moving
constexpr double kDuplicateShare = 0.02;
//! automatic reference window: this long, growing up to kMaxWindowSeconds until it holds kMinWindowFrames new frames
constexpr double kDefaultWindowMs = 400.0;
constexpr double kMaxWindowSeconds = 1.0;
constexpr size_t kMinWindowFrames = 10;
//! the perceived smoothness needs this many steps and this share of measurable frames in the last second
constexpr size_t kMinRollingSteps = 6;
constexpr double kMinRollingMeasurable = 0.5;
//! asynchronous capture, a picture was lost before a new frame: the previous picture was recorded once, the
//! step is about twice as long as the usual step (the median of the last kLostUsualSteps) ...
constexpr double kLostMinRatio = 1.6;
constexpr double kLostMaxRatio = 2.4;
constexpr size_t kLostUsualSteps = 9;
//! ... and the pictures are shown for about one interval or less: a picture that is shown longer is always
//! recorded, and one that is shown for twice as long is recorded twice
constexpr double kLostMaxStep = 1.2;
//! a picture recorded twice before such a step counts as repeated by the capture (and the next one as lost)
//! if the capture lost at least this many pictures in the last second in the way described above
constexpr size_t kMinLostEvidence = 3;

//! Framerate of evenly paced motion that looks as smooth as the steps: the new frames per second divided by
//! 1 + (coefficient of variation of the steps)^2 = fps * sum(step)^2 / (displayed intervals * sum(step^2)).
//! Evenly paced steps give the framerate of the new frames; it doesn't depend on the scale of the steps.
double perceived_smoothness(double fps, double step_sum, double step_square_sum, double displayed_intervals)
{
    return fps * step_sum * step_sum / (displayed_intervals * step_square_sum);
}

float median(std::vector<float> & values)
{
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

//! weighted median of (value, weight) pairs with the sum of weights `total`: the first value, in ascending
//! order, at which the cumulated weight reaches half of the total. Reorders the pairs.
double weighted_median(std::pair<double, double> * begin, std::pair<double, double> * end, double total)
{
    if (begin == end) return 0.0;
    const double half = 0.5 * total;
    double below = 0.0; // weight of the pairs before begin
    while (end - begin > 16)
    {
        std::pair<double, double> * middle = begin + (end - begin) / 2;
        std::nth_element(begin, middle, end);
        double left = 0.0;
        for (const std::pair<double, double> * pair = begin; pair < middle; ++pair) left += pair->second;
        if (below + left >= half)
        {
            end = middle;
        }
        else if (below + left + middle->second >= half)
        {
            return middle->first;
        }
        else
        {
            below += left + middle->second;
            begin = middle + 1;
        }
    }
    std::sort(begin, end);
    for (const std::pair<double, double> * pair = begin; pair < end; ++pair)
    {
        below += pair->second;
        if (below >= half) return pair->first;
    }
    return (end - 1)->first;
}

} // namespace

int motion_cell(const MotionGrid & grid, float x, float y)
{
    const int cell_x = std::min(grid.cells_x - 1, std::max(0, static_cast<int>(x * static_cast<float>(grid.cells_x))));
    const int cell_y = std::min(grid.cells_y - 1, std::max(0, static_cast<int>(y * static_cast<float>(grid.cells_y))));
    return cell_y * grid.cells_x + cell_x;
}

int MotionGridCounts::moving_cells() const
{
    int cells = 0;
    for (const uint8_t points : moving)
    {
        if (points >= kMotionMinCellPoints) ++cells;
    }
    return cells;
}

bool MotionGridCounts::sparse() const
{
    const int cells = moving_cells();
    return cells > 0 && cells < kMotionSparseCells;
}

void MotionCellBinner::add(float x, float y, float dx, float dy)
{
    _points.push_back({ static_cast<uint16_t>(motion_cell(_grid, x, y)), dx, dy });
}

void MotionCellBinner::finish(MotionField & field, size_t min_points, float scale, bool agree)
{
    std::sort(_points.begin(), _points.end(), [](const Point & a, const Point & b) { return a.cell < b.cell; });
    // median motion of the points [first, last)
    auto median_motion = [this](size_t first, size_t last, float & dx, float & dy) {
        _values.clear();
        for (size_t k = first; k < last; ++k) _values.push_back(_points[k].dx);
        dx = median(_values);
        _values.clear();
        for (size_t k = first; k < last; ++k) _values.push_back(_points[k].dy);
        dy = median(_values);
    };
    for (size_t begin = 0; begin < _points.size();)
    {
        size_t end = begin;
        while (end < _points.size() && _points[end].cell == _points[begin].cell) ++end;
        const size_t cell = _points[begin].cell;
        const size_t all = end - begin;
        size_t count = all;
        float dx = 0.0f, dy = 0.0f;
        if (count >= min_points) median_motion(begin, end, dx, dy);
        if (agree && count >= min_points)
        {
            // the points that moved like the median come first
            const float tolerance = std::max(kMotionAgreePixels, kMotionAgreeShare * std::sqrt(dx * dx + dy * dy));
            const auto others = std::partition(_points.begin() + static_cast<std::ptrdiff_t>(begin), _points.begin() + static_cast<std::ptrdiff_t>(end),
                                               [&](const Point & point) {
                const float ex = point.dx - dx, ey = point.dy - dy;
                return ex * ex + ey * ey <= tolerance * tolerance;
            });
            count = static_cast<size_t>(others - (_points.begin() + static_cast<std::ptrdiff_t>(begin)));
            if (2 * count <= all) count = 0;
            else if (count >= min_points) median_motion(begin, begin + count, dx, dy);
        }
        if (count >= min_points)
        {
            field.points[cell] = static_cast<uint8_t>(std::min<size_t>(count, 255));
            field.dx[cell] = dx * scale;
            field.dy[cell] = dy * scale;
        }
        begin = end;
    }
}

int motion_scale(int width, int height)
{
    // the limit lies as the frame does: a recording of a phone's screen is not a very high wide picture
    const double long_side = std::max(width, height);
    const double short_side = std::min(width, height);
    return std::max(1, static_cast<int>(std::ceil(std::max(long_side / kMaxLongSide, short_side / kMaxShortSide) - 1e-9)));
}

MotionGrid motion_grid(int width, int height)
{
    MotionGrid grid;
    if (height > width) std::swap(grid.cells_x, grid.cells_y);
    return grid;
}

cv::Size motion_picture_size(int width, int height)
{
    const int scale = motion_scale(width, height);
    return cv::Size(width / scale / 2 * 2, height / scale / 2 * 2);
}

MotionField MotionEstimator::estimate(const cv::Mat & luma, std::vector<MotionPoint> * points)
{
    MotionField field;
    if (points) points->clear();
    const int scale = motion_scale(luma.cols, luma.rows);
    field.pixel_scale = static_cast<float>(scale);
    if (scale > 1) cv::resize(luma, _current, cv::Size(luma.cols / scale, luma.rows / scale), 0, 0, cv::INTER_AREA);
    else luma.copyTo(_current);
    // the pyramid (with derivatives: image and derivative per level) is reused as the previous one of the next frame
    const cv::Size window(21, 21);
    const int levels = cv::buildOpticalFlowPyramid(_current, _current_pyramid, window, 3, true);

    if (!_previous_pyramid.empty() && _previous_size == _current.size())
    {
        field.estimated = true;
        const MotionGrid grid = motion_grid(_current.cols, _current.rows);
        if (_points.empty() || _points_size != _current.size())
        {
            _points_size = _current.size();
            _points.clear();
            for (int y = 0; y < grid.points_y(); ++y)
            {
                for (int x = 0; x < grid.points_x(); ++x)
                {
                    _points.emplace_back((static_cast<float>(x) + 0.5f) * static_cast<float>(_current.cols) / static_cast<float>(grid.points_x()) - 0.5f,
                                         (static_cast<float>(y) + 0.5f) * static_cast<float>(_current.rows) / static_cast<float>(grid.points_y()) - 0.5f);
                }
            }
        }
        const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.01);
        // OpenCV's eigenvalue is the one of the gradient matrix per pixel / 1024 (derivatives * 32, sums / 2^20)
        const double min_eigenvalue = kMotionMinTexture / 1024.0;
        _binner.clear(grid);
        if (points) points->clear();
        const float width = static_cast<float>(_current.cols), height = static_cast<float>(_current.rows);
        // tracks the points there and back - a correct match returns to its point - and adds those to their cells
        auto track = [&](const std::vector<cv::Point2f> & from, MotionGridCounts * counts) {
            if (from.empty()) return;
            cv::calcOpticalFlowPyrLK(_previous_pyramid, _current_pyramid, from, _tracked, _status, _error, window, levels, criteria, 0, min_eigenvalue);
            cv::calcOpticalFlowPyrLK(_current_pyramid, _previous_pyramid, _tracked, _back, _back_status, _error, window, levels, criteria, 0, min_eigenvalue);
            for (size_t p = 0; p < from.size(); ++p)
            {
                const cv::Point2f back = _back[p] - from[p];
                const cv::Point2f motion = _tracked[p] - from[p];
                const bool returns = _status[p] && _back_status[p]
                                  && back.x * back.x + back.y * back.y <= kMotionMaxBackError * kMotionMaxBackError;
                if (points)
                {
                    points->push_back({ from[p].x, from[p].y, motion.x, motion.y,
                                        returns ? MotionPoint::Used : _status[p] ? MotionPoint::Rejected : MotionPoint::NoTexture });
                }
                if (!returns) continue;
                const float x = (from[p].x + 0.5f) / width, y = (from[p].y + 0.5f) / height;
                _binner.add(x, y, motion.x, motion.y);
                if (counts) counts->add(motion_cell(grid, x, y), motion.x, motion.y);
            }
        };
        MotionGridCounts counts;
        track(_points, &counts);
        // a frame without motion is treated like the last one with motion
        if (counts.moved())
        {
            _moved_counts = counts;
            _sparse = counts.sparse();
        }
        const bool added = _sparse && levels >= 1;
        if (added)
        {
            _texture_points(grid, _moved_counts);
            track(_extra, nullptr);
        }
        _binner.finish(field, kMotionMinCellPoints, field.pixel_scale, added);
    }
    else
    {
        _sparse = false;
    }
    _previous_size = _current.size();
    std::swap(_previous_pyramid, _current_pyramid);
    return field;
}

void MotionEstimator::_texture_points(const MotionGrid & grid, const MotionGridCounts & counts)
{
    // the previous frame at half the resolution: level 1 of its pyramid (with its border, so a view of it)
    const cv::Mat & image = _previous_pyramid[2];
    // OpenCV sums the squares of the Sobel responses, scaled by 1 / (4 * block * 255), over the block
    cv::cornerMinEigenVal(image, _texture, kMotionTextureBlock, 3);
    const double sobel_scale = 8.0 / (4.0 * kMotionTextureBlock * 255.0);
    const float minimum = static_cast<float>(kMotionTextureMin * kMotionTextureBlock * kMotionTextureBlock * sobel_scale * sobel_scale);

    const int parts_x = grid.cells_x * kMotionTextureParts, parts_y = grid.cells_y * kMotionTextureParts;
    const size_t parts = static_cast<size_t>(parts_x) * static_cast<size_t>(parts_y);
    _part_texture.assign(parts, 0.0f);
    _part_points.resize(parts);
    for (int y = 0; y < image.rows; ++y)
    {
        const float * texture = _texture.ptr<float>(y);
        // the position at the measured resolution: a pixel of level 1 covers two of them
        // (the pixel belongs to the part floor((2 p + 1) * parts / size), as in motion_texture.comp)
        const int part_y = std::min(parts_y - 1, (2 * y + 1) * parts_y / _current.rows);
        for (int x = 0; x < image.cols; ++x)
        {
            if (texture[x] < minimum) continue;
            const int part_x = std::min(parts_x - 1, (2 * x + 1) * parts_x / _current.cols);
            const size_t part = static_cast<size_t>(part_y) * static_cast<size_t>(parts_x) + static_cast<size_t>(part_x);
            if (texture[x] <= _part_texture[part]) continue;
            _part_texture[part] = texture[x];
            _part_points[part] = cv::Point2f(2.0f * static_cast<float>(x) + 0.5f, 2.0f * static_cast<float>(y) + 0.5f);
        }
    }
    _extra.clear();
    for (size_t part = 0; part < parts; ++part)
    {
        if (_part_texture[part] <= 0.0f) continue;
        const int cell = static_cast<int>(part / static_cast<size_t>(parts_x)) / kMotionTextureParts * grid.cells_x
                       + static_cast<int>(part % static_cast<size_t>(parts_x)) / kMotionTextureParts;
        if (counts.is_static(cell)) continue;
        _extra.push_back(_part_points[part]);
    }
}

MotionAnalyzer::MotionAnalyzer(double fps, double window_ms, bool async_capture)
    : _fps(fps)
    , _half_window(static_cast<size_t>(std::max(2.0, std::round((window_ms > 0.0 ? window_ms : kDefaultWindowMs) / 2000.0 * fps))))
    // a given window is used as it is, the automatic one may grow
    , _max_half_window(window_ms > 0.0 ? _half_window
                                       : std::max(_half_window, static_cast<size_t>(std::round(kMaxWindowSeconds / 2.0 * fps))))
    , _rolling_window(static_cast<size_t>(std::max(1.0, std::ceil(fps - 1e-9))))
    , _async_capture(async_capture)
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
    _band_low = -1.0;
    _band_high = 0.0;
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
    // The new frames of the window. A new frame stands for the intervals since the previous new frame (its
    // duplicates): its weight is the sum of their weights.
    _groups.clear();
    double window = 0.0;
    {
        double pending = 0.0;
        size_t intervals = 0;
        for (size_t k = first; k <= last; ++k)
        {
            if (!_entries[k].field.estimated)
            {
                pending = 0.0;
                intervals = 0;
                continue;
            }
            pending += weight(k);
            ++intervals;
            if (_entries[k].duplicate) continue;
            _groups.push_back({ &_entries[k].field, pending, pending / static_cast<double>(intervals) });
            window += pending;
            pending = 0.0;
            intervals = 0;
        }
    }

    // Reference motion of a cell: its motion per display interval, v = sum(motion) / sum(display time) over the
    // new frames of the window in which the cell was tracked. If the cell wasn't tracked in a new frame - which
    // mostly happens with fast motion - its motion and its time are left out; counting it as "no motion" would lower v.
    size_t reference_cells = 0;
    if (entry.field.estimated && window > 0.0)
    {
        const double min_speed = kMinSpeed * entry.field.pixel_scale;
        for (size_t cell = 0; cell < static_cast<size_t>(kMotionCells); ++cell)
        {
            double vx = 0.0, vy = 0.0, tracked = 0.0;
            for (const Group & group : _groups)
            {
                if (group.field->points[cell] == 0) continue;
                vx += group.scale * group.field->dx[cell];
                vy += group.scale * group.field->dy[cell];
                tracked += group.weight;
            }
            _reference[cell].usable = false;
            if (tracked <= kMinTrackedShare * window) continue;
            vx /= tracked;
            vy /= tracked;
            const double speed_squared = vx * vx + vy * vy;
            if (speed_squared < min_speed * min_speed) continue;
            _reference[cell] = { true, vx / speed_squared, vy / speed_squared, std::sqrt(speed_squared) };
            ++reference_cells;
        }
    }
    // Step of a motion field: every cell with motion votes with m.v / |v|^2, the weighted median wins.
    std::array<std::pair<double, double>, kMotionCells> votes; // (step, weight)
    size_t voters = 0;
    double weight_sum = 0.0;
    auto step_of = [&](const MotionField & field) {
        voters = 0;
        weight_sum = 0.0;
        for (size_t cell = 0; cell < static_cast<size_t>(kMotionCells); ++cell)
        {
            const Reference & v = _reference[cell];
            if (!v.usable || field.points[cell] == 0) continue;
            votes[voters++] = { field.dx[cell] * v.x + field.dy[cell] * v.y, v.speed };
            weight_sum += v.speed;
        }
        return voters >= static_cast<size_t>(kMinCells) ? weighted_median(votes.data(), votes.data() + voters, weight_sum) : 0.0;
    };
    if (reference_cells >= static_cast<size_t>(kMinCells))
    {
        // The steps are scaled so that the animation advances by one interval per displayed interval on average over
        // the window. The reference motion alone doesn't ensure that: a cell is tracked more often in the frames with
        // small steps, and the motion of a frame is part of its own reference.
        double measured = 0.0, displayed = 0.0;
        for (const Group & group : _groups)
        {
            const double step = step_of(*group.field);
            if (voters < static_cast<size_t>(kMinCells)) continue;
            measured += group.scale * std::max(0.0, step);
            displayed += group.weight;
        }
        const double scale = measured > 0.0 ? displayed / measured : 1.0;
        const double step = step_of(entry.field);
        if (voters >= static_cast<size_t>(kMinCells))
        {
            sample.measurable = true;
            sample.step = scale * step;
            // the cells never agree exactly: scaled median absolute deviation of their votes
            for (size_t k = 0; k < voters; ++k) votes[k].first = std::fabs(votes[k].first - step);
            sample.step_spread = scale * 1.4826 * weighted_median(votes.data(), votes.data() + voters, weight_sum);
        }
    }

    if (entry.field.estimated && !entry.duplicate) ++_new_frames;

    // animation chain: a new frame ends the step which started at the previous new frame
    if (!sample.measurable)
    {
        _reset_chain();
    }
    else
    {
        ++_measurable_frames;
        if (!entry.duplicate) ++_measurable_new_frames;
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
                // how far the animation advanced beyond the time the previous picture was recorded, in intervals
                double error = _accumulated_step - static_cast<double>(_shown);
                if (_async_capture)
                {
                    // The drift moved by `error`. Within the band of one interval that is the capture, beyond it
                    // the band is pushed along and the rest is the animation error.
                    const double moved = error;
                    if (moved > _band_high + 1.0)
                    {
                        error = moved - (_band_high + 1.0);
                        _band_low = _band_high = moved - 1.0;
                    }
                    else if (moved < _band_low)
                    {
                        error = moved - _band_low;
                        _band_low = _band_high = moved;
                    }
                    else
                    {
                        error = 0.0;
                        _band_low = std::max(_band_low, moved - 1.0);
                        _band_high = std::min(_band_high, moved);
                    }
                    // the drift of this frame is the new origin
                    _band_low -= moved;
                    _band_high -= moved;
                    const double retimed = moved - error;
                    if (std::fabs(retimed) > 0.5) ++_retimed_frames;
                    // A lost picture shows as a step twice as long as the steps around it. If the previous picture
                    // was recorded an interval short of that step, a picture is missing. If it was recorded twice,
                    // either the capture repeated it and lost the next one, or the game showed it twice: that
                    // counts as the capture while it demonstrably loses pictures.
                    while (!_lost_evidence.empty() && _lost_evidence.front() + static_cast<long long>(_rolling_window) <= _finalized)
                    {
                        _lost_evidence.pop_front();
                    }
                    const bool missing = _shown == 1;
                    const bool repeated = _shown == 2 && _lost_evidence.size() >= kMinLostEvidence;
                    if ((missing || repeated) && _rolling_steps.size() >= kMinRollingSteps)
                    {
                        // the usual step: of the last few pictures, the scale of the steps drifts if the motion
                        // speeds up or slows down
                        _recent_steps.clear();
                        const size_t count = std::min(_rolling_steps.size(), kLostUsualSteps);
                        for (size_t k = _rolling_steps.size() - count; k < _rolling_steps.size(); ++k)
                        {
                            _recent_steps.push_back(static_cast<float>(_rolling_steps[k].step));
                        }
                        const double usual = median(_recent_steps);
                        if (usual <= kLostMaxStep && _accumulated_step >= kLostMinRatio * usual && _accumulated_step <= kLostMaxRatio * usual)
                        {
                            sample.lost = 1;
                            if (missing) _lost_evidence.push_back(_finalized);
                            else ++_lost_after_repeat;
                        }
                    }
                }
                sample.has_error = true;
                sample.error_ms = error * 1000.0 / _fps;
                _errors.push_back(static_cast<float>(sample.error_ms));
                _step_spread.push_back(static_cast<float>(sample.step_spread));
                _shown_total += _shown;
                // a lost picture and the one after it share the step
                const int pictures = 1 + sample.lost;
                const double step = _accumulated_step / pictures;
                _lost_frames += sample.lost;
                _animation_frames += pictures;
                _step_sum += _accumulated_step;
                _step_square_sum += pictures * step * step;
                for (int k = 0; k < pictures; ++k)
                {
                    _rolling_steps.push_back({ _finalized, step, static_cast<double>(_shown) / pictures });
                }
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
    while (!_rolling_steps.empty() && _rolling_steps.front().frame + static_cast<long long>(_rolling_window) <= _finalized)
    {
        _rolling_steps.pop_front();
    }
    if (_rolling_steps.size() >= kMinRollingSteps
        && static_cast<double>(_rolling_measurable_count) > kMinRollingMeasurable * static_cast<double>(_rolling_window))
    {
        double sum = 0.0, square_sum = 0.0, displayed = 0.0;
        for (const RollingStep & step : _rolling_steps)
        {
            sum += step.step;
            square_sum += step.step * step.step;
            displayed += step.displayed;
        }
        if (square_sum > 0.0 && displayed > 0.0)
        {
            sample.has_perceived_fps = true;
            sample.perceived_fps = perceived_smoothness(_fps, sum, square_sum, displayed);
            _rolling_values.push_back(static_cast<float>(sample.perceived_fps));
        }
    }
    return sample;
}

MotionAnalyzer::Summary MotionAnalyzer::summary() const
{
    Summary result;
    result.frames = _finalized;
    result.measurable_frames = _measurable_frames;
    result.new_frames = _new_frames;
    result.measurable_new_frames = _measurable_new_frames;
    result.animation_frames = _animation_frames;
    result.retimed_frames = _retimed_frames;
    result.lost_frames = _lost_frames;
    result.lost_after_repeat = _lost_after_repeat;
    if (_shown_total > 0) result.motion_fps = _fps * static_cast<double>(_animation_frames) / static_cast<double>(_shown_total);
    if (_step_square_sum > 0.0 && _shown_total > 0)
    {
        result.perceived_fps = perceived_smoothness(_fps, _step_sum, _step_square_sum, static_cast<double>(_shown_total));
    }
    if (!_step_spread.empty())
    {
        std::vector<float> sorted = _step_spread;
        result.median_step_spread_ms = median(sorted) * 1000.0 / _fps;
    }
    if (!_rolling_values.empty())
    {
        std::vector<float> sorted = _rolling_values;
        std::sort(sorted.begin(), sorted.end());
        result.perceived_fps_median = sorted[sorted.size() / 2];
        result.perceived_fps_low = sorted[sorted.size() / 10];
    }
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
        const double interval = 1000.0 / _fps;
        long long too_short = 0, too_long = 0;
        for (float error : _errors)
        {
            if (std::fabs(std::fabs(error) - interval) > 0.25 * interval) continue;
            ++(error < 0.0f ? too_short : too_long);
        }
        result.interval_errors_short = static_cast<double>(too_short) / static_cast<double>(_errors.size());
        result.interval_errors_long = static_cast<double>(too_long) / static_cast<double>(_errors.size());
        result.mean_abs_error_ms = sum / static_cast<double>(absolute.size());
        result.median_abs_error_ms = absolute[absolute.size() / 2];
        result.p95_abs_error_ms = absolute[std::min(absolute.size() - 1, static_cast<size_t>(0.95 * static_cast<double>(absolute.size())))];
        result.max_abs_error_ms = absolute.back();
    }
    return result;
}

} // namespace trdrop
