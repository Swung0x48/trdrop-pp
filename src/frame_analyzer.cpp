#include "frame_analyzer.h"

#include <algorithm>
#include <numeric>

namespace trdrop {

namespace {

//! how much of a row has to be different to see it as a "tear cut"
constexpr double kTearRowCompleteness = 0.1;

size_t decrement_modulo(size_t value, size_t max_value)
{
    return value == 0 ? max_value : value - 1;
}

//! checks if the two normalized row differences count as a tear
bool are_tear_rows(double row_diff_a, double row_diff_b)
{
    // old frame in row a (duplicate row), new frame in row b
    if (row_diff_a == 0.0 && row_diff_b >= kTearRowCompleteness) return true;
    // new frame in row a, old frame in row b (duplicate row)
    if (row_diff_a >= kTearRowCompleteness && row_diff_b == 0.0) return true;
    return false;
}

//! returns the percentage of the biggest tear found by comparing the height
double max_tear_percentage(const std::vector<double> & row_differences, const std::vector<Tear> & tears)
{
    double max_percentage = 0.0;
    const size_t rows = row_differences.size();
    for (const Tear & tear : tears)
    {
        if (row_differences[tear.index_a] == 0.0)
        {
            // walk in the direction of index_b + 1
            size_t index = tear.index_b;
            size_t counter = 0;
            while (row_differences[index] > 0.0)
            {
                counter += 1;
                if (index == rows - 1) break;
                index += 1;
            }
            max_percentage = std::max(max_percentage, static_cast<double>(counter) / static_cast<double>(rows));
        }
        if (row_differences[tear.index_b] == 0.0)
        {
            // walk in the direction of index_a - 1
            size_t index = tear.index_a;
            size_t counter = 0;
            while (row_differences[index] > 0.0)
            {
                counter += 1;
                if (index == 0) break;
                index -= 1;
            }
            max_percentage = std::max(max_percentage, static_cast<double>(counter) / static_cast<double>(rows));
        }
    }
    return max_percentage;
}

} // namespace

FrameAnalyzer::FrameAnalyzer(const std::vector<int> & recorded_framerates)
{
    for (int i = 0; i < kMaxVideoCount; ++i)
    {
        const int recorded_framerate = i < static_cast<int>(recorded_framerates.size()) ? recorded_framerates[i] : 0;
        _frame_diff_lists.emplace_back(static_cast<size_t>(std::max(0, recorded_framerate)), 0.0);
        _current_framecount.push_back(0);
    }
    _last_differences.assign(kMaxVideoCount, 0.0);
    _last_changed_shares.assign(kMaxVideoCount, 0.0);
    _share_history.resize(kMaxVideoCount);
}

void FrameAnalyzer::process(const std::vector<RowCounts> & row_counts, const std::vector<VideoOptions> & video_options)
{
    const size_t video_count = row_counts.size();
    _tears.clear();
    _has_difference = false;

    if (!_received_first_frames)
    {
        // we can only calculate the difference if we have at least two sets of frames
        _received_first_frames = true;
        return;
    }

    std::vector<std::vector<Tear>> tears_per_video(video_count);

    for (size_t i = 0; i < video_count; ++i)
    {
        const RowCounts & counts = row_counts[i];
        // a resolution change mid-stream can't be compared row by row, count it as a new frame
        if (!counts.compared)
        {
            _frame_diff_lists[i][_current_framecount[i]] = 1.0;
            _last_differences[i] = 1.0;
            _last_changed_shares[i] = 1.0;
            continue;
        }
        // normalized amount of different pixels per row (0.0 - 1.0),
        // same arithmetic as summing the 255 / 0 threshold mask of the original
        const int rows = static_cast<int>(counts.counts.size());
        _row_differences.assign(counts.counts.size(), 0.0);
        const double normalization = 255.0 * static_cast<double>(counts.cols);
        long long changed_pixels = 0;
        for (size_t row = 0; row < counts.counts.size(); ++row)
        {
            _row_differences[row] = static_cast<double>(static_cast<long long>(counts.counts[row]) * 255) / normalization;
            changed_pixels += counts.counts[row];
        }
        const double pixels = static_cast<double>(counts.cols) * static_cast<double>(std::max(1, rows));
        _last_changed_shares[i] = pixels > 0.0 ? static_cast<double>(changed_pixels) / pixels : 0.0;

        // lossy codecs: compression noise changes a few pixels of identical source frames. The frame
        // (and for the tear detection every row) only counts as changed above the tolerance
        const double row_tolerance = std::max(0.0, video_options[i].tolerance_pct) / 100.0;
        double tolerance = row_tolerance;
        if (video_options[i].tolerance_auto)
        {
            // relative to what a new frame changed recently: 90th percentile of the last 2 seconds
            std::deque<double> & history = _share_history[i];
            history.push_back(_last_changed_shares[i]);
            if (history.size() > std::max<size_t>(2, 2 * _frame_diff_lists[i].size())) history.pop_front();
            _sorted_shares.assign(history.begin(), history.end());
            const auto reference = _sorted_shares.begin() + static_cast<std::ptrdiff_t>(0.9 * static_cast<double>(_sorted_shares.size() - 1));
            std::nth_element(_sorted_shares.begin(), reference, _sorted_shares.end());
            tolerance = std::max(tolerance, kAutoToleranceRelative * *reference);
        }
        if (tolerance > 0.0)
        {
            const bool frame_changed = _last_changed_shares[i] > tolerance;
            for (double & row_difference : _row_differences)
            {
                if (!frame_changed || row_difference <= row_tolerance) row_difference = 0.0;
            }
        }

        const double dismiss_tear_percentage = video_options[i].dismiss_tear_percentage / 100.0;
        const double difference = _frame_difference(_row_differences, dismiss_tear_percentage, i, rows, tears_per_video[i]);
        _frame_diff_lists[i][_current_framecount[i]] = difference;
        _last_differences[i] = difference;
    }

    for (const auto & video_tears : tears_per_video)
    {
        _tears.insert(_tears.end(), video_tears.begin(), video_tears.end());
    }

    // increment the framecounter for each video and loop around the recorded framerate
    for (size_t i = 0; i < _current_framecount.size(); ++i)
    {
        const size_t recorded_framerate = _frame_diff_lists[i].size();
        _current_framecount[i] += 1;
        if (recorded_framerate != 0) _current_framecount[i] %= recorded_framerate;
    }

    _has_difference = true;
}

std::vector<double> FrameAnalyzer::framerates() const
{
    std::vector<double> result;
    for (size_t i = 0; i < _frame_diff_lists.size(); ++i) result.push_back(_calculate_framerate(i));
    return result;
}

std::vector<double> FrameAnalyzer::frametimes() const
{
    std::vector<double> result;
    for (size_t i = 0; i < _frame_diff_lists.size(); ++i) result.push_back(_calculate_frametime(i));
    return result;
}

double FrameAnalyzer::_calculate_framerate(size_t video_index) const
{
    // sum of the frame differences within the last second (0's, 1's and partial tears)
    const auto & diffs = _frame_diff_lists[video_index];
    return std::accumulate(diffs.begin(), diffs.end(), 0.0);
}

double FrameAnalyzer::_calculate_frametime(size_t video_index) const
{
    const auto & diffs = _frame_diff_lists[video_index];
    const size_t recorded_framerate = diffs.size();
    // if the recorded framerate is zero, there is no video loaded for that index
    if (recorded_framerate == 0) return 0.0;
    // we need at least two different frames in the window, otherwise the loop below would never terminate
    if (_calculate_framerate(video_index) < 2.0) return 0.0;

    // we always look at the previous frame first
    size_t position = decrement_modulo(_current_framecount[video_index], recorded_framerate - 1);
    // count how long the last completed frame was shown (in 1 / recorded_framerate seconds)
    double frame_time_counter = 0.0;
    bool found_last_diff = false;
    while (true)
    {
        const double diff = diffs[position];
        if (found_last_diff)
        {
            if (diff == 0.0) frame_time_counter += 1.0;
            if (diff > 0.0) { frame_time_counter += 1.0; break; }
        }
        else if (diff > 0.0)
        {
            found_last_diff = true;
        }
        position = decrement_modulo(position, recorded_framerate - 1);
    }
    const double frametime_in_s = frame_time_counter / static_cast<double>(recorded_framerate);
    return frametime_in_s * 1000.0;
}

//! returns 0 if the compared frames were identical
//! returns 1 if the frames were at least in 1 pixel different
//! if we detect a tear, we check how big it is (height) and see if it's above the dismiss_tear_percentage
//!     if it is, we return the percentage of the frame which was not a tear (1 - tear_percentage)
//!     if it is NOT, we return 1, i.e. we think of them as completely different
double FrameAnalyzer::_frame_difference(const std::vector<double> & row_differences
                                      , double dismiss_tear_percentage
                                      , size_t video_index
                                      , int video_height
                                      , std::vector<Tear> & tears_out) const
{
    bool found_different_pixel = false;
    for (size_t row = 0; row < row_differences.size(); ++row)
    {
        if (row_differences[row] > 0.0) { found_different_pixel = true; }
        if (row == 0) continue;
        if (are_tear_rows(row_differences[row - 1], row_differences[row]))
        {
            Tear tear;
            tear.index_a = row - 1;
            tear.index_b = row;
            tear.video_index = video_index;
            tear.original_video_height = video_height;
            tears_out.push_back(tear);
        }
    }

    if (!tears_out.empty())
    {
        const double max_percentage = max_tear_percentage(row_differences, tears_out);
        // if the tear is not big enough, we dismiss it
        if (max_percentage < dismiss_tear_percentage) return found_different_pixel ? 1.0 : 0.0;
        // if the tear is big enough, we return the remaining frame
        return 1.0 - max_percentage;
    }
    // no tear: a different pixel "rounds" to a full new frame, otherwise it's a duplicate frame
    return found_different_pixel ? 1.0 : 0.0;
}

} // namespace trdrop
