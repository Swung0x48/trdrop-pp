#pragma once

#include <deque>
#include <vector>

#include "diff_engine.h"
#include "options.h"

namespace trdrop {

//! a single detected tear (TearData in the original)
struct Tear {
    //! the two neighbouring rows which define the tear (index_a + 1 == index_b)
    size_t index_a = 0;
    size_t index_b = 0;
    size_t video_index = 0;
    //! height of the analyzed video
    int original_video_height = 1;
};

//! Applies the framerate, tear and frametime detection algorithms to the per row differences
//! of two consecutive frames (FrameProcessing in the original). The per pixel part is done by a DiffEngine
class FrameAnalyzer {
public:
    //! the framerate is counted within a window of `recorded framerate` frames,
    //! as we can't detect a higher framerate than the one the video was recorded with
    explicit FrameAnalyzer(const std::vector<int> & recorded_framerates);

    //! updates framerate / frametime / tears with the row differences of the next set of frames.
    //! the first call only marks the first frames as received
    void process(const std::vector<RowCounts> & row_counts, const std::vector<VideoOptions> & video_options);

    //! true if the last call to process() compared frames
    bool has_difference() const { return _has_difference; }
    //! current framerate per video slot (size kMaxVideoCount)
    std::vector<double> framerates() const;
    //! current frametime in ms per video slot (size kMaxVideoCount)
    std::vector<double> frametimes() const;
    //! tears found in the last call to process()
    const std::vector<Tear> & tears() const { return _tears; }
    //! the raw difference value of the last comparison per video (0 = duplicate, 1 = new frame, (0,1) = tear)
    const std::vector<double> & last_differences() const { return _last_differences; }
    //! share (0 - 1) of the pixels which were different in the last comparison per video
    const std::vector<double> & last_changed_shares() const { return _last_changed_shares; }

private:
    double _calculate_framerate(size_t video_index) const;
    double _calculate_frametime(size_t video_index) const;
    double _frame_difference(const std::vector<double> & row_differences
                           , double dismiss_tear_percentage
                           , size_t video_index
                           , int video_height
                           , std::vector<Tear> & tears_out) const;

private:
    //! per video: ring buffer of frame differences with the size of the recorded framerate
    std::vector<std::vector<double>> _frame_diff_lists;
    //! per video: current write position in _frame_diff_lists
    std::vector<size_t> _current_framecount;
    std::vector<double> _row_differences;
    std::vector<Tear> _tears;
    std::vector<double> _last_differences;
    std::vector<double> _last_changed_shares;
    //! per video: changed shares of the last 2 seconds, for the automatic tolerance
    std::vector<std::deque<double>> _share_history;
    //! per video: changed shares of the last duplicates (2 seconds of them, no keyframes), the noise of the recording
    std::vector<std::deque<double>> _noise_history;
    std::vector<double> _sorted_shares;
    bool _received_first_frames = false;
    bool _has_difference = false;
};

} // namespace trdrop
