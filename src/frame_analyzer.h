#pragma once

#include <vector>

#include <opencv2/core.hpp>

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

//! Holds two consecutive frames per video, calculates the difference frame and applies
//! the framerate, tear and frametime detection algorithms (FrameProcessing in the original)
class FrameAnalyzer {
public:
    //! the framerate is counted within a window of `recorded framerate` frames,
    //! as we can't detect a higher framerate than the one the video was recorded with
    explicit FrameAnalyzer(const std::vector<int> & recorded_framerates);

    //! compares the frames to the previous ones and updates framerate / frametime / tears.
    //! the first call only caches the frames
    void process(const std::vector<cv::Mat> & frames, const std::vector<VideoOptions> & video_options);

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
    //! the frames of t-1 as grayscale
    std::vector<cv::Mat> _cached_gray;
    std::vector<Tear> _tears;
    std::vector<double> _last_differences;
    bool _received_first_frames = false;
    bool _has_difference = false;
};

} // namespace trdrop
