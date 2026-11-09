#include "diff_engine.h"

#include <utility>

#include <opencv2/imgproc.hpp>

namespace trdrop {

namespace {

class CpuDiffEngine final : public DiffEngine {
public:
    explicit CpuDiffEngine(size_t video_count)
        : _frames(video_count), _gray(video_count), _previous_gray(video_count), _result(video_count)
    { }

    std::string description() const override { return "cpu (OpenCV)"; }

    std::vector<cv::Mat> & next_frames() override { return _frames; }

    void submit(const std::vector<VideoOptions> & video_options) override
    {
        for (size_t i = 0; i < _frames.size(); ++i)
        {
            // single channel frames already are the luma plane (FFmpeg decoding)
            if (_frames[i].channels() == 1) _frames[i].copyTo(_gray[i]);
            else cv::cvtColor(_frames[i], _gray[i], cv::COLOR_BGR2GRAY);
            RowCounts & result = _result[i];
            result.compared = !_previous_gray[i].empty() && _previous_gray[i].size() == _gray[i].size();
            result.cols = _gray[i].cols;
            result.counts.clear();
            if (result.compared)
            {
                // a pixel is "different" if the grayscale difference is greater than the pixel difference option
                cv::absdiff(_previous_gray[i], _gray[i], _absolute_difference);
                cv::threshold(_absolute_difference, _mask, video_options[i].pixel_difference, 255, cv::THRESH_BINARY);
                cv::reduce(_mask, _row_sums, 1, cv::REDUCE_SUM, CV_32S);
                result.counts.resize(static_cast<size_t>(_mask.rows));
                for (int row = 0; row < _mask.rows; ++row)
                {
                    result.counts[static_cast<size_t>(row)] = static_cast<uint32_t>(_row_sums.at<int>(row, 0) / 255);
                }
            }
            std::swap(_gray[i], _previous_gray[i]);
        }
        _pending = 1;
    }

    size_t pending() const override { return _pending; }
    size_t max_pending() const override { return 1; }

    void collect(std::vector<RowCounts> & out) override
    {
        out = _result;
        _pending = 0;
    }

private:
    std::vector<cv::Mat> _frames;
    std::vector<cv::Mat> _gray;
    std::vector<cv::Mat> _previous_gray;
    std::vector<RowCounts> _result;
    cv::Mat _absolute_difference;
    cv::Mat _mask;
    cv::Mat _row_sums;
    size_t _pending = 0;
};

} // namespace

std::unique_ptr<DiffEngine> make_cpu_engine(size_t video_count)
{
    return std::make_unique<CpuDiffEngine>(video_count);
}

} // namespace trdrop
