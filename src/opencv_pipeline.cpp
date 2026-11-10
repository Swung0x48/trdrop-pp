#include "frame_pipeline.h"

#include <algorithm>
#include <chrono>
#include <deque>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "video_sources.h"

namespace trdrop {

double FramePipeline::progress() const
{
    double result = -1.0;
    for (size_t i = 0; i < count(); ++i)
    {
        const long long total = reported_frame_count(i);
        if (total <= 0) continue;
        result = std::max(result, static_cast<double>(frames_read()) / static_cast<double>(total));
    }
    return std::min(result, 1.0);
}

void FramePipeline::enable_motion()
{
    throw std::runtime_error("the motion analysis is not available with " + description());
}

namespace {

using Clock = std::chrono::steady_clock;

class OpenCvPipeline final : public FramePipeline {
public:
    OpenCvPipeline(const std::vector<std::string> & paths, std::unique_ptr<DiffEngine> engine)
        : _sources(paths), _engine(std::move(engine))
    { }

    std::string description() const override { return "opencv decode, " + _engine->description(); }

    bool submit_next(const std::vector<VideoOptions> & video_options) override
    {
        std::vector<cv::Mat> & frames = _engine->next_frames();
        if (!_sources.read_next_frames(frames)) return false;
        const Clock::time_point start = Clock::now();
        if (!_motion.empty())
        {
            std::vector<MotionField> fields(frames.size());
            for (size_t i = 0; i < frames.size(); ++i)
            {
                cv::cvtColor(frames[i], _gray, cv::COLOR_BGR2GRAY);
                fields[i] = _motion[i].estimate(_gray);
            }
            _motion_queue.push_back(std::move(fields));
        }
        _engine->submit(video_options);
        _analysis += Clock::now() - start;
        return true;
    }

    size_t pending() const override { return _engine->pending(); }
    size_t max_pending() const override { return _engine->max_pending(); }

    void collect(std::vector<RowCounts> & out) override
    {
        const Clock::time_point start = Clock::now();
        _engine->collect(out);
        if (!_motion.empty())
        {
            for (size_t i = 0; i < out.size(); ++i) out[i].motion = _motion_queue.front()[i];
            _motion_queue.pop_front();
        }
        _analysis += Clock::now() - start;
    }

    void enable_motion() override { _motion.resize(_sources.count()); }

    double analysis_seconds() const override { return std::chrono::duration<double>(_analysis).count(); }

    size_t count() const override { return _sources.count(); }
    double recorded_framerate(size_t index) const override { return _sources.recorded_framerate(index); }
    long long reported_frame_count(size_t index) const override { return _sources.reported_frame_count(index); }
    int frame_width(size_t index) const override { return _sources.frame_width(index); }
    int frame_height(size_t index) const override { return _sources.frame_height(index); }
    long long frames_read() const override { return _sources.frames_read(); }
    std::string codec_name(size_t index) const override { return _sources.codec_name(index); }
    bool lossy_codec(size_t index) const override { return _sources.lossy_codec(index); }

private:
    VideoSources _sources;
    std::unique_ptr<DiffEngine> _engine;
    //! motion estimation on the CPU, the fields wait for collect() in submission order
    std::vector<MotionEstimator> _motion;
    std::deque<std::vector<MotionField>> _motion_queue;
    cv::Mat _gray;
    Clock::duration _analysis {};
};

} // namespace

std::unique_ptr<FramePipeline> make_opencv_pipeline(const std::vector<std::string> & paths, std::unique_ptr<DiffEngine> engine)
{
    return std::make_unique<OpenCvPipeline>(paths, std::move(engine));
}

} // namespace trdrop
