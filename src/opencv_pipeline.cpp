#include "frame_pipeline.h"

#include <algorithm>
#include <chrono>

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
        if (!_sources.read_next_frames(_engine->next_frames())) return false;
        const Clock::time_point start = Clock::now();
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
        _analysis += Clock::now() - start;
    }

    double analysis_seconds() const override { return std::chrono::duration<double>(_analysis).count(); }

    size_t count() const override { return _sources.count(); }
    double recorded_framerate(size_t index) const override { return _sources.recorded_framerate(index); }
    long long reported_frame_count(size_t index) const override { return _sources.reported_frame_count(index); }
    int frame_width(size_t index) const override { return _sources.frame_width(index); }
    int frame_height(size_t index) const override { return _sources.frame_height(index); }
    long long frames_read() const override { return _sources.frames_read(); }

private:
    VideoSources _sources;
    std::unique_ptr<DiffEngine> _engine;
    Clock::duration _analysis {};
};

} // namespace

std::unique_ptr<FramePipeline> make_opencv_pipeline(const std::vector<std::string> & paths, std::unique_ptr<DiffEngine> engine)
{
    return std::make_unique<OpenCvPipeline>(paths, std::move(engine));
}

} // namespace trdrop
