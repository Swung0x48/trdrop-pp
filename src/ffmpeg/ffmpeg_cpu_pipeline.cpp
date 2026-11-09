// FFmpeg decoding on the CPU, the luma (Y) plane is compared with the OpenCV CPU engine.
// Same semantics as the Vulkan Video pipeline, used as its fallback / reference.

#include "frame_pipeline.h"

#include <chrono>
#include <stdexcept>

extern "C" {
#include <libavutil/pixdesc.h>
}

#include "ffmpeg/ffmpeg_decoder.h"

namespace trdrop {

namespace {

using Clock = std::chrono::steady_clock;

class FfmpegCpuPipeline final : public FramePipeline {
public:
    explicit FfmpegCpuPipeline(const std::vector<std::string> & paths)
        : _engine(make_cpu_engine(paths.size()))
    {
        try
        {
            for (const std::string & path : paths)
            {
                _decoders.push_back(std::make_unique<ffmpeg::VideoDecoder>(path, nullptr));
                _frames.push_back(av_frame_alloc());
                if (!_frames.back()) throw std::runtime_error("out of memory");
            }
            // decode the first frames to check the pixel format, they are the first submission
            for (size_t i = 0; i < _decoders.size(); ++i)
            {
                if (!_decoders[i]->next(_frames[i])) throw std::runtime_error("'" + paths[i] + "' contains no frames");
                _check_format(i);
            }
            _first_frames_ready = true;
        }
        catch (...)
        {
            _release();
            throw;
        }
    }

    ~FfmpegCpuPipeline() override { _release(); }

    std::string description() const override { return "ffmpeg cpu decode, " + _engine->description() + ", luma"; }

    bool submit_next(const std::vector<VideoOptions> & video_options) override
    {
        std::vector<cv::Mat> & targets = _engine->next_frames();
        for (size_t i = 0; i < _decoders.size(); ++i)
        {
            if (!_first_frames_ready)
            {
                if (!_decoders[i]->next(_frames[i])) return false;
                _check_format(i);
            }
            const AVFrame * frame = _frames[i];
            if (frame->linesize[0] <= 0) throw std::runtime_error("unsupported frame layout");
            targets[i] = cv::Mat(frame->height, frame->width, CV_8UC1, frame->data[0], static_cast<size_t>(frame->linesize[0]));
        }
        _first_frames_ready = false;
        const Clock::time_point start = Clock::now();
        _engine->submit(video_options); // copies the luma planes
        _analysis += Clock::now() - start;
        ++_frames_read;
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

    size_t count() const override { return _decoders.size(); }
    double recorded_framerate(size_t index) const override { return _decoders.at(index)->framerate(); }
    long long reported_frame_count(size_t index) const override { return _decoders.at(index)->frame_count(); }
    int frame_width(size_t index) const override { return _decoders.at(index)->width(); }
    int frame_height(size_t index) const override { return _decoders.at(index)->height(); }
    long long frames_read() const override { return _frames_read; }

private:
    void _check_format(size_t index) const
    {
        const AVPixelFormat format = static_cast<AVPixelFormat>(_frames[index]->format);
        if (!ffmpeg::is_luma_format(format))
        {
            const char * name = av_get_pix_fmt_name(format);
            throw std::runtime_error(std::string("pixel format ") + (name ? name : "unknown") + " has no 8-bit luma plane");
        }
    }

    void _release()
    {
        for (AVFrame *& frame : _frames) av_frame_free(&frame);
        _frames.clear();
        _decoders.clear();
    }

    std::unique_ptr<DiffEngine> _engine;
    std::vector<std::unique_ptr<ffmpeg::VideoDecoder>> _decoders;
    std::vector<AVFrame *> _frames;
    bool _first_frames_ready = false;
    long long _frames_read = 0;
    Clock::duration _analysis {};
};

} // namespace

std::unique_ptr<FramePipeline> make_ffmpeg_cpu_pipeline(const std::vector<std::string> & paths)
{
    return std::make_unique<FfmpegCpuPipeline>(paths);
}

} // namespace trdrop
