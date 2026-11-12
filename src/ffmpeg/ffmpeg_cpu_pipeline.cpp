// FFmpeg decoding on the CPU, the Y, U and V planes are compared on the CPU (row parallel).
// Same rule as yuv_diff.comp, so the results are identical to the Vulkan Video pipeline:
// a pixel (x, y) changed if |dY| > threshold or if the chroma sample covering it,
// (x >> shift_x, y >> shift_y), has |dU| > threshold or |dV| > threshold.

#include "frame_pipeline.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

extern "C" {
#include <libavutil/pixdesc.h>
}

#include "ffmpeg/ffmpeg_decoder.h"
#include "ffmpeg/yuv_picture.h"

namespace trdrop {

namespace {

using Clock = std::chrono::steady_clock;

//! per row count of changed pixels between two frames of the same size and format.
//! Works on bands of rows with OpenCV's vectorized operations, the bands run in parallel.
void compare_yuv(const AVFrame * current, const AVFrame * previous, int threshold, std::vector<uint32_t> & counts)
{
    const ffmpeg::YuvLayout layout = ffmpeg::yuv_layout(static_cast<AVPixelFormat>(current->format));
    const int width = current->width;
    const int height = current->height;
    const int chroma_width = AV_CEIL_RSHIFT(width, layout.shift_x);
    counts.assign(static_cast<size_t>(height), 0);

    auto plane = [](const AVFrame * frame, int index, int row, int rows, int bytes_per_row) {
        return cv::Mat(rows, bytes_per_row, CV_8UC1, frame->data[index] + static_cast<ptrdiff_t>(row) * frame->linesize[index],
                       static_cast<size_t>(frame->linesize[index]));
    };
    // 255 where |a - b| > threshold
    auto changed = [threshold](const cv::Mat & a, const cv::Mat & b, cv::Mat & difference, cv::Mat & mask) {
        cv::absdiff(a, b, difference);
        cv::compare(difference, threshold, mask, cv::CMP_GT);
    };

    // bands start at a multiple of the chroma row height, so a band covers whole chroma rows
    constexpr int kBandRows = 32;
    const int bands = (height + kBandRows - 1) / kBandRows;
    cv::parallel_for_(cv::Range(0, bands), [&](const cv::Range & range) {
        cv::Mat difference, mask, chroma_difference, chroma_mask, v_mask, expanded;
        for (int band = range.start; band < range.end; ++band)
        {
            const int y0 = band * kBandRows;
            const int rows = std::min(kBandRows, height - y0);
            changed(plane(current, 0, y0, rows, width), plane(previous, 0, y0, rows, width), difference, mask);

            if (layout.chroma != ffmpeg::YuvLayout::None)
            {
                const int cy0 = y0 >> layout.shift_y;
                const int chroma_rows = ((y0 + rows - 1) >> layout.shift_y) - cy0 + 1;
                if (layout.chroma == ffmpeg::YuvLayout::Interleaved)
                {
                    changed(plane(current, 1, cy0, chroma_rows, 2 * chroma_width), plane(previous, 1, cy0, chroma_rows, 2 * chroma_width),
                            chroma_difference, v_mask);
                    // a UV pair changed if one of its two bytes is 255: compare the pairs as 16-bit values
                    const cv::Mat pairs(chroma_rows, chroma_width, CV_16UC1, v_mask.data, v_mask.step);
                    cv::compare(pairs, 0, chroma_mask, cv::CMP_NE);
                }
                else
                {
                    changed(plane(current, 1, cy0, chroma_rows, chroma_width), plane(previous, 1, cy0, chroma_rows, chroma_width),
                            chroma_difference, chroma_mask);
                    changed(plane(current, 2, cy0, chroma_rows, chroma_width), plane(previous, 2, cy0, chroma_rows, chroma_width),
                            chroma_difference, v_mask);
                    cv::bitwise_or(chroma_mask, v_mask, chroma_mask);
                }
                // every luma pixel gets the flag of the chroma sample covering it (x >> shift_x, y >> shift_y)
                cv::resize(chroma_mask, expanded, cv::Size(chroma_width << layout.shift_x, chroma_rows << layout.shift_y), 0, 0, cv::INTER_NEAREST);
                cv::bitwise_or(mask, expanded(cv::Rect(0, 0, width, rows)), mask);
            }
            for (int row = 0; row < rows; ++row)
            {
                counts[static_cast<size_t>(y0 + row)] = static_cast<uint32_t>(cv::countNonZero(mask.row(row)));
            }
        }
    });
}

class FfmpegCpuPipeline final : public FramePipeline {
public:
    explicit FfmpegCpuPipeline(const std::vector<std::string> & paths)
        : _results(paths.size())
    {
        try
        {
            for (const std::string & path : paths)
            {
                Video video;
                video.decoder = std::make_unique<ffmpeg::VideoDecoder>(path, nullptr);
                video.current = av_frame_alloc();
                video.previous = av_frame_alloc();
                const bool allocated = video.current && video.previous;
                _videos.push_back(std::move(video));
                if (!allocated) throw std::runtime_error("out of memory");
            }
            // decode the first frames to check the pixel format, they are the first submission
            for (size_t i = 0; i < _videos.size(); ++i)
            {
                if (!_videos[i].decoder->next(_videos[i].current)) throw std::runtime_error("'" + paths[i] + "' contains no frames");
                _check_format(_videos[i].current);
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

    FfmpegCpuPipeline(const FfmpegCpuPipeline &) = delete;
    FfmpegCpuPipeline & operator=(const FfmpegCpuPipeline &) = delete;

    std::string description() const override { return "ffmpeg cpu decode + cpu analysis, yuv"; }

    bool submit_next(const std::vector<VideoOptions> & video_options) override
    {
        if (!_first_frames_ready)
        {
            for (Video & video : _videos)
            {
                // the previous frame stays referenced for the comparison
                av_frame_unref(video.previous);
                av_frame_move_ref(video.previous, video.current);
                if (!video.decoder->next(video.current)) return false;
                _check_format(video.current);
            }
        }
        _first_frames_ready = false;

        const Clock::time_point start = Clock::now();
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            const AVFrame * current = _videos[i].current;
            const AVFrame * previous = _videos[i].previous;
            RowCounts & result = _results[i];
            result.cols = current->width;
            result.coding = ffmpeg::frame_coding(current);
            result.compared = previous->data[0] && previous->width == current->width && previous->height == current->height
                           && previous->format == current->format;
            if (result.compared) compare_yuv(current, previous, std::max(0, video_options[i].pixel_difference), result.counts);
            else result.counts.clear();
            result.detail.reset();
            result.resolution.reset();
            if (!_motion.empty())
            {
                const cv::Mat luma(current->height, current->width, CV_8UC1, current->data[0], static_cast<size_t>(current->linesize[0]));
                if (_motion_detail)
                {
                    result.detail = std::make_shared<MotionDetail>();
                    ffmpeg::motion_picture(ffmpeg::yuv_planes(current), result.detail->picture, _scratch);
                }
                result.motion = _motion[i].estimate(luma, _motion_detail ? &result.detail->points : nullptr);
                if (_resolution)
                {
                    // the edge columns are only used for frames of a pan
                    result.resolution = std::make_shared<ResolutionFrame>();
                    const ResolutionLayout layout = resolution_layout(luma.cols, luma.rows, video_options[i].resolution_rows_from, video_options[i].resolution_rows_to);
                    measure_resolution(luma, layout, resolution_pan_usable(resolution_pan(result.motion, luma.cols, luma.rows)), *result.resolution);
                }
            }
        }
        _analysis += Clock::now() - start;
        _pending = 1;
        ++_frames_read;
        return true;
    }

    size_t pending() const override { return _pending; }
    size_t max_pending() const override { return 1; }

    void collect(std::vector<RowCounts> & out) override
    {
        out = _results;
        _pending = 0;
    }

    void enable_motion() override { _motion.resize(_videos.size()); }
    void enable_motion_detail() override { _motion_detail = true; }
    void enable_resolution() override
    {
        if (_motion.empty()) throw std::logic_error("the resolution analysis needs the motion analysis");
        _resolution = true;
    }

    double analysis_seconds() const override { return std::chrono::duration<double>(_analysis).count(); }

    size_t count() const override { return _videos.size(); }
    double recorded_framerate(size_t index) const override { return _videos.at(index).decoder->framerate(); }
    long long reported_frame_count(size_t index) const override { return _videos.at(index).decoder->frame_count(); }
    int frame_width(size_t index) const override { return _videos.at(index).decoder->width(); }
    int frame_height(size_t index) const override { return _videos.at(index).decoder->height(); }
    long long frames_read() const override { return _frames_read; }
    std::string codec_name(size_t index) const override { return _videos.at(index).decoder->codec_name(); }
    bool lossy_codec(size_t index) const override { return _videos.at(index).decoder->lossy(); }

private:
    struct Video {
        std::unique_ptr<ffmpeg::VideoDecoder> decoder;
        AVFrame * current = nullptr;
        AVFrame * previous = nullptr;
    };

    static void _check_format(const AVFrame * frame)
    {
        const AVPixelFormat format = static_cast<AVPixelFormat>(frame->format);
        if (!ffmpeg::is_luma_format(format))
        {
            const char * name = av_get_pix_fmt_name(format);
            throw std::runtime_error(std::string("pixel format ") + (name ? name : "unknown") + " is no 8-bit YUV format");
        }
        for (int plane = 0; plane < 3; ++plane)
        {
            if (frame->data[plane] && frame->linesize[plane] <= 0) throw std::runtime_error("unsupported frame layout");
        }
    }

    void _release()
    {
        for (Video & video : _videos)
        {
            av_frame_free(&video.current);
            av_frame_free(&video.previous);
            video.decoder.reset();
        }
        _videos.clear();
    }

    std::vector<Video> _videos;
    std::vector<RowCounts> _results;
    std::vector<MotionEstimator> _motion;
    bool _motion_detail = false;
    bool _resolution = false;
    cv::Mat _scratch;
    bool _first_frames_ready = false;
    size_t _pending = 0;
    long long _frames_read = 0;
    Clock::duration _analysis {};
};

} // namespace

std::unique_ptr<FramePipeline> make_ffmpeg_cpu_pipeline(const std::vector<std::string> & paths)
{
    return std::make_unique<FfmpegCpuPipeline>(paths);
}

DecoderChoice choose_decoder(const std::vector<std::string> & paths, bool vulkan_available, bool prefer_gpu)
{
    // CPUs with at least this many threads decode 4K H.264 faster than the hardware decoder
    // (measured: i7-14700KF 28 threads ~320 fps on the CPU, RTX 4070 (SUPER) ~230 fps)
    constexpr unsigned kH264CpuThreads = 16;

    bool yuv = true;
    bool vulkan = true;
    bool h264 = true;
    std::string not_yuv;
    std::string no_vulkan;
    for (const std::string & path : paths)
    {
        const ffmpeg::Probe probe = ffmpeg::probe(path);
        // let OpenCV report the error / try its own backends
        if (!probe.opened) return { "opencv", probe.error };
        if (probe.format != AV_PIX_FMT_NONE && !ffmpeg::is_luma_format(probe.format))
        {
            yuv = false;
            not_yuv = "pixel format " + ffmpeg::format_name(probe.format) + " is no 8-bit YUV format";
        }
        if (!probe.vulkan)
        {
            vulkan = false;
            no_vulkan = "no Vulkan decoding for " + probe.codec;
        }
        h264 &= probe.codec_id == AV_CODEC_ID_H264;
    }
    if (!yuv) return { "opencv", not_yuv };
    if (vulkan_available && vulkan)
    {
        const unsigned threads = std::thread::hardware_concurrency();
        if (h264 && threads >= kH264CpuThreads && !prefer_gpu)
        {
            return { "ffmpeg", "H.264 decodes faster on the CPU with " + std::to_string(threads) + " threads" };
        }
        return { "vulkan", "" };
    }
    return { "ffmpeg", vulkan_available ? no_vulkan : "this build has no Vulkan" };
}

} // namespace trdrop
