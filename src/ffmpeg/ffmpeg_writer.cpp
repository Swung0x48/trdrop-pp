// Encodes pictures with libavcodec and writes them with libavformat.

#include "video_writer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/mathematics.h>
}

#include "ffmpeg/ffmpeg_decoder.h"
#include "ffmpeg/ffmpeg_encoder.h"

namespace trdrop {

namespace ffmpeg {

namespace {

//! H.264 encoders for frames in memory, in the order they are tried: the GPU's, a software encoder, the
//! operating system's, and MPEG-4 as the one every build has. Which exist depends on the FFmpeg build,
//! which work on the machine.
const char * const kEncoders[] = {
#ifdef __ANDROID__
    // the hardware encoder of the device
    "h264_mediacodec",
#endif
    "h264_nvenc", "h264_amf", "h264_qsv", "h264_videotoolbox", "libx264", "h264_mf", "libopenh264", "mpeg4",
};

//! bits per pixel and frame, at 60 fps (1920x1080: 8.7 Mbit/s); arrows and text need more than the picture
constexpr double kBitsPerPixel = 0.07;

//! the format an encoder is fed with: planar 4:2:0 if it takes it, otherwise NV12
AVPixelFormat memory_format(const AVCodec * codec)
{
    const void * configs = nullptr;
    int count = 0;
    if (avcodec_get_supported_config(nullptr, codec, AV_CODEC_CONFIG_PIX_FORMAT, 0, &configs, &count) < 0) return AV_PIX_FMT_NONE;
    if (!configs) return AV_PIX_FMT_YUV420P; // any format
    const AVPixelFormat * formats = static_cast<const AVPixelFormat *>(configs);
    bool nv12 = false;
    for (int i = 0; i < count; ++i)
    {
        if (formats[i] == AV_PIX_FMT_YUV420P) return AV_PIX_FMT_YUV420P;
        nv12 |= formats[i] == AV_PIX_FMT_NV12;
    }
    return nv12 ? AV_PIX_FMT_NV12 : AV_PIX_FMT_NONE;
}

} // namespace

EncodedVideo::EncodedVideo(const std::string & path, int width, int height, double fps)
    : EncodedVideo(path, width, height, fps, nullptr, nullptr)
{ }

EncodedVideo::EncodedVideo(const std::string & path, int width, int height, double fps, const char * encoder, AVBufferRef * hw_frames)
    : _path(path)
{
    try
    {
        _open(width, height, fps, encoder, hw_frames);
    }
    catch (...)
    {
        _release();
        throw;
    }
}

EncodedVideo::~EncodedVideo() { _release(); }

void EncodedVideo::_check(int result, const char * what) const
{
    if (result < 0) throw std::runtime_error("writing '" + _path + "' failed (" + what + ": " + error_string(result) + ")");
}

void EncodedVideo::_open(int width, int height, double fps, const char * encoder, AVBufferRef * hw_frames)
{
    if (avformat_alloc_output_context2(&_format, nullptr, nullptr, _path.c_str()) < 0 || !_format)
    {
        throw std::runtime_error("no video format for the name '" + _path + "', use .mp4, .mkv or .mov");
    }
    const AVRational rate = av_d2q(fps, 1000000);
    if (rate.num <= 0 || rate.den <= 0) throw std::runtime_error("video writer: invalid framerate");

    // an encoder that doesn't work here (no such GPU) says so in FFmpeg's log, as do some that work: the
    // result is reported by this program
    struct QuietLog {
        int level = av_log_get_level();
        QuietLog() { av_log_set_level(AV_LOG_QUIET); }
        ~QuietLog() { av_log_set_level(level); }
    } quiet;
    std::string tried;
    const char * const given[] = { encoder };
    const char * const * names = encoder ? given : kEncoders;
    const size_t name_count = encoder ? 1 : sizeof(kEncoders) / sizeof(kEncoders[0]);
    for (size_t n = 0; n < name_count; ++n)
    {
        const char * name = names[n];
        const AVCodec * codec = avcodec_find_encoder_by_name(name);
        if (!codec || avformat_query_codec(_format->oformat, codec->id, FF_COMPLIANCE_NORMAL) != 1) continue;
        const AVPixelFormat format = hw_frames ? reinterpret_cast<const AVHWFramesContext *>(hw_frames->data)->format : memory_format(codec);
        if (format == AV_PIX_FMT_NONE) continue;
        AVCodecContext * context = avcodec_alloc_context3(codec);
        if (!context) throw std::runtime_error("out of memory");
        context->width = width;
        context->height = height;
        context->pix_fmt = format;
        context->time_base = av_inv_q(rate);
        context->framerate = rate;
        // a key frame every second, no B-frames: the file is stepped through frame by frame
        context->gop_size = std::max(1, static_cast<int>(std::lround(fps)));
        context->max_b_frames = 0;
        context->bit_rate = static_cast<int64_t>(kBitsPerPixel * width * height * 60.0 * std::sqrt(std::max(1.0, fps) / 60.0));
        if (_format->oformat->flags & AVFMT_GLOBALHEADER) context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        if (hw_frames)
        {
            context->hw_frames_ctx = av_buffer_ref(hw_frames);
            if (!context->hw_frames_ctx)
            {
                avcodec_free_context(&context);
                throw std::runtime_error("out of memory");
            }
        }
        const int result = avcodec_open2(context, codec, nullptr);
        if (result < 0)
        {
            // e.g. no such GPU in this machine
            tried += std::string(tried.empty() ? "" : ", ") + name + ": " + error_string(result);
            avcodec_free_context(&context);
            continue;
        }
        _codec = context;
        _encoder_name = name;
        break;
    }
    if (!_codec) throw std::runtime_error("no video encoder for '" + _path + "' works" + (tried.empty() ? "" : " (" + tried + ")"));

    _stream = avformat_new_stream(_format, nullptr);
    if (!_stream) throw std::runtime_error("out of memory");
    _stream->time_base = _codec->time_base;
    _stream->avg_frame_rate = rate;
    _check(avcodec_parameters_from_context(_stream->codecpar, _codec), "avcodec_parameters_from_context");
    if (!(_format->oformat->flags & AVFMT_NOFILE))
    {
        const int result = avio_open(&_format->pb, _path.c_str(), AVIO_FLAG_WRITE);
        if (result < 0) throw std::runtime_error("could not open '" + _path + "' for writing (" + error_string(result) + ")");
    }
    _check(avformat_write_header(_format, nullptr), "avformat_write_header");
    _packet = av_packet_alloc();
    if (!_packet) throw std::runtime_error("out of memory");
}

void EncodedVideo::encode(AVFrame * frame)
{
    frame->pts = _next_pts++;
    _check(avcodec_send_frame(_codec, frame), "avcodec_send_frame");
    _write_packets();
}

void EncodedVideo::finish()
{
    if (!_codec || _finished) return;
    _finished = true;
    _check(avcodec_send_frame(_codec, nullptr), "avcodec_send_frame");
    _write_packets();
    _check(av_write_trailer(_format), "av_write_trailer");
    if (!(_format->oformat->flags & AVFMT_NOFILE)) _check(avio_closep(&_format->pb), "closing the file");
}

void EncodedVideo::_write_packets()
{
    while (true)
    {
        const int result = avcodec_receive_packet(_codec, _packet);
        if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) return;
        _check(result, "avcodec_receive_packet");
        if (_packet->duration == 0) _packet->duration = 1;
        av_packet_rescale_ts(_packet, _codec->time_base, _stream->time_base);
        _packet->stream_index = _stream->index;
        _check(av_interleaved_write_frame(_format, _packet), "av_interleaved_write_frame");
        av_packet_unref(_packet);
    }
}

void EncodedVideo::_release()
{
    av_packet_free(&_packet);
    avcodec_free_context(&_codec);
    if (_format)
    {
        if (_format->pb && !(_format->oformat->flags & AVFMT_NOFILE)) avio_closep(&_format->pb);
        avformat_free_context(_format);
        _format = nullptr;
    }
}

} // namespace ffmpeg

namespace {

//! pictures in memory: converted to the format of the encoder
class FfmpegVideoWriter final : public VideoWriter {
public:
    FfmpegVideoWriter(const std::string & path, int width, int height, double fps)
        : _video(path, width, height, fps)
    {
        _frame = av_frame_alloc();
        if (!_frame) throw std::runtime_error("out of memory");
        _frame->format = _video.format();
        _frame->width = width;
        _frame->height = height;
        if (av_frame_get_buffer(_frame, 0) < 0)
        {
            av_frame_free(&_frame);
            throw std::runtime_error("out of memory");
        }
    }

    ~FfmpegVideoWriter() override { av_frame_free(&_frame); }

    FfmpegVideoWriter(const FfmpegVideoWriter &) = delete;
    FfmpegVideoWriter & operator=(const FfmpegVideoWriter &) = delete;

    std::string description() const override { return _video.encoder_name(); }

    void write(const cv::Mat & picture) override
    {
        if (picture.cols != _video.width() || picture.rows != _video.height() || picture.type() != CV_8UC3)
        {
            throw std::runtime_error("video writer: unexpected picture");
        }
        if (av_frame_make_writable(_frame) < 0) throw std::runtime_error("out of memory");
        cv::cvtColor(picture, _yuv, cv::COLOR_BGR2YUV_I420);
        const int width = picture.cols;
        const int height = picture.rows;
        const int chroma_width = width / 2;
        const int chroma_height = height / 2;
        const uint8_t * y = _yuv.data;
        const uint8_t * u = y + static_cast<size_t>(width) * height;
        const uint8_t * v = u + static_cast<size_t>(chroma_width) * chroma_height;
        for (int row = 0; row < height; ++row)
        {
            std::memcpy(_frame->data[0] + static_cast<ptrdiff_t>(row) * _frame->linesize[0], y + static_cast<size_t>(row) * width, static_cast<size_t>(width));
        }
        for (int row = 0; row < chroma_height; ++row)
        {
            const uint8_t * u_row = u + static_cast<size_t>(row) * chroma_width;
            const uint8_t * v_row = v + static_cast<size_t>(row) * chroma_width;
            if (_video.format() == AV_PIX_FMT_NV12)
            {
                uint8_t * pairs = _frame->data[1] + static_cast<ptrdiff_t>(row) * _frame->linesize[1];
                for (int x = 0; x < chroma_width; ++x)
                {
                    pairs[2 * x] = u_row[x];
                    pairs[2 * x + 1] = v_row[x];
                }
            }
            else
            {
                std::memcpy(_frame->data[1] + static_cast<ptrdiff_t>(row) * _frame->linesize[1], u_row, static_cast<size_t>(chroma_width));
                std::memcpy(_frame->data[2] + static_cast<ptrdiff_t>(row) * _frame->linesize[2], v_row, static_cast<size_t>(chroma_width));
            }
        }
        _video.encode(_frame);
    }

    void finish() override { _video.finish(); }

private:
    ffmpeg::EncodedVideo _video;
    AVFrame * _frame = nullptr;
    cv::Mat _yuv;
};

} // namespace

std::unique_ptr<VideoWriter> make_ffmpeg_video_writer(const std::string & path, int width, int height, double fps)
{
    return std::make_unique<FfmpegVideoWriter>(path, width, height, fps);
}

} // namespace trdrop
