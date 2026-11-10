// Encodes pictures with libavcodec and writes them with libavformat.

#include "video_writer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

extern "C" {
#include <libavutil/mathematics.h>
}

#include "ffmpeg/ffmpeg_decoder.h"

namespace trdrop {

namespace {

//! H.264 encoders in the order they are tried: the GPU's, a software encoder, the operating system's, and
//! MPEG-4 as the one every build has. Which exist depends on the FFmpeg build, which work on the machine.
const char * const kEncoders[] = {
    "h264_nvenc", "h264_amf", "h264_qsv", "h264_videotoolbox", "libx264", "h264_mf", "libopenh264", "mpeg4",
};

//! bits per pixel and frame, at 60 fps (1920x1080: 8.7 Mbit/s); arrows and text need more than the picture
constexpr double kBitsPerPixel = 0.07;

class FfmpegVideoWriter final : public VideoWriter {
public:
    FfmpegVideoWriter(const std::string & path, int width, int height, double fps)
        : _path(path)
    {
        try
        {
            _open(width, height, fps);
        }
        catch (...)
        {
            _release();
            throw;
        }
    }

    ~FfmpegVideoWriter() override { _release(); }

    FfmpegVideoWriter(const FfmpegVideoWriter &) = delete;
    FfmpegVideoWriter & operator=(const FfmpegVideoWriter &) = delete;

    std::string description() const override { return _encoder_name; }

    void write(const cv::Mat & picture) override
    {
        if (!_codec || picture.cols != _codec->width || picture.rows != _codec->height || picture.type() != CV_8UC3)
        {
            throw std::runtime_error("video writer: unexpected picture");
        }
        _check(av_frame_make_writable(_frame), "av_frame_make_writable");
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
            if (_codec->pix_fmt == AV_PIX_FMT_NV12)
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
        _frame->pts = _next_pts++;
        _check(avcodec_send_frame(_codec, _frame), "avcodec_send_frame");
        _write_packets();
    }

    void finish() override
    {
        if (!_codec || _finished) return;
        _finished = true;
        _check(avcodec_send_frame(_codec, nullptr), "avcodec_send_frame");
        _write_packets();
        _check(av_write_trailer(_format), "av_write_trailer");
        if (!(_format->oformat->flags & AVFMT_NOFILE)) _check(avio_closep(&_format->pb), "closing the file");
    }

private:
    void _check(int result, const char * what) const
    {
        if (result < 0) throw std::runtime_error("writing '" + _path + "' failed (" + what + ": " + ffmpeg::error_string(result) + ")");
    }

    //! the format the encoder is fed with: planar 4:2:0 if it takes it, otherwise NV12
    static AVPixelFormat _pixel_format(const AVCodec * codec)
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

    void _open(int width, int height, double fps)
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
        for (const char * name : kEncoders)
        {
            const AVCodec * codec = avcodec_find_encoder_by_name(name);
            if (!codec || avformat_query_codec(_format->oformat, codec->id, FF_COMPLIANCE_NORMAL) != 1) continue;
            const AVPixelFormat format = _pixel_format(codec);
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
            const int result = avcodec_open2(context, codec, nullptr);
            if (result < 0)
            {
                // e.g. no such GPU in this machine
                tried += std::string(tried.empty() ? "" : ", ") + name + ": " + ffmpeg::error_string(result);
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
            if (result < 0) throw std::runtime_error("could not open '" + _path + "' for writing (" + ffmpeg::error_string(result) + ")");
        }
        _check(avformat_write_header(_format, nullptr), "avformat_write_header");

        _frame = av_frame_alloc();
        _packet = av_packet_alloc();
        if (!_frame || !_packet) throw std::runtime_error("out of memory");
        _frame->format = _codec->pix_fmt;
        _frame->width = width;
        _frame->height = height;
        _check(av_frame_get_buffer(_frame, 0), "av_frame_get_buffer");
    }

    void _write_packets()
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

    void _release()
    {
        av_frame_free(&_frame);
        av_packet_free(&_packet);
        avcodec_free_context(&_codec);
        if (_format)
        {
            if (_format->pb && !(_format->oformat->flags & AVFMT_NOFILE)) avio_closep(&_format->pb);
            avformat_free_context(_format);
            _format = nullptr;
        }
    }

    std::string _path;
    std::string _encoder_name;
    AVFormatContext * _format = nullptr;
    AVCodecContext * _codec = nullptr;
    AVStream * _stream = nullptr;
    AVFrame * _frame = nullptr;
    AVPacket * _packet = nullptr;
    cv::Mat _yuv;
    int64_t _next_pts = 0;
    bool _finished = false;
};

} // namespace

std::unique_ptr<VideoWriter> make_ffmpeg_video_writer(const std::string & path, int width, int height, double fps)
{
    return std::make_unique<FfmpegVideoWriter>(path, width, height, fps);
}

} // namespace trdrop
