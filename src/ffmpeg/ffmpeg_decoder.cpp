#include "ffmpeg/ffmpeg_decoder.h"

#include <cmath>
#include <stdexcept>

extern "C" {
#include <libavutil/pixdesc.h>
}

namespace trdrop::ffmpeg {

namespace {

constexpr double kEpsilon = 0.000025; // eps_zero of the OpenCV FFmpeg backend

double to_double(AVRational value)
{
    return value.num == 0 ? 0.0 : static_cast<double>(value.num) / static_cast<double>(value.den);
}

} // namespace

std::string error_string(int error)
{
    char buffer[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(error, buffer, sizeof(buffer));
    return buffer;
}

bool is_luma_format(AVPixelFormat format)
{
    switch (format)
    {
    case AV_PIX_FMT_GRAY8:
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUVJ420P:
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUVJ422P:
    case AV_PIX_FMT_YUV440P:
    case AV_PIX_FMT_YUVJ440P:
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_YUVJ444P:
    case AV_PIX_FMT_YUVA420P:
    case AV_PIX_FMT_NV12:
    case AV_PIX_FMT_NV21:
    case AV_PIX_FMT_NV16:
    case AV_PIX_FMT_NV24:
    case AV_PIX_FMT_NV42:
        return true;
    default:
        return false;
    }
}

std::string vulkan_decode_blocker(const std::string & path)
{
    AVFormatContext * format = nullptr;
    if (avformat_open_input(&format, path.c_str(), nullptr, nullptr) < 0) return "could not open '" + path + "'";
    std::string reason;
    const AVCodec * decoder = nullptr;
    const int stream = avformat_find_stream_info(format, nullptr) < 0
                     ? -1 : av_find_best_stream(format, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    if (stream < 0 || !decoder)
    {
        reason = "no decodable video stream in '" + path + "'";
    }
    else
    {
        bool vulkan = false;
        for (int i = 0;; ++i)
        {
            const AVCodecHWConfig * config = avcodec_get_hw_config(decoder, i);
            if (!config) break;
            if (config->device_type == AV_HWDEVICE_TYPE_VULKAN && (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)) vulkan = true;
        }
        const AVPixelFormat pixel_format = static_cast<AVPixelFormat>(format->streams[stream]->codecpar->format);
        if (!vulkan)
        {
            reason = std::string("no Vulkan decoding for ") + decoder->name;
        }
        else if (pixel_format != AV_PIX_FMT_NONE && !is_luma_format(pixel_format))
        {
            const char * name = av_get_pix_fmt_name(pixel_format);
            reason = std::string("pixel format ") + (name ? name : "unknown") + " has no 8-bit luma plane";
        }
    }
    avformat_close_input(&format);
    return reason;
}

VideoDecoder::VideoDecoder(const std::string & path, AVBufferRef * hw_device, int extra_hw_frames)
{
    try
    {
        int result = avformat_open_input(&_format, path.c_str(), nullptr, nullptr);
        if (result < 0) throw std::runtime_error("could not open video '" + path + "' (" + error_string(result) + ")");
        result = avformat_find_stream_info(_format, nullptr);
        if (result < 0) throw std::runtime_error("could not read '" + path + "' (" + error_string(result) + ")");
        const AVCodec * decoder = nullptr;
        _stream = av_find_best_stream(_format, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
        if (_stream < 0 || !decoder) throw std::runtime_error("no decodable video stream in '" + path + "'");

        _codec = avcodec_alloc_context3(decoder);
        if (!_codec) throw std::runtime_error("out of memory");
        avcodec_parameters_to_context(_codec, _format->streams[_stream]->codecpar);
        _codec->opaque = this;
        if (hw_device)
        {
            _codec->hw_device_ctx = av_buffer_ref(hw_device);
            _codec->get_format = &VideoDecoder::_get_format;
            // frames stay referenced while the GPU works on them
            _codec->extra_hw_frames = extra_hw_frames;
        }
        else
        {
            _codec->thread_count = 0; // automatic
        }
        result = avcodec_open2(_codec, decoder, nullptr);
        if (result < 0) throw std::runtime_error("could not open the " + std::string(decoder->name) + " decoder (" + error_string(result) + ")");

        _packet = av_packet_alloc();
        if (!_packet) throw std::runtime_error("out of memory");
    }
    catch (...)
    {
        avcodec_free_context(&_codec);
        avformat_close_input(&_format);
        throw;
    }
}

VideoDecoder::~VideoDecoder()
{
    av_packet_free(&_packet);
    avcodec_free_context(&_codec);
    avformat_close_input(&_format);
}

AVPixelFormat VideoDecoder::_get_format(AVCodecContext * context, const AVPixelFormat * formats)
{
    for (const AVPixelFormat * format = formats; *format != AV_PIX_FMT_NONE; ++format)
    {
        if (*format == AV_PIX_FMT_VULKAN) return *format;
    }
    // no Vulkan decoding for this stream: fail instead of silently decoding on the CPU
    static_cast<VideoDecoder *>(context->opaque)->_hardware_refused = true;
    return AV_PIX_FMT_NONE;
}

bool VideoDecoder::next(AVFrame * frame)
{
    av_frame_unref(frame);
    while (true)
    {
        int result = avcodec_receive_frame(_codec, frame);
        if (result == 0) return true;
        if (result == AVERROR_EOF) return false;
        if (result != AVERROR(EAGAIN))
        {
            if (_hardware_refused) throw std::runtime_error("no Vulkan decoding for " + codec_name());
            throw std::runtime_error("decoding failed (" + error_string(result) + ")");
        }
        if (_flushing) return false;

        result = av_read_frame(_format, _packet);
        if (result < 0)
        {
            // end of the file: drain the frames buffered in the decoder
            _flushing = true;
            avcodec_send_packet(_codec, nullptr);
            continue;
        }
        if (_packet->stream_index == _stream)
        {
            result = avcodec_send_packet(_codec, _packet);
            // a broken packet is skipped, like the OpenCV backend does
            if (result < 0 && result != AVERROR_INVALIDDATA)
            {
                av_packet_unref(_packet);
                if (_hardware_refused) throw std::runtime_error("no Vulkan decoding for " + codec_name());
                throw std::runtime_error("decoding failed (" + error_string(result) + ")");
            }
        }
        av_packet_unref(_packet);
    }
}

double VideoDecoder::framerate() const
{
    const AVStream * stream = _format->streams[_stream];
    double fps = to_double(stream->avg_frame_rate);
    if (fps < kEpsilon) fps = to_double(av_guess_frame_rate(_format, const_cast<AVStream *>(stream), nullptr));
    if (fps < kEpsilon) fps = 1.0 / to_double(stream->time_base);
    return fps;
}

long long VideoDecoder::frame_count() const
{
    const AVStream * stream = _format->streams[_stream];
    long long frames = stream->nb_frames;
    if (frames == 0)
    {
        double seconds = static_cast<double>(_format->duration) / static_cast<double>(AV_TIME_BASE);
        if (seconds < kEpsilon) seconds = static_cast<double>(stream->duration) * to_double(stream->time_base);
        frames = static_cast<long long>(std::floor(seconds * framerate() + 0.5));
    }
    return frames;
}

int VideoDecoder::width() const { return _format->streams[_stream]->codecpar->width; }
int VideoDecoder::height() const { return _format->streams[_stream]->codecpar->height; }

std::string VideoDecoder::codec_name() const { return _codec && _codec->codec ? _codec->codec->name : "unknown"; }

} // namespace trdrop::ffmpeg
