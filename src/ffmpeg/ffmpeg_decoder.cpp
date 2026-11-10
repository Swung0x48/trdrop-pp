#include "ffmpeg/ffmpeg_decoder.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

#ifndef _WIN32
#include <unistd.h>
#endif

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
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

int open_input(AVFormatContext ** format, const std::string & path)
{
#ifndef _WIN32
    if (path.size() > 3 && path.compare(0, 3, "fd:") == 0 && path.find_first_not_of("0123456789", 3) == std::string::npos)
    {
        // FFmpeg's fd protocol takes the descriptor as an option and reads from where it stands
        const int descriptor = std::atoi(path.c_str() + 3);
        lseek(descriptor, 0, SEEK_SET);
        AVDictionary * options = nullptr;
        av_dict_set_int(&options, "fd", descriptor, 0);
        const int result = avformat_open_input(format, "fd:", nullptr, &options);
        av_dict_free(&options);
        return result;
    }
#endif
    return avformat_open_input(format, path.c_str(), nullptr, nullptr);
}

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

YuvLayout yuv_layout(AVPixelFormat format)
{
    YuvLayout layout;
    const AVPixFmtDescriptor * desc = av_pix_fmt_desc_get(format);
    if (!desc || desc->nb_components < 3) return layout;
    layout.chroma = desc->comp[1].plane == desc->comp[2].plane ? YuvLayout::Interleaved : YuvLayout::Planar;
    layout.shift_x = desc->log2_chroma_w;
    layout.shift_y = desc->log2_chroma_h;
    return layout;
}

Probe probe(const std::string & path)
{
    Probe result;
    AVFormatContext * format = nullptr;
    if (open_input(&format, path) < 0)
    {
        result.error = "could not open '" + path + "'";
        return result;
    }
    const AVCodec * decoder = nullptr;
    const int stream = avformat_find_stream_info(format, nullptr) < 0
                     ? -1 : av_find_best_stream(format, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    if (stream < 0 || !decoder)
    {
        result.error = "no decodable video stream in '" + path + "'";
    }
    else
    {
        result.opened = true;
        result.codec = decoder->name;
        result.codec_id = decoder->id;
        result.format = static_cast<AVPixelFormat>(format->streams[stream]->codecpar->format);
        for (int i = 0;; ++i)
        {
            const AVCodecHWConfig * config = avcodec_get_hw_config(decoder, i);
            if (!config) break;
            if (config->device_type == AV_HWDEVICE_TYPE_VULKAN && (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)) result.vulkan = true;
        }
    }
    avformat_close_input(&format);
    return result;
}

std::string format_name(AVPixelFormat format)
{
    const char * name = av_get_pix_fmt_name(format);
    return name ? name : "unknown";
}

std::string vulkan_decode_blocker(const std::string & path)
{
    const Probe result = probe(path);
    if (!result.opened) return result.error;
    if (!result.vulkan) return "no Vulkan decoding for " + result.codec;
    if (result.format != AV_PIX_FMT_NONE && !is_luma_format(result.format))
    {
        return "pixel format " + format_name(result.format) + " is no 8-bit YUV format";
    }
    return "";
}

VideoDecoder::VideoDecoder(const std::string & path, AVBufferRef * hw_device, int extra_hw_frames)
{
    try
    {
        int result = open_input(&_format, path);
        if (result < 0) throw std::runtime_error("could not open video '" + path + "' (" + error_string(result) + ")");
        result = avformat_find_stream_info(_format, nullptr);
        if (result < 0) throw std::runtime_error("could not read '" + path + "' (" + error_string(result) + ")");
        const AVCodec * decoder = nullptr;
        _stream = av_find_best_stream(_format, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
        if (_stream < 0 || !decoder) throw std::runtime_error("no decodable video stream in '" + path + "'");

        if (hw_device)
        {
            const AVHWDeviceType type = reinterpret_cast<const AVHWDeviceContext *>(hw_device->data)->type;
            _hw_format = type == AV_HWDEVICE_TYPE_MEDIACODEC ? AV_PIX_FMT_MEDIACODEC : AV_PIX_FMT_VULKAN;
        }
        auto open = [&](const AVCodec * codec, AVDictionary ** codec_options) {
            _codec = avcodec_alloc_context3(codec);
            if (!_codec) throw std::runtime_error("out of memory");
            avcodec_parameters_to_context(_codec, _format->streams[_stream]->codecpar);
            _codec->opaque = this;
            // automatic thread count; with hardware decoding the frame threads parse the next
            // frames while the GPU decodes (10 - 15% faster than a single thread)
            _codec->thread_count = 0;
            if (hw_device)
            {
                _codec->hw_device_ctx = av_buffer_ref(hw_device);
                _codec->get_format = &VideoDecoder::_get_format;
                // frames stay referenced while the GPU works on them
                _codec->extra_hw_frames = extra_hw_frames;
            }
            const int opened = avcodec_open2(_codec, codec, codec_options);
            if (opened < 0) avcodec_free_context(&_codec);
            return opened;
        };
#ifdef __ANDROID__
        // The hardware decoder of the device (MediaCodec) if FFmpeg has one for the codec: into frames in
        // memory, or with a MediaCodec device into the window of the device (the frames are then rendered
        // to it, see libavcodec/mediacodec.h)
        {
            const AVCodec * hardware = avcodec_find_decoder_by_name((std::string(decoder->name) + "_mediacodec").c_str());
            int opened = AVERROR_DECODER_NOT_FOUND;
            // Hardware decoders pace themselves for playback unless they are told to run as fast as they can
            // (the highest operating rate); a decoder that refuses the request is opened without it
            for (const char * operating_rate : { "32767", "" })
            {
                if (!hardware || opened >= 0) break;
                AVDictionary * codec_options = nullptr;
                // a native window needs the codec of the NDK
                if (_hw_format == AV_PIX_FMT_MEDIACODEC) av_dict_set(&codec_options, "ndk_codec", "1", 0);
                if (*operating_rate) av_dict_set(&codec_options, "operating_rate", operating_rate, 0);
                opened = open(hardware, &codec_options);
                av_dict_free(&codec_options);
            }
            if (opened < 0)
            {
                _codec = nullptr;
                if (_hw_format == AV_PIX_FMT_MEDIACODEC)
                {
                    throw std::runtime_error("no MediaCodec decoder for " + std::string(decoder->name) + " (" + error_string(opened) + ")");
                }
            }
        }
#endif
        if (!_codec)
        {
            result = open(decoder, nullptr);
            if (result < 0) throw std::runtime_error("could not open the " + std::string(decoder->name) + " decoder (" + error_string(result) + ")");
        }

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
        if (*format == static_cast<VideoDecoder *>(context->opaque)->_hw_format) return *format;
    }
    // no hardware decoding for this stream: fail instead of silently decoding on the CPU
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
            if (_hardware_refused) throw std::runtime_error("no hardware decoding for " + codec_name());
            throw std::runtime_error("decoding failed (" + error_string(result) + ")");
        }
        if (_flushing) return false;

        if (!_packet_pending)
        {
            result = av_read_frame(_format, _packet);
            if (result < 0)
            {
                // end of the file: drain the frames buffered in the decoder
                _flushing = true;
                avcodec_send_packet(_codec, nullptr);
                continue;
            }
        }
        _packet_pending = false;
        if (_packet->stream_index == _stream)
        {
            result = avcodec_send_packet(_codec, _packet);
            if (result == AVERROR(EAGAIN))
            {
                // the decoder wants its frames to be taken first (hardware decoders), the packet is sent again
                _packet_pending = true;
                continue;
            }
            // a broken packet is skipped, like the OpenCV backend does
            if (result < 0 && result != AVERROR_INVALIDDATA)
            {
                av_packet_unref(_packet);
                if (_hardware_refused) throw std::runtime_error("no hardware decoding for " + codec_name());
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

bool VideoDecoder::lossy() const
{
    // codecs with a lossless mode (H.264, HEVC, ...) are treated as lossy, the mode can't be told from the stream
    const AVCodecDescriptor * descriptor = _codec ? avcodec_descriptor_get(_codec->codec_id) : nullptr;
    return descriptor && (descriptor->props & AV_CODEC_PROP_LOSSY) != 0;
}

std::string VideoDecoder::codec_name() const { return _codec && _codec->codec ? _codec->codec->name : "unknown"; }

} // namespace trdrop::ffmpeg
