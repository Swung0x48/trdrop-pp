#pragma once

#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixfmt.h>
}

namespace trdrop::ffmpeg {

//! FFmpeg error code as text
std::string error_string(int error);

//! true for the 8-bit YUV formats whose first plane is the full resolution luma (Y) plane
bool is_luma_format(AVPixelFormat format);

//! chroma layout of a format accepted by is_luma_format
struct YuvLayout {
    enum Chroma { None = 0, Interleaved = 1, Planar = 2 };
    //! None: gray, Interleaved: UV pairs in plane 1 (NV12, ...), Planar: U in plane 1, V in plane 2
    Chroma chroma = None;
    //! chroma subsampling (log2), e.g. 1 / 1 for 4:2:0
    int shift_x = 0;
    int shift_y = 0;
};
YuvLayout yuv_layout(AVPixelFormat format);

//! container / codec parameters of the video stream (no decoding)
struct Probe {
    bool opened = false;
    std::string error;
    std::string codec;
    AVCodecID codec_id = AV_CODEC_ID_NONE;
    //! AV_PIX_FMT_NONE if the container doesn't tell
    AVPixelFormat format = AV_PIX_FMT_NONE;
    //! the decoder supports Vulkan hardware decoding
    bool vulkan = false;
};
Probe probe(const std::string & path);

std::string format_name(AVPixelFormat format);

//! cheap check from the container / codec parameters (no device, no decoding):
//! empty if the video may be decodable with Vulkan Video into an 8-bit YUV format, otherwise the reason
std::string vulkan_decode_blocker(const std::string & path);

//! demuxes and decodes the video stream of one file
class VideoDecoder {
public:
    //! hw_device: decode with this hardware device (AV_PIX_FMT_VULKAN frames), nullptr = software decoding.
    //! throws std::runtime_error if the file can't be opened
    VideoDecoder(const std::string & path, AVBufferRef * hw_device, int extra_hw_frames = 0);
    ~VideoDecoder();

    VideoDecoder(const VideoDecoder &) = delete;
    VideoDecoder & operator=(const VideoDecoder &) = delete;

    //! decodes the next frame into `frame` (unreferenced first), false at the end of the stream
    bool next(AVFrame * frame);

    // same values as the OpenCV FFmpeg backend reports
    double framerate() const;
    long long frame_count() const;
    int width() const;
    int height() const;

    std::string codec_name() const;
    //! true if the decoder was asked for hardware frames but didn't get them
    bool hardware_refused() const { return _hardware_refused; }

private:
    static AVPixelFormat _get_format(AVCodecContext * context, const AVPixelFormat * formats);

    AVFormatContext * _format = nullptr;
    AVCodecContext * _codec = nullptr;
    AVPacket * _packet = nullptr;
    int _stream = -1;
    bool _flushing = false;
    bool _hardware_refused = false;
};

} // namespace trdrop::ffmpeg
