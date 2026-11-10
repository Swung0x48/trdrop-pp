#pragma once

#include <string>

#include "ffmpeg/ffmpeg_decoder.h"

namespace trdrop::ffmpeg {

//! a video encoder and the file its packets are written to
class EncodedVideo {
public:
    //! Frames in memory: H.264 with the first encoder that works on the machine (MPEG-4 if none does),
    //! the encoder is fed with format() frames (planar 4:2:0 or NV12).
    //! throws std::runtime_error if the file can't be written
    EncodedVideo(const std::string & path, int width, int height, double fps);
    //! Frames on a hardware device: the encoder `encoder` is fed with frames of `hw_frames`
    EncodedVideo(const std::string & path, int width, int height, double fps, const char * encoder, AVBufferRef * hw_frames);
    ~EncodedVideo();

    EncodedVideo(const EncodedVideo &) = delete;
    EncodedVideo & operator=(const EncodedVideo &) = delete;

    const std::string & encoder_name() const { return _encoder_name; }
    AVPixelFormat format() const { return _codec->pix_fmt; }
    int width() const { return _codec->width; }
    int height() const { return _codec->height; }

    //! the next frame, its pts is set here
    void encode(AVFrame * frame);
    //! writes what is left, the file is complete afterwards
    void finish();

private:
    void _open(int width, int height, double fps, const char * encoder, AVBufferRef * hw_frames);
    void _check(int result, const char * what) const;
    void _write_packets();
    void _release();

    std::string _path;
    std::string _encoder_name;
    AVFormatContext * _format = nullptr;
    AVCodecContext * _codec = nullptr;
    AVStream * _stream = nullptr;
    AVPacket * _packet = nullptr;
    int64_t _next_pts = 0;
    bool _finished = false;
};

} // namespace trdrop::ffmpeg
