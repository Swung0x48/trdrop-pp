#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace trdrop {

//! holds all opened videos and reads them in lockstep (VideoCaptureList in the original)
class VideoSources {
public:
    //! opens every path, throws std::runtime_error if one can't be opened
    explicit VideoSources(const std::vector<std::string> & paths);

    //! reads the next frame of every video into `frames` (one Mat per video, existing buffers
    //! of the right size / type are reused), returns false as soon as one video ends
    bool read_next_frames(std::vector<cv::Mat> & frames);

    size_t count() const { return _captures.size(); }

    //! raw framerate the video was recorded with (CAP_PROP_FPS)
    double recorded_framerate(size_t index) const;
    //! frames reported by the container, may be 0 / inaccurate for some formats
    long long reported_frame_count(size_t index) const;
    int frame_width(size_t index) const;
    int frame_height(size_t index) const;
    //! FOURCC / short name of the codec as reported by OpenCV
    std::string codec_name(size_t index) const;
    //! false for the known lossless / raw codecs, true for everything else
    bool lossy_codec(size_t index) const;
    //! frames read so far (same for every video)
    long long frames_read() const { return _frames_read; }

private:
    std::vector<cv::VideoCapture> _captures;
    long long _frames_read = 0;
};

} // namespace trdrop
