#pragma once

#include <memory>
#include <string>
#include <vector>

#include "diff_engine.h"
#include "motion_video_target.h"
#include "options.h"

namespace trdrop {

//! Reads the input videos in lockstep and computes the per row differences of consecutive frames.
//!
//! Per set of frames: submit_next() decodes the next frame of every video and starts the comparison.
//! Up to max_pending() submissions may be in flight, collect() returns them in submission order.
class FramePipeline {
public:
    virtual ~FramePipeline() = default;

    //! human readable decoder / backend name for the input info
    virtual std::string description() const = 0;

    //! false as soon as one video ends
    virtual bool submit_next(const std::vector<VideoOptions> & video_options) = 0;
    virtual size_t pending() const = 0;
    virtual size_t max_pending() const = 0;
    //! waits for the oldest submission, one entry per video
    virtual void collect(std::vector<RowCounts> & out) = 0;

    //! also measure the motion between the frames (RowCounts::motion), has to be called before the first
    //! submit_next(). throws std::runtime_error if the pipeline can't do it
    virtual void enable_motion();
    //! also keep the picture and the tracked points of every frame (RowCounts::detail), after enable_motion().
    //! throws std::runtime_error if the pipeline can't do it
    virtual void enable_motion_detail();
    //! The motion video of video `index`, drawn and encoded by the pipeline itself where the frames are (on
    //! the GPU): its frames get MotionDetail::gpu_picture instead of a picture. nullptr if the pipeline doesn't
    //! do that or can't here, `reason` then says why (empty: it never does). Before the first submit_next()
    virtual std::unique_ptr<MotionVideoTarget> start_motion_video(size_t index, const std::string & path, double fps, std::string & reason);

    //! main thread time spent in the comparison (without decoding), < 0 if decoding and
    //! comparison can't be separated (both run on the GPU)
    virtual double analysis_seconds() const = 0;

    // video information
    virtual size_t count() const = 0;
    //! raw framerate the video was recorded with
    virtual double recorded_framerate(size_t index) const = 0;
    //! frames reported by the container, may be 0 / inaccurate for some formats
    virtual long long reported_frame_count(size_t index) const = 0;
    virtual int frame_width(size_t index) const = 0;
    virtual int frame_height(size_t index) const = 0;
    //! sets of frames submitted so far
    virtual long long frames_read() const = 0;
    //! short name of the video codec (e.g. h264, ffv1)
    virtual std::string codec_name(size_t index) const = 0;
    //! true if the codec is (or may be) lossy, i.e. identical source frames may decode to slightly different frames
    virtual bool lossy_codec(size_t index) const = 0;

    //! progress of the shortest video, 0 .. 1 (or < 0 if unknown)
    double progress() const;
};

//! OpenCV decoding (BGR frames, grayscale via cv::cvtColor) with the given engine
std::unique_ptr<FramePipeline> make_opencv_pipeline(const std::vector<std::string> & paths, std::unique_ptr<DiffEngine> engine);

#ifdef USE_FFMPEG
//! FFmpeg decoding on the CPU, compares the Y, U and V planes on the CPU.
//! throws std::runtime_error if an input has no 8-bit YUV format
std::unique_ptr<FramePipeline> make_ffmpeg_cpu_pipeline(const std::vector<std::string> & paths);

//! the decoder --decode auto uses for these inputs ("vulkan", "ffmpeg" or "opencv") and why
struct DecoderChoice {
    std::string decoder;
    std::string reason;
};
//! vulkan_available: Vulkan Video decoding may be used (Vulkan build, not --no-vk)
//! prefer_gpu: the GPU does more than the comparison (motion), use it whenever it can decode
DecoderChoice choose_decoder(const std::vector<std::string> & paths, bool vulkan_available, bool prefer_gpu);
#endif

#ifdef TRDROP_MEDIACODEC_VULKAN
//! Android: MediaCodec decoding into hardware buffers, analysis and motion video on the GPU with Vulkan.
//! throws std::runtime_error if the device or an input can't do that
std::unique_ptr<FramePipeline> make_mediacodec_vulkan_pipeline(const std::vector<std::string> & paths);
#endif

#if defined(USE_FFMPEG) && defined(USE_VULKAN)
//! FFmpeg Vulkan Video decoding, the Y, U and V planes stay on the GPU and are compared there.
//! throws std::runtime_error if Vulkan decoding isn't available for an input
std::unique_ptr<FramePipeline> make_vulkan_video_pipeline(const std::vector<std::string> & paths, int device_index);
#endif

} // namespace trdrop
