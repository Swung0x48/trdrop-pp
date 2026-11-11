#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace trdrop {

//! the original supports comparing up to three videos in parallel
constexpr int kMaxVideoCount = 3;

//! defaults for lossy codecs (H.264, HEVC, ...), whose duplicate frames differ by compression noise
constexpr double kLossyTolerancePct = 0.5;
constexpr int kLossyPixelDifference = 4;
//! automatic tolerance: additionally this share of what recently changed in a new frame (90th percentile of the
//! changed pixels per frame over the last 2 seconds). Encoders predict duplicate frames partly from the next
//! frame, which leaks a few percent of it into them
constexpr double kAutoToleranceRelative = 0.05;
//! automatic tolerance: kLossyTolerancePct is what the duplicates of a poor recording need. Where the recording is
//! cleaner the minimum for frames follows its noise, so that small changes (a static camera with a few distant
//! objects, a menu) still count: kAutoToleranceNoiseFactor times what the recent duplicates changed, between
//! kAutoToleranceMinPct and kLossyTolerancePct. "What they changed" is the value a third of them stay below - frames
//! with a small change count as duplicates until the minimum is lower, and up to two thirds of them don't move it -,
//! taken from the last 2 seconds of duplicate frames once there are kAutoToleranceNoiseSamples of them.
//! Measured on a 1080p240 capture and five re-encodes of it: that third is 0.0001 - 0.001% of the picture (99% of
//! the duplicates: below 0.02%) with a constant quality at 19 - 37 Mbit/s, 0.004 - 0.016% (99%: 0.3 - 0.8%) at 2 - 6
//! Mbit/s. Not for keyframes, which encoders code anew: a duplicate changes 0.1 - 10% of the picture there.
//! Where the minimum is lower, kAutoToleranceRelative looks at the last second only
constexpr double kAutoToleranceNoiseFactor = 45.0;
constexpr double kAutoToleranceMinPct = 0.05;
constexpr size_t kAutoToleranceNoiseSamples = 30;

//! how a frame is coded, as far as the decoder tells
enum class FrameCoding {
    Unknown,
    //! a keyframe / intra coded picture
    Intra,
    //! predicted from other frames
    Predicted,
};

//! per video analysis options (pixel difference from FramerateOptions, dismiss percentage from TearOptions)
struct VideoOptions {
    //! grayscale difference (0 - 255) a pixel may have without counting as "different"
    int pixel_difference = 0;
    //! --pixel-diff was given (otherwise a default for lossy codecs may be chosen)
    bool pixel_difference_set = false;
    //! percentage (0 - 100) of the pixels of a frame / row that may be different without the frame / row
    //! counting as changed, for lossy codecs. < 0: automatic (0 for lossless codecs; lossy codecs:
    //! kLossyTolerancePct for rows and as the minimum for frames, see tolerance_auto)
    double tolerance_pct = -1.0;
    //! tolerance_pct is only the minimum, the tolerance follows the content (see kAutoToleranceRelative)
    bool tolerance_auto = false;
    //! tears smaller than this percentage (0 - 100) of the frame height are dismissed
    double dismiss_tear_percentage = 10.0;
    //! overrides the recorded framerate of the source (0 = read from the container)
    double source_fps_override = 0.0;
};

struct Options {
    std::vector<std::string> inputs;
    std::vector<VideoOptions> videos; // same size as inputs
    //! per frame data, "-" = stdout
    std::string output_path = "-";
    //! optional file with one row per detected tear (empty = disabled)
    std::string tears_path;
    //! stop after this many frames (0 = until the shortest video ends)
    long long max_frames = 0;
    bool quiet = false;
    bool print_summary = true;
    //! use the Vulkan backend (only in builds with USE_VULKAN), false: OpenCV on the CPU
    bool use_vulkan = true;
    //! index of the Vulkan device (-1 = automatic: first discrete GPU)
    int vk_device = -1;
    //! decoder: auto, vulkan (FFmpeg Vulkan Video, luma on the GPU), ffmpeg (FFmpeg on the CPU, luma), opencv
    std::string decode = "auto";
    //! measure the motion between the frames: animation step, animation error, perceived smoothness
    bool motion = false;
    //! length of the window the motion of a frame is compared to, 0: automatic
    //! (400 ms, longer if the framerate is low: until it holds 10 new frames, at most 1 second)
    double motion_window_ms = 0.0;
    //! the capture didn't run in step with what it recorded: a picture may have been recorded up to one
    //! interval late and pictures may be missing (see MotionAnalyzer)
    bool async_capture = false;
    //! per input video: file for a copy of the video with its motion drawn on it (empty: none)
    std::vector<std::string> motion_videos;
    //! the arrows of the motion video are drawn this many times longer than the motion
    double arrow_scale = 1.0;
    //! called while the videos are analyzed (frames read, progress 0 - 1 or < 0 if unknown), the analysis
    //! stops with an error if it returns false. For programs that embed the analysis (trdrop_library.h)
    std::function<bool(long long, double)> progress;
};

} // namespace trdrop
