#pragma once

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
};

} // namespace trdrop
