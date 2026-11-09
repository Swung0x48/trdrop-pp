#pragma once

#include <string>
#include <vector>

namespace trdrop {

//! the original supports comparing up to three videos in parallel
constexpr int kMaxVideoCount = 3;

//! per video analysis options (pixel difference from FramerateOptions, dismiss percentage from TearOptions)
struct VideoOptions {
    //! grayscale difference (0 - 255) a pixel may have without counting as "different"
    int pixel_difference = 0;
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
};

} // namespace trdrop
