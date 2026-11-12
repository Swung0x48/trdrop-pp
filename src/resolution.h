#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <vector>

#include <opencv2/core.hpp>

#include "motion.h"

namespace trdrop {

//! Render resolution of an upscaled picture (--resolution), from two traces an upscaler leaves:
//!
//! Pixel grid. A picture that was scaled up has a second difference (p[x-1] - 2 p[x] + p[x+1]) whose size depends
//! on where the pixel lies between two source pixels, and that repeats with the source sampling rate: the mean
//! |second difference| per column (per row) has a peak in its spectrum at r cycles per pixel for a picture
//! rendered at r times the output resolution, folded into 0 - 0.5 (r and 1 - r give the same peak; an exact 0.5
//! gives none, both pixel positions are alike). Video encoders leave peaks of their own at multiples of 1 / 32.
//!
//! Picture entering at the edge of a pan. A temporal upscaler (DLSS, FSR 2+, XeSS, TSR) builds its detail from
//! many frames. While the camera pans, the picture that enters at one edge has no frames before it and holds
//! what a single rendered frame holds: nothing finer than the render resolution, half the render sampling rate.
//! The vertical spectrum of the columns next to that edge is compared with the same columns of the other edge,
//! where the picture leaves after it crossed the screen: where the contrast of the entering picture falls
//! tells how fine a single frame resolves. Vertical, because the motion blur of a horizontal pan is horizontal.
//!
//! Both are measured per frame - on the CPU (measure_resolution) or by the pipeline on the GPU, with the same
//! numbers - and put together by ResolutionAnalyzer.

//! the vertical spectrum of an edge column: mean power of segments of kResolutionSegment rows (Hann window, every
//! kResolutionHop rows), bins kResolutionFirstBin ... of the segment's DFT: bin k is k / kResolutionSegment cycles
//! per pixel, 0.047 - 0.469
constexpr int kResolutionSegment = 64;
constexpr int kResolutionHop = 32;
constexpr int kResolutionFirstBin = 3;
constexpr int kResolutionBins = 28;
//! columns next to the left and to the right edge that are measured (at most 1 / 8 of the width)
constexpr int kResolutionEdgeColumns = 256;

//! which part of a frame is measured
struct ResolutionLayout {
    //! 0: the frame is too small
    int width = 0;
    int height = 0;
    //! the pixel grid uses the rows [0, lattice_rows): not the bottom of the picture, where overlays tend to be
    int lattice_rows = 0;
    //! columns per edge, and the rows of their spectra: `segments` segments starting at row0
    int edge = 0;
    int row0 = 0;
    int segments = 0;

    bool operator==(const ResolutionLayout & other) const
    {
        return width == other.width && height == other.height && lattice_rows == other.lattice_rows && edge == other.edge
            && row0 == other.row0 && segments == other.segments;
    }
};
ResolutionLayout resolution_layout(int width, int height);

//! what is measured of one frame
struct ResolutionFrame {
    ResolutionLayout layout;
    //! per column x: sum over the rows of |p[x-1] - 2 p[x] + p[x+1]| (0 for the first and the last column)
    std::vector<uint32_t> column_sums;
    //! per row y < lattice_rows: sum over the columns of |p[y-1] - 2 p[y] + p[y+1]| (0 for the first and the last)
    std::vector<uint32_t> row_sums;
    //! the spectra of the edge columns were measured
    bool edges = false;
    //! kResolutionBins values per column: the columns 0 ... edge - 1 from the left edge, then those from the
    //! right edge (the outermost column first)
    std::vector<float> edge_power;
};

//! measures a frame on the CPU. luma: 8-bit single channel. edges: also the spectra of the edge columns
//! (they are only used for frames of a pan, see resolution_pan)
void measure_resolution(const cv::Mat & luma, bool edges, ResolutionFrame & out);

//! the horizontal pan of a frame, from its motion field
struct ResolutionPan {
    //! the motion could be measured and the cells agree on it (a picture that stands still is valid, with 0)
    bool valid = false;
    //! how far the picture moved since the previous frame next to the edge it enters at, in pixels
    //! (> 0: to the right, it enters at the left edge), and the vertical motion of the whole picture
    float dx = 0.0f;
    float dy = 0.0f;
};
ResolutionPan resolution_pan(const MotionField & field, int width, int height);
//! the edge columns of a frame with this pan may be used: only then their spectra are needed
bool resolution_pan_usable(const ResolutionPan & pan);

//! how long ago a group of edge columns entered the picture, in recorded frames
constexpr int kResolutionAges = 6;
//! the outer half of what entered with the last frame, its inner half, 1 frame ago, 2 - 3, 4 - 7, 8 - 15
extern const char * const kResolutionAgeNames[kResolutionAges];

//! Collects the frames of a video and estimates the render resolution.
class ResolutionAnalyzer {
public:
    struct Summary {
        int width = 0;
        int height = 0;
        long long frames = 0;

        // pixel grid: the strongest peak both axes have at the same frequency
        bool lattice = false;
        //! cycles per pixel, and how many times the background of the spectrum around it the peak is (x, y)
        double lattice_frequency = 0.0;
        double lattice_strength_x = 0.0;
        double lattice_strength_y = 0.0;
        //! the strongest peak lies on the block grid of video encoders: it says nothing either way
        bool lattice_on_codec_grid = false;
        double codec_frequency = 0.0;
        double codec_strength = 0.0;

        // picture entering at the edge of a pan
        //! frames used, picture moving to the right / to the left
        long long pan_frames_right = 0;
        long long pan_frames_left = 0;
        //! per age: contrast of the entering picture relative to the leaving one per bin (1 at the first bin),
        //! empty if no frame had columns of that age
        std::array<std::vector<double>, kResolutionAges> contrast;
        //! of the newest columns: the frequency (cycles per pixel) at which the contrast is down to one half
        //! (0: it never is), the lowest contrast and its frequency
        double half_contrast_frequency = 0.0;
        double lowest_contrast = 1.0;
        double lowest_contrast_frequency = 0.0;
        //! the contrast of the oldest columns measured at that frequency (0: none), and their age
        double recovered_contrast = 0.0;
        int recovered_age = -1;

        //! the estimate: render resolution / output resolution (0: none), and how it was found
        double scale = 0.0;
        //! the other scale with the same pixel grid (0: none)
        double other_scale = 0.0;
        enum Source { None, Lattice, LatticeAndPan, Pan } source = None;
    };

    //! next frame: what was measured of it (nullptr: nothing) and its motion
    void push(const ResolutionFrame * frame, const MotionField & field);
    Summary summary() const;

    //! frequency of a bin of the edge spectra, cycles per pixel
    static double bin_frequency(int bin) { return static_cast<double>(kResolutionFirstBin + bin) / kResolutionSegment; }

private:
    void _add_lattice(const ResolutionFrame & frame);

    ResolutionLayout _layout;
    long long _frames = 0;
    //! summed magnitude spectra of the detrended column / row profiles
    std::vector<double> _spectrum_x;
    std::vector<double> _spectrum_y;
    long long _lattice_frames = 0;
    cv::Mat _profile;
    cv::Mat _transformed;
    std::vector<float> _window_x;
    std::vector<float> _window_y;
    //! pans of the last frames, the newest first; NaN: unknown
    std::deque<float> _pans;
    //! summed power per direction (0: to the right), age and bin: entering / leaving edge
    std::array<std::array<std::array<double, kResolutionBins>, kResolutionAges>, 2> _entering {};
    std::array<std::array<std::array<double, kResolutionBins>, kResolutionAges>, 2> _leaving {};
    std::array<std::array<long long, kResolutionAges>, 2> _columns {};
    std::array<long long, 2> _pan_frames {};
};

} // namespace trdrop
