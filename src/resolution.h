#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <iosfwd>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "motion.h"

namespace trdrop {

//! Render resolution of an upscaled picture (--resolution), from two traces an upscaler leaves:
//!
//! Pixel grid. A picture that was scaled up has a second difference (p[x-1] - 2 p[x] + p[x+1]) whose size depends
//! on where the pixel lies between two source pixels, and that repeats with the source sampling rate: the mean
//! |second difference| per column (per row) has a peak in its spectrum at r cycles per pixel for a picture
//! rendered at r times the output resolution, folded into 0 - 0.5 (r and 1 - r give the same peak). Video encoders
//! leave peaks of their own at multiples of 1 / 32, so 25% and 75% can't be told from them; a scale of 50% puts its
//! peak at 0.5, where the encoders of the recordings tried left none. A grid that only one axis shows is a weaker
//! trace: a regular pattern in the picture (a fence, a HUD) gives that too, a pattern that moves with the camera
//! only in a part of the video.
//!
//! Picture entering at the edge of a pan. A temporal upscaler (DLSS, FSR 2+, XeSS, TSR) builds its detail from
//! many frames. While the camera pans, the picture that enters at one edge has no frames before it and holds
//! what a single rendered frame holds: nothing finer than the render resolution, half the render sampling rate.
//! The vertical spectrum of the columns next to that edge is compared with the same columns of the other edge,
//! where the picture leaves after it crossed the screen: where the contrast of the entering picture falls
//! tells how fine a single frame resolves. Vertical, because the motion blur of a horizontal pan is horizontal.
//! That held for DLSS at 33% (two recordings); with TSR at 72%, FSR 3, DLSS 4 and 4.5 at 50% and others the
//! entering picture had as much contrast as the leaving one or more, and the pan says nothing.
//!
//! Stair steps. An upscaler that shows what has no history as it was rendered shows the stair steps of the
//! rendering there: an edge that is nearly level climbs one rendered pixel at a time, 1 / r output pixels. In the
//! same strip of entering columns the row of the strongest step from one row to the next is taken per column, in
//! bands of rows; where it forms plateaus over the columns, the heights between neighbouring plateaus are counted,
//! and the same in the leaving strip. Steps of 1 pixel say nothing (a smooth edge has them); steps of 2 and more
//! in the entering picture and not in the leaving one give 1 / r directly, with no part for the encoder - in whole
//! pixels, so that 50% to about 60% all read 50%.
//!
//! All are measured per frame - on the CPU (measure_resolution) or by the pipeline on the GPU, with the same
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
//! stair steps: per edge column the strongest step between two rows in bands of kResolutionStairBand rows, every
//! kResolutionStairHop rows; heights of 1 ... kResolutionStairHeights pixels are counted (the last: or more)
constexpr int kResolutionStairBand = 48;
constexpr int kResolutionStairHop = 24;
constexpr int kResolutionStairHeights = 8;

//! which part of a frame is measured
struct ResolutionLayout {
    //! 0: the frame is too small
    int width = 0;
    int height = 0;
    //! the pixel grid uses the rows [0, lattice_rows): not the bottom of the picture, where overlays tend to be
    int lattice_rows = 0;
    //! columns per edge, and the rows of their spectra: `segments` segments starting at row0 (0: too few rows)
    int edge = 0;
    int row0 = 0;
    int segments = 0;
    //! bands of rows for the stair steps, within the rows of the segments (0: too few rows)
    int stair_bands = 0;
    //! ... as they were asked for, percent of the height
    double rows_from = 0.0;
    double rows_to = 0.0;

    bool operator==(const ResolutionLayout & other) const
    {
        return width == other.width && height == other.height && lattice_rows == other.lattice_rows && edge == other.edge
            && row0 == other.row0 && segments == other.segments && stair_bands == other.stair_bands;
    }
};
//! rows_from, rows_to: the part of the height (percent, from the top) whose edge columns are measured
//! (VideoOptions::resolution_rows_from / _to)
ResolutionLayout resolution_layout(int width, int height, double rows_from, double rows_to);

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
    //! stair_bands values per column, the columns as in edge_power: (row of the strongest step in the band << 8)
    //! | its size (0 - 255), the step from the row to the one below, the first of equal ones
    std::vector<uint16_t> edge_stairs;
};

//! measures a frame on the CPU. luma: 8-bit single channel, layout: the one of its size. edges: also the spectra
//! of the edge columns (they are only used for frames of a pan, see resolution_pan)
void measure_resolution(const cv::Mat & luma, const ResolutionLayout & layout, bool edges, ResolutionFrame & out);

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

//! peaks of the pixel grid that are reported
constexpr int kResolutionPeaks = 4;

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

        //! a peak of the pixel grid, off the block grid of video encoders
        struct Peak {
            //! both axes have it, or only the columns (the horizontal grid) / only the rows
            enum Axes { Both, Columns, Rows } axes = Both;
            //! cycles per pixel
            double frequency = 0.0;
            //! how many times the background of the spectrum around it the peak is, in the columns / rows
            double strength_x = 0.0;
            double strength_y = 0.0;
            //! the share of the pieces of the video (kLatticeChunk frames each) whose own spectrum has the peak, in
            //! the columns / rows (-1: the video is too short for that). A grid is there whatever the picture shows
            double share_x = -1.0;
            double share_y = -1.0;
        };
        //! the peaks that count, at most kResolutionPeaks: those of both axes first, the strongest first
        std::vector<Peak> peaks;
        //! the strongest peak both axes have at the same frequency, whether it counts (lattice) or not
        bool lattice = false;
        double lattice_frequency = 0.0;
        double lattice_strength_x = 0.0;
        double lattice_strength_y = 0.0;
        //! the strongest peak on the block grid of video encoders below 0.5 (both axes): it says nothing either way
        double codec_frequency = 0.0;
        double codec_strength = 0.0;
        //! ... and it is strong enough to hide a grid there
        bool codec_peak = false;
        //! the peak at 0.5 cycles per pixel (a render scale of 50%; 25% and 75% as their second harmonic) and the
        //! encoder's at 0.25, in the columns / rows. half_peak: it counts - both axes have it, and it is not small
        //! against the one at 0.25
        double half_strength_x = 0.0;
        double half_strength_y = 0.0;
        double quarter_strength_x = 0.0;
        double quarter_strength_y = 0.0;
        //! both axes have a peak at 0.5
        bool half_present = false;
        bool half_peak = false;

        // picture entering at the edge of a pan
        //! the part of the height that was measured, percent
        double rows_from = 0.0;
        double rows_to = 0.0;
        //! frames used, picture moving to the right / to the left
        long long pan_frames_right = 0;
        long long pan_frames_left = 0;
        //! per age: contrast of the entering picture relative to the leaving one per bin (1 at the first bin),
        //! empty if no frame had columns of that age
        std::array<std::vector<double>, kResolutionAges> contrast;
        //! of the newest columns: the frequency (cycles per pixel) at which the contrast is down to one half
        //! (0: it never is), the lowest and the highest contrast and their frequencies
        double half_contrast_frequency = 0.0;
        double lowest_contrast = 1.0;
        double lowest_contrast_frequency = 0.0;
        double highest_contrast = 1.0;
        double highest_contrast_frequency = 0.0;
        // stair steps of nearly level edges, in the strip that entered within about 3 frames and in the same strip
        // of the leaving edge
        //! [h]: steps of h pixels (the last: or more); [0] is not used
        std::array<long long, kResolutionStairHeights + 1> stairs_entering {};
        std::array<long long, kResolutionStairHeights + 1> stairs_leaving {};
        long long stair_steps_entering = 0;
        long long stair_steps_leaving = 0;
        //! share of the steps that are 2 pixels or higher
        double stair_high_entering = 0.0;
        double stair_high_leaving = 0.0;
        //! mean height of the entering steps of 2 pixels and more (the open last class left out; 0: none)
        double stair_height = 0.0;
        //! the steps tell the render scale: enough of them, most of them high, and not so in the leaving picture
        bool stairs = false;
        //! ... 1 / stair_height
        double stair_scale = 0.0;

        //! an upward fan: the entering picture has more contrast than the leaving one and loses it with its age, in
        //! both directions of the pan. A temporal upscaler that shows the pixels without history as they were
        //! rendered - aliased - where DLSS at 33% smoothed them (TSR, FSR 2.1, FSR 3, "NXSR", DLSS 4.5 at 50% did)
        bool fan_up = false;
        //! the contrast of the oldest columns measured at that frequency (0: none), and their age
        double recovered_contrast = 0.0;
        int recovered_age = -1;

        //! a render scale the traces fit
        struct Candidate {
            //! render resolution / output resolution
            double scale = 0.0;
            //! how much speaks for it. These are levels of a rule, not probabilities:
            //! High: a grid in both axes, and the pan picks this one of its two scales (the other one is 1.2 times
            //! further from what the pan reads) or the stair steps of the entering picture fit it
            //! Medium: a grid in both axes; nothing tells its two scales apart, or the fan is an upward one and this
            //! is the larger scale (the true one in the recordings tried). The peak at 0.5 with stair steps of 2
            //! Low: the scale of a grid the pan or the stair steps speak against, the smaller scale of a grid with
            //! an upward fan, a second grid, a grid in one axis only, the peak at 0.5, the stair steps or the pan
            //! alone
            enum Confidence { Low, Medium, High } confidence = Low;
            enum Evidence { Grid, GridAndPan, GridOtherScale, GridAndStairs, GridAgainstStairs, GridAndFan, GridAgainstFan, GridOneAxis,
                            HalfPeak, HalfAndStairs, Stairs, Pan } evidence = Grid;
            //! index into peaks (-1: none)
            int peak = -1;
        };
        //! the most likely first; empty: no trace of upscaling
        std::vector<Candidate> candidates;
    };

    //! next frame: what was measured of it (nullptr: nothing) and its motion
    void push(const ResolutionFrame * frame, const MotionField & field);
    Summary summary() const;
    //! what the summary is made of, as CSV (--resolution-data): the spectra of the pixel grid and of the picture
    //! entering / leaving in a pan. Lines starting with # describe the video, then curve,direction,age,frequency,value
    void write_data(std::ostream & out, const std::string & video) const;

    //! frequency of a bin of the edge spectra, cycles per pixel
    static double bin_frequency(int bin) { return static_cast<double>(kResolutionFirstBin + bin) / kResolutionSegment; }

private:
    //! the pixel grid of one axis
    struct LatticeAxis {
        //! summed magnitude spectra of the detrended profiles: of the video, and of the piece that is being filled
        std::vector<double> spectrum;
        std::vector<double> chunk;
        //! per bin: the pieces whose spectrum has a peak there
        std::vector<uint32_t> hits;
        std::vector<float> window;
    };
    void _add_lattice(const ResolutionFrame & frame);
    void _add_profile(const std::vector<uint32_t> & sums, LatticeAxis & axis);
    static void _close_chunk(LatticeAxis & axis);
    double _share(const LatticeAxis & axis, double bin) const;

    ResolutionLayout _layout;
    long long _frames = 0;
    LatticeAxis _grid_x;
    LatticeAxis _grid_y;
    long long _lattice_frames = 0;
    long long _chunks = 0;
    cv::Mat _profile;
    cv::Mat _transformed;
    //! pans of the last frames, the newest first; NaN: unknown
    std::deque<float> _pans;
    //! summed power per direction (0: to the right), age and bin: entering / leaving edge
    std::array<std::array<std::array<double, kResolutionBins>, kResolutionAges>, 2> _entering {};
    std::array<std::array<std::array<double, kResolutionBins>, kResolutionAges>, 2> _leaving {};
    std::array<std::array<long long, kResolutionAges>, 2> _columns {};
    std::array<long long, 2> _pan_frames {};
    //! stair steps by height: of the entering strip, of the leaving strip
    std::array<std::array<long long, kResolutionStairHeights + 1>, 2> _stairs {};
};

} // namespace trdrop
