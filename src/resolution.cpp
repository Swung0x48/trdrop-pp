#include "resolution.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <mutex>
#include <ostream>

namespace trdrop {

const char * const kResolutionAgeNames[kResolutionAges] = {
    "entered with this frame, outer half", "entered with this frame, inner half", "1 frame ago", "2 - 3 frames ago",
    "4 - 7 frames ago", "8 - 15 frames ago",
};

namespace {

//! a pan has to move the picture by at least this many pixels per frame - half of it are the newest columns - and
//! at most this far: the newest columns have to lie within the measured ones, and the tracking has to follow
constexpr float kMinPan = 4.0f;
constexpr float kMaxPan = 64.0f;
//! the picture leaving at the other edge is the reference. It has been on the screen for a while if the pan has
//! kept its direction: no frame of the last kPanRun moved the other way (by kPanMoved pixels or more), and
//! kPanRunMoving of them moved this way
constexpr size_t kPanRun = 20;
constexpr size_t kPanRunMoving = 15;
constexpr float kPanMoved = 2.0f;
//! cells with motion a frame needs for its pan, and the share of them that has to agree with it
constexpr size_t kPanCells = 24;
constexpr double kPanAgreeing = 0.6;
//! of the grid's columns, those next to an edge that tell how fast the picture moves there. A camera that turns
//! moves the picture faster at the edges than in the middle
constexpr int kPanEdgeCells = 4;
constexpr size_t kPanMinEdgeCells = 6;

//! pixel grid: the profiles lose their slow variation (mean over this many values), peaks are compared with the
//! median of the spectrum this many bins to both sides
constexpr int kLatticeDetrend = 31;
constexpr int kLatticeBackground = 40;
//! frequencies below this are the content
constexpr double kLatticeMinFrequency = 0.06;
//! a peak counts if both axes have it this many times above the background
constexpr double kLatticeMinStrength = 3.0;
//! the video in pieces of this many frames: a piece has a peak where its own spectrum is this many times its
//! background. A pixel grid is there whatever the picture shows, a pattern in the picture is not
constexpr long long kLatticeChunk = 60;
constexpr long long kLatticeMinChunks = 4;
constexpr double kLatticeChunkStrength = 2.0;
//! a peak that only one axis has counts with this strength if it stands this many times the spread of that axis'
//! spectrum above it (a short video has a restless spectrum), if this share of the pieces has it, and if it is
//! this far (cycles per pixel) from the block grid of video encoders, whose peaks have skirts
constexpr double kOneAxisMinStrength = 2.0;
constexpr double kOneAxisMinSpreads = 10.0;
constexpr double kOneAxisMinShare = 0.5;
constexpr double kOneAxisCodecGap = 0.003;
//! peaks nearer to each other than this (cycles per pixel) are one peak
constexpr double kPeakApart = 0.006;
//! the pan picks one of the two scales of a grid if the other one is this much further from what the pan reads
//! (difference of the logarithms of the two ratios: a factor of 1.2)
constexpr double kPanMinMargin = 0.182;
//! the peak at 0.5 counts if it is at least this share of the encoder's peak at 0.25, in both axes. Encoders alone
//! left 0.15 - 0.4 of it there (the peak at 0.5 is their weakest), a render scale of 50% 0.54 and more
constexpr double kHalfPeakShare = 0.4;
//! stair steps: the strip is what entered in this many frames, at least kStairMinStrip columns; a band of rows
//! shows an edge if the strongest steps of its columns have this median size, lie this far inside the band and
//! differ by 3 rows at most between this share of neighbouring columns; a plateau is this many columns on one
//! row, and two plateaus are neighbours with at most kStairGap columns between them
constexpr float kStairFrames = 3.0f;
constexpr int kStairMinStrip = 32;
constexpr int kStairMinStep = 10;
constexpr int kStairMargin = 3;
constexpr double kStairOneEdge = 0.95;
constexpr int kStairPlateau = 5;
constexpr int kStairGap = 3;
//! the steps tell the scale with this many of them in the entering strip, this share of them 2 pixels or higher,
//! and at most kStairSettled of those of the leaving strip (which needs kStairMinSettledSteps to be judged).
//! They fit a scale if 1 / their mean height is within this of it (difference of the logarithms: a factor of 1.4);
//! of the two scales of a grid the one that is nearer by kPanMinMargin is picked
constexpr long long kStairMinSteps = 100;
constexpr long long kStairMinSettledSteps = 30;
constexpr double kStairHighShare = 0.6;
constexpr double kStairSettled = 0.3;
constexpr double kStairFit = 0.35;
//! the peak at 0.5 and steps of 2 pixels mean the same scale within this (difference of the logarithms)
constexpr double kStairHalf = 0.2;

//! an upward fan: the newest columns have this much contrast against the leaving picture at some frequency, and
//! the oldest columns measured (2 - 3 frames old at least) have lost this share of what is above 1 there. Both
//! directions of the pan have to show it, a pan in one direction compares two parts of the scene
constexpr double kFanMinContrast = 1.25;
constexpr double kFanSettled = 0.5;
//! video encoders work in blocks of 4 - 64 pixels: peaks at multiples of 1 / 32 are theirs
constexpr double kCodecGrid = 32.0;
//! frames of a pan needed for its result
constexpr long long kMinPanFrames = 30;

float median_of(std::vector<float> & values)
{
    if (values.empty()) return 0.0f;
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle), values.end());
    return values[middle];
}

//! cos / sin of the bins of a segment with the Hann window folded in: [bin][n]
struct SegmentBasis {
    float re[kResolutionBins][kResolutionSegment];
    float im[kResolutionBins][kResolutionSegment];
    SegmentBasis()
    {
        const double pi = 3.14159265358979323846;
        for (int b = 0; b < kResolutionBins; ++b)
        {
            for (int n = 0; n < kResolutionSegment; ++n)
            {
                // the table the shader uses: cos / sin of 2 pi j / segment, j = (bin * n) mod segment
                const int j = ((kResolutionFirstBin + b) * n) % kResolutionSegment;
                const float window = 0.5f - 0.5f * static_cast<float>(std::cos(2.0 * pi * n / kResolutionSegment));
                re[b][n] = window * static_cast<float>(std::cos(2.0 * pi * j / kResolutionSegment));
                im[b][n] = -window * static_cast<float>(std::sin(2.0 * pi * j / kResolutionSegment));
            }
        }
    }
};

const SegmentBasis & segment_basis()
{
    static const SegmentBasis basis;
    return basis;
}

} // namespace

ResolutionLayout resolution_layout(int width, int height, double rows_from, double rows_to)
{
    ResolutionLayout layout;
    if (width < 8 * kResolutionSegment || height < 4 * kResolutionSegment) return layout;
    layout.width = width;
    layout.height = height;
    layout.lattice_rows = height * 88 / 100;
    layout.edge = std::min(kResolutionEdgeColumns, width / 8);
    layout.rows_from = rows_from;
    layout.rows_to = rows_to;
    layout.row0 = std::min(height, std::max(0, static_cast<int>(std::lround(height * rows_from / 100.0))));
    const int row1 = std::min(height, std::max(0, static_cast<int>(std::lround(height * rows_to / 100.0))));
    // a part that is too short for one segment: no spectra
    if (row1 - layout.row0 >= kResolutionSegment) layout.segments = (row1 - layout.row0 - kResolutionSegment) / kResolutionHop + 1;
    // the stair steps use the rows of the segments; a band reads one row below its last
    const int rows = layout.segments >= 1 ? (layout.segments - 1) * kResolutionHop + kResolutionSegment : 0;
    if (rows > kResolutionStairBand) layout.stair_bands = (rows - kResolutionStairBand - 1) / kResolutionStairHop + 1;
    return layout;
}

void measure_resolution(const cv::Mat & luma, const ResolutionLayout & layout, bool edges, ResolutionFrame & out)
{
    CV_Assert(luma.type() == CV_8UC1);
    CV_Assert(layout.width == 0 || (layout.width == luma.cols && layout.height == luma.rows));
    out.layout = layout;
    out.edges = false;
    out.column_sums.clear();
    out.row_sums.clear();
    out.edge_power.clear();
    out.edge_stairs.clear();
    if (layout.width == 0) return;
    const int width = layout.width;
    const int rows = layout.lattice_rows;
    out.column_sums.assign(static_cast<size_t>(width), 0);
    out.row_sums.assign(static_cast<size_t>(rows), 0);

    // sums of integers: the order doesn't matter, the bands of rows run in parallel
    constexpr int kBandRows = 64;
    const int bands = (rows + kBandRows - 1) / kBandRows;
    std::mutex merge;
    cv::parallel_for_(cv::Range(0, bands), [&](const cv::Range & range) {
        std::vector<uint32_t> columns(static_cast<size_t>(width), 0);
        for (int band = range.start; band < range.end; ++band)
        {
            const int y1 = std::min(rows, (band + 1) * kBandRows);
            for (int y = band * kBandRows; y < y1; ++y)
            {
                const uint8_t * p = luma.ptr<uint8_t>(y);
                uint32_t * c = columns.data();
                for (int x = 1; x < width - 1; ++x)
                {
                    c[x] += static_cast<uint32_t>(std::abs(static_cast<int>(p[x - 1]) - 2 * static_cast<int>(p[x]) + static_cast<int>(p[x + 1])));
                }
                if (y == 0 || y == rows - 1) continue;
                const uint8_t * up = luma.ptr<uint8_t>(y - 1);
                const uint8_t * down = luma.ptr<uint8_t>(y + 1);
                uint32_t sum = 0;
                for (int x = 0; x < width; ++x)
                {
                    sum += static_cast<uint32_t>(std::abs(static_cast<int>(up[x]) - 2 * static_cast<int>(p[x]) + static_cast<int>(down[x])));
                }
                out.row_sums[static_cast<size_t>(y)] = sum;
            }
        }
        const std::lock_guard<std::mutex> lock(merge);
        for (int x = 0; x < width; ++x) out.column_sums[static_cast<size_t>(x)] += columns[static_cast<size_t>(x)];
    });

    if (!edges || layout.segments < 1) return;
    const SegmentBasis & basis = segment_basis();
    const int edge = layout.edge;
    out.edge_power.assign(static_cast<size_t>(2 * edge) * kResolutionBins, 0.0f);
    cv::parallel_for_(cv::Range(0, 2 * edge), [&](const cv::Range & range) {
        float values[kResolutionSegment];
        for (int j = range.start; j < range.end; ++j)
        {
            const int x = j < edge ? j : width - 1 - (j - edge);
            float power[kResolutionBins] = {};
            for (int s = 0; s < layout.segments; ++s)
            {
                const int y0 = layout.row0 + s * kResolutionHop;
                for (int n = 0; n < kResolutionSegment; ++n) values[n] = static_cast<float>(luma.at<uint8_t>(y0 + n, x));
                for (int b = 0; b < kResolutionBins; ++b)
                {
                    float re = 0.0f, im = 0.0f;
                    for (int n = 0; n < kResolutionSegment; ++n)
                    {
                        re += values[n] * basis.re[b][n];
                        im += values[n] * basis.im[b][n];
                    }
                    power[b] += re * re + im * im;
                }
            }
            float * result = &out.edge_power[static_cast<size_t>(j) * kResolutionBins];
            for (int b = 0; b < kResolutionBins; ++b) result[b] = power[b] / static_cast<float>(layout.segments);
        }
    });
    // stair steps: the strongest step between two rows per band (resolution_stairs.comp does the same)
    const int stair_bands = layout.stair_bands;
    out.edge_stairs.assign(static_cast<size_t>(2 * edge) * static_cast<size_t>(stair_bands), 0);
    if (stair_bands > 0)
    {
        cv::parallel_for_(cv::Range(0, 2 * edge), [&](const cv::Range & range) {
            for (int j = range.start; j < range.end; ++j)
            {
                const int x = j < edge ? j : width - 1 - (j - edge);
                for (int band = 0; band < stair_bands; ++band)
                {
                    const int y0 = layout.row0 + band * kResolutionStairHop;
                    int strongest = 0, row = 0;
                    int above = luma.at<uint8_t>(y0, x);
                    for (int n = 0; n < kResolutionStairBand; ++n)
                    {
                        const int below = luma.at<uint8_t>(y0 + n + 1, x);
                        const int size = std::abs(above - below);
                        if (size > strongest)
                        {
                            strongest = size;
                            row = n;
                        }
                        above = below;
                    }
                    out.edge_stairs[static_cast<size_t>(j) * static_cast<size_t>(stair_bands) + static_cast<size_t>(band)] = static_cast<uint16_t>((row << 8) | strongest);
                }
            }
        });
    }
    out.edges = true;
}

ResolutionPan resolution_pan(const MotionField & field, int width, int height)
{
    ResolutionPan pan;
    if (!field.estimated) return pan;
    const MotionGrid grid = motion_grid(width, height);
    std::vector<float> dx, dy;
    for (int cell = 0; cell < kMotionCells; ++cell)
    {
        if (field.points[static_cast<size_t>(cell)] == 0) continue;
        dx.push_back(field.dx[static_cast<size_t>(cell)]);
        dy.push_back(field.dy[static_cast<size_t>(cell)]);
    }
    if (dx.size() < kPanCells) return pan;
    std::vector<float> scratch = dx;
    const float all_dx = median_of(scratch);
    scratch = dy;
    pan.dy = median_of(scratch);
    // the cells have to move like the picture: a turning camera moves the edges up to twice as fast as the middle
    const float tolerance = std::max(3.0f, 0.5f * std::abs(all_dx));
    size_t agreeing = 0;
    for (const float value : dx)
    {
        if (std::abs(value - all_dx) <= tolerance) ++agreeing;
    }
    if (static_cast<double>(agreeing) < kPanAgreeing * static_cast<double>(dx.size())) return pan;
    pan.valid = true;
    pan.dx = all_dx;
    if (std::abs(all_dx) < kPanMoved) return pan;
    // how fast the picture moves where it enters
    scratch.clear();
    for (int cy = 0; cy < grid.cells_y; ++cy)
    {
        for (int k = 0; k < kPanEdgeCells; ++k)
        {
            const int cx = all_dx > 0.0f ? k : grid.cells_x - 1 - k;
            const size_t cell = static_cast<size_t>(cy * grid.cells_x + cx);
            if (field.points[cell] == 0) continue;
            if (field.dx[cell] * all_dx <= 0.0f) continue;
            scratch.push_back(field.dx[cell]);
        }
    }
    if (scratch.size() >= kPanMinEdgeCells) pan.dx = median_of(scratch);
    return pan;
}

bool resolution_pan_usable(const ResolutionPan & pan)
{
    if (!pan.valid) return false;
    const float speed = std::abs(pan.dx);
    return speed >= kMinPan && speed <= kMaxPan && std::abs(pan.dy) <= std::max(2.0f, 0.15f * speed);
}

namespace {

//! every bin of a spectrum relative to the median of the bins around it
std::vector<double> relative_strengths(const std::vector<double> & spectrum)
{
    const int n = static_cast<int>(spectrum.size());
    std::vector<double> strength(spectrum.size(), 0.0);
    std::vector<double> window;
    for (int k = 0; k < n; ++k)
    {
        const int a = std::max(0, k - kLatticeBackground), b = std::min(n, k + kLatticeBackground + 1);
        window.assign(spectrum.begin() + a, spectrum.begin() + b);
        std::nth_element(window.begin(), window.begin() + static_cast<std::ptrdiff_t>(window.size() / 2), window.end());
        const double background = window[window.size() / 2];
        strength[static_cast<size_t>(k)] = background > 0.0 ? spectrum[static_cast<size_t>(k)] / background : 0.0;
    }
    return strength;
}

//! the strongest bin within `tolerance` cycles per pixel of a frequency; n: bin k is k / n cycles per pixel
double strongest_near(const std::vector<double> & strength, double n, double frequency, double tolerance)
{
    const int first = std::max(0, static_cast<int>(std::ceil((frequency - tolerance) * n)));
    const int last = std::min(static_cast<int>(strength.size()) - 1, static_cast<int>(std::floor((frequency + tolerance) * n)));
    double best = 0.0;
    for (int k = first; k <= last; ++k) best = std::max(best, strength[static_cast<size_t>(k)]);
    return best;
}

//! how far a frequency is from the block grid of video encoders, cycles per pixel
double block_grid_distance(double frequency)
{
    const double on_grid = frequency * kCodecGrid;
    return std::abs(on_grid - std::round(on_grid)) / kCodecGrid;
}

//! the middle and the spread (1.4826 x the median distance from the middle: the standard deviation of a normal
//! distribution) of the strengths between two frequencies, off the block grid
void middle_and_spread(const std::vector<double> & strength, double n, double from, double to, double & middle, double & spread)
{
    std::vector<double> values;
    for (size_t k = 0; k < strength.size(); ++k)
    {
        const double frequency = static_cast<double>(k) / n;
        if (frequency < from || frequency > to || block_grid_distance(frequency) <= kOneAxisCodecGap) continue;
        values.push_back(strength[k]);
    }
    middle = 1.0;
    spread = 0.0;
    if (values.empty()) return;
    const std::ptrdiff_t half = static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), values.begin() + half, values.end());
    middle = values[static_cast<size_t>(half)];
    for (double & value : values) value = std::abs(value - middle);
    std::nth_element(values.begin(), values.begin() + half, values.end());
    spread = 1.4826 * values[static_cast<size_t>(half)];
}

//! counts the stair steps of a strip of edge columns by their height. stairs: bands values per column, the
//! outermost column first; strip: columns from the edge
void count_stairs(const uint16_t * stairs, int bands, int strip, std::array<long long, kResolutionStairHeights + 1> & heights)
{
    std::vector<int> rows(static_cast<size_t>(strip));
    std::vector<int> sizes(static_cast<size_t>(strip));
    for (int band = 0; band < bands; ++band)
    {
        int lowest = kResolutionStairBand, highest = 0;
        for (int x = 0; x < strip; ++x)
        {
            const uint16_t value = stairs[static_cast<size_t>(x) * static_cast<size_t>(bands) + static_cast<size_t>(band)];
            rows[static_cast<size_t>(x)] = value >> 8;
            sizes[static_cast<size_t>(x)] = value & 0xFF;
            lowest = std::min(lowest, rows[static_cast<size_t>(x)]);
            highest = std::max(highest, rows[static_cast<size_t>(x)]);
        }
        // an edge that stays inside the band ...
        if (lowest < kStairMargin || highest > kResolutionStairBand - 1 - kStairMargin) continue;
        std::nth_element(sizes.begin(), sizes.begin() + strip / 2, sizes.end());
        if (sizes[static_cast<size_t>(strip / 2)] < kStairMinStep) continue;
        // ... and is one edge: its row changes little from column to column
        int smooth = 0;
        for (int x = 1; x < strip; ++x)
        {
            if (std::abs(rows[static_cast<size_t>(x)] - rows[static_cast<size_t>(x) - 1]) <= 3) ++smooth;
        }
        if (static_cast<double>(smooth) < kStairOneEdge * static_cast<double>(strip - 1)) continue;
        // plateaus, and the height between two that are neighbours
        int previous_end = -1, previous_row = 0, start = 0;
        for (int x = 1; x <= strip; ++x)
        {
            if (x < strip && rows[static_cast<size_t>(x)] == rows[static_cast<size_t>(start)]) continue;
            if (x - start >= kStairPlateau)
            {
                const int row = rows[static_cast<size_t>(start)];
                if (previous_end >= 0 && start - previous_end <= kStairGap && row != previous_row)
                {
                    heights[static_cast<size_t>(std::min(std::abs(row - previous_row), kResolutionStairHeights))] += 1;
                }
                previous_end = x;
                previous_row = row;
            }
            start = x;
        }
    }
}

//! the bin of a peak, between the bins: the top of the parabola through it and its neighbours
double refined_bin(const std::vector<double> & strength, size_t k)
{
    if (k == 0 || k + 1 >= strength.size()) return static_cast<double>(k);
    const double curvature = strength[k - 1] - 2.0 * strength[k] + strength[k + 1];
    if (!(curvature < 0.0)) return static_cast<double>(k);
    return static_cast<double>(k) + std::min(0.5, std::max(-0.5, 0.5 * (strength[k - 1] - strength[k + 1]) / curvature));
}

} // namespace

void ResolutionAnalyzer::_add_profile(const std::vector<uint32_t> & sums, LatticeAxis & axis)
{
    // the profile without its slow variation, windowed: magnitude spectrum
    const int n = static_cast<int>(sums.size());
    if (n < 4 * kLatticeDetrend) return;
    if (static_cast<int>(axis.window.size()) != n)
    {
        axis.window.resize(static_cast<size_t>(n));
        const double pi = 3.14159265358979323846;
        for (int i = 0; i < n; ++i) axis.window[static_cast<size_t>(i)] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * pi * i / (n - 1)));
        axis.spectrum.assign(static_cast<size_t>(n / 2 + 1), 0.0);
        axis.chunk.assign(static_cast<size_t>(n / 2 + 1), 0.0);
        axis.hits.assign(static_cast<size_t>(n / 2 + 1), 0);
    }
    _profile.create(1, n, CV_32FC1);
    float * profile = _profile.ptr<float>(0);
    // the first and the last value are not measured, they take their neighbour's
    std::vector<double> prefix(static_cast<size_t>(n) + 1, 0.0);
    for (int i = 0; i < n; ++i)
    {
        const int source = std::min(n - 2, std::max(1, i));
        prefix[static_cast<size_t>(i) + 1] = prefix[static_cast<size_t>(i)] + static_cast<double>(sums[static_cast<size_t>(source)]);
    }
    const int half = kLatticeDetrend / 2;
    for (int i = 0; i < n; ++i)
    {
        const int a = std::max(0, i - half), b = std::min(n, i + half + 1);
        const double mean = (prefix[static_cast<size_t>(b)] - prefix[static_cast<size_t>(a)]) / static_cast<double>(b - a);
        const double value = prefix[static_cast<size_t>(i) + 1] - prefix[static_cast<size_t>(i)];
        profile[i] = static_cast<float>(value - mean) * axis.window[static_cast<size_t>(i)];
    }
    cv::dft(_profile, _transformed, cv::DFT_COMPLEX_OUTPUT);
    const cv::Vec2f * bins = _transformed.ptr<cv::Vec2f>(0);
    for (size_t k = 0; k < axis.spectrum.size(); ++k)
    {
        const double magnitude = std::sqrt(static_cast<double>(bins[k][0]) * bins[k][0] + static_cast<double>(bins[k][1]) * bins[k][1]);
        axis.spectrum[k] += magnitude;
        axis.chunk[k] += magnitude;
    }
}

void ResolutionAnalyzer::_close_chunk(LatticeAxis & axis)
{
    if (axis.chunk.empty()) return;
    // a piece has a peak at a bin if the bin or one next to it stands out of the piece's own spectrum
    const std::vector<double> strength = relative_strengths(axis.chunk);
    const size_t n = strength.size();
    for (size_t k = 0; k < n; ++k)
    {
        const double peak = std::max(strength[k], std::max(k > 0 ? strength[k - 1] : 0.0, k + 1 < n ? strength[k + 1] : 0.0));
        if (peak >= kLatticeChunkStrength) axis.hits[k] += 1;
    }
    std::fill(axis.chunk.begin(), axis.chunk.end(), 0.0);
}

void ResolutionAnalyzer::_add_lattice(const ResolutionFrame & frame)
{
    _add_profile(frame.column_sums, _grid_x);
    _add_profile(frame.row_sums, _grid_y);
    ++_lattice_frames;
    if (_lattice_frames % kLatticeChunk == 0)
    {
        _close_chunk(_grid_x);
        _close_chunk(_grid_y);
        ++_chunks;
    }
}

double ResolutionAnalyzer::_share(const LatticeAxis & axis, double bin) const
{
    if (_chunks < kLatticeMinChunks || axis.hits.empty()) return -1.0;
    const size_t k = static_cast<size_t>(std::min(static_cast<double>(axis.hits.size() - 1), std::max(0.0, std::round(bin))));
    return static_cast<double>(axis.hits[k]) / static_cast<double>(_chunks);
}

void ResolutionAnalyzer::push(const ResolutionFrame * frame, const MotionField & field)
{
    ++_frames;
    const bool measured = frame && frame->layout.width > 0;
    if (measured && _layout.width == 0) _layout = frame->layout;
    // a frame of another size is not measured, and nothing is known about its pan
    const bool usable = measured && frame->layout == _layout;
    const ResolutionPan pan = usable ? resolution_pan(field, _layout.width, _layout.height) : ResolutionPan();
    _pans.push_front(pan.valid ? pan.dx : std::numeric_limits<float>::quiet_NaN());
    if (_pans.size() > 16 + kPanRun) _pans.pop_back();
    if (!usable) return;
    _add_lattice(*frame);

    if (!frame->edges || !resolution_pan_usable(pan) || _pans.size() < kPanRun) return;
    // the pan kept its direction
    size_t moving = 0;
    for (size_t k = 0; k < kPanRun; ++k)
    {
        const float value = _pans[k];
        if (std::isnan(value)) return;
        if (std::abs(value) < kPanMoved) continue;
        if ((value > 0.0f) != (pan.dx > 0.0f)) return;
        ++moving;
    }
    if (moving < kPanRunMoving) return;

    // when a column entered: the summed motion of the last frames reaches it. travelled[k]: the last k + 1 frames
    float travelled[16];
    float sum = 0.0f;
    for (size_t k = 0; k < 16; ++k)
    {
        const float value = k < _pans.size() ? _pans[k] : std::numeric_limits<float>::quiet_NaN();
        // before the frames that are known: unknown, the older groups are left out
        if (std::isnan(value)) sum = std::numeric_limits<float>::infinity();
        else sum += std::abs(value);
        travelled[k] = sum;
    }
    const float bounds[kResolutionAges + 1] = { 0.0f, 0.5f * travelled[0], travelled[0], travelled[1], travelled[3], travelled[7], travelled[15] };
    const size_t direction = pan.dx > 0.0f ? 0 : 1;
    const int edge = _layout.edge;
    // entering at the left edge if the picture moves to the right
    const float * entering = frame->edge_power.data() + (direction == 0 ? 0 : static_cast<size_t>(edge) * kResolutionBins);
    const float * leaving = frame->edge_power.data() + (direction == 0 ? static_cast<size_t>(edge) * kResolutionBins : 0);
    for (int age = 0; age < kResolutionAges; ++age)
    {
        if (!(bounds[age + 1] <= static_cast<float>(edge))) break;
        for (int column = 0; column < edge; ++column)
        {
            const float position = static_cast<float>(column) + 0.5f;
            if (position < bounds[age] || position >= bounds[age + 1]) continue;
            for (int b = 0; b < kResolutionBins; ++b)
            {
                _entering[direction][static_cast<size_t>(age)][static_cast<size_t>(b)] += entering[static_cast<size_t>(column) * kResolutionBins + static_cast<size_t>(b)];
                _leaving[direction][static_cast<size_t>(age)][static_cast<size_t>(b)] += leaving[static_cast<size_t>(column) * kResolutionBins + static_cast<size_t>(b)];
            }
            _columns[direction][static_cast<size_t>(age)] += 1;
        }
    }
    _pan_frames[direction] += 1;

    // stair steps: of what entered in the last frames, and of as many columns of the leaving edge
    const int bands = _layout.stair_bands;
    if (bands > 0 && frame->edge_stairs.size() == static_cast<size_t>(2 * edge) * static_cast<size_t>(bands))
    {
        const int strip = std::min(edge, std::max(kStairMinStrip, static_cast<int>(std::lround(kStairFrames * std::abs(pan.dx)))));
        const uint16_t * left = frame->edge_stairs.data();
        const uint16_t * right = left + static_cast<size_t>(edge) * static_cast<size_t>(bands);
        count_stairs(direction == 0 ? left : right, bands, strip, _stairs[0]);
        count_stairs(direction == 0 ? right : left, bands, strip, _stairs[1]);
    }
}

ResolutionAnalyzer::Summary ResolutionAnalyzer::summary() const
{
    Summary result;
    result.width = _layout.width;
    result.height = _layout.height;
    result.frames = _frames;
    result.rows_from = _layout.rows_from;
    result.rows_to = _layout.rows_to;

    // ---- pixel grid: peaks relative to the background around them
    if (_lattice_frames > 0 && !_grid_x.spectrum.empty() && !_grid_y.spectrum.empty())
    {
        const std::vector<double> sx = relative_strengths(_grid_x.spectrum), sy = relative_strengths(_grid_y.spectrum);
        // bin k of n / 2 + 1 bins: k / n cycles per pixel
        const double nx = static_cast<double>(_layout.width), ny = static_cast<double>(_layout.lattice_rows);
        const double tolerance = 1.5 / nx + 1.5 / ny;
        // 0.5 is looked at by itself
        const double below_half = 0.5 - 2.0 * tolerance;
        // what a peak of one axis has to reach
        double middle_x = 1.0, spread_x = 0.0, middle_y = 1.0, spread_y = 0.0;
        middle_and_spread(sx, nx, kLatticeMinFrequency, below_half, middle_x, spread_x);
        middle_and_spread(sy, ny, kLatticeMinFrequency, below_half, middle_y, spread_y);
        const double one_axis_x = std::max(kOneAxisMinStrength, middle_x + kOneAxisMinSpreads * spread_x);
        const double one_axis_y = std::max(kOneAxisMinStrength, middle_y + kOneAxisMinSpreads * spread_y);
        struct Found {
            Summary::Peak peak;
            double score;
        };
        std::vector<Found> found;
        double best = 0.0, best_codec = 0.0;
        for (size_t k = 0; k < sx.size(); ++k)
        {
            const double frequency = static_cast<double>(k) / nx;
            if (frequency < kLatticeMinFrequency || frequency > below_half) continue;
            const double rows = strongest_near(sy, ny, frequency, tolerance);
            const double both = std::min(sx[k], rows);
            const double off_grid = block_grid_distance(frequency);
            if (off_grid <= tolerance)
            {
                if (both > best_codec)
                {
                    best_codec = both;
                    result.codec_frequency = frequency;
                    result.codec_strength = both;
                }
                continue;
            }
            if (off_grid <= kOneAxisCodecGap)
            {
                // the skirt of an encoder's peak that is stronger
                const double grid = std::round(frequency * kCodecGrid) / kCodecGrid;
                if (std::min(strongest_near(sx, nx, grid, tolerance), strongest_near(sy, ny, grid, tolerance)) >= both) continue;
            }
            Summary::Peak peak;
            peak.frequency = refined_bin(sx, k) / nx;
            peak.strength_x = sx[k];
            peak.strength_y = rows;
            peak.share_x = _share(_grid_x, static_cast<double>(k));
            peak.share_y = _share(_grid_y, frequency * ny);
            if (both > best)
            {
                best = both;
                result.lattice_frequency = peak.frequency;
                result.lattice_strength_x = sx[k];
                result.lattice_strength_y = rows;
            }
            if (both >= kLatticeMinStrength)
            {
                peak.axes = Summary::Peak::Both;
                found.push_back({ peak, both });
            }
            else if (sx[k] >= one_axis_x && off_grid > kOneAxisCodecGap && peak.share_x >= kOneAxisMinShare)
            {
                peak.axes = Summary::Peak::Columns;
                found.push_back({ peak, sx[k] });
            }
        }
        for (size_t j = 0; j < sy.size(); ++j)
        {
            const double frequency = static_cast<double>(j) / ny;
            if (frequency < kLatticeMinFrequency || frequency > below_half) continue;
            if (sy[j] < one_axis_y || block_grid_distance(frequency) <= kOneAxisCodecGap) continue;
            const double columns = strongest_near(sx, nx, frequency, tolerance);
            if (std::min(sy[j], columns) >= kLatticeMinStrength) continue;        // a peak of both axes, found above
            Summary::Peak peak;
            peak.axes = Summary::Peak::Rows;
            peak.frequency = refined_bin(sy, j) / ny;
            peak.strength_x = columns;
            peak.strength_y = sy[j];
            peak.share_x = _share(_grid_x, frequency * nx);
            peak.share_y = _share(_grid_y, static_cast<double>(j));
            if (peak.share_y >= kOneAxisMinShare) found.push_back({ peak, sy[j] });
        }
        result.lattice = best >= kLatticeMinStrength;
        result.codec_peak = best_codec >= kLatticeMinStrength;
        // those of both axes first, the strongest first; what lies next to a peak that was taken belongs to it
        std::stable_sort(found.begin(), found.end(), [](const Found & a, const Found & b) {
            const bool a_both = a.peak.axes == Summary::Peak::Both, b_both = b.peak.axes == Summary::Peak::Both;
            if (a_both != b_both) return a_both;
            return a.score > b.score;
        });
        for (const Found & candidate : found)
        {
            if (static_cast<int>(result.peaks.size()) >= kResolutionPeaks) break;
            bool taken = false;
            for (const Summary::Peak & peak : result.peaks)
            {
                if (std::abs(peak.frequency - candidate.peak.frequency) < kPeakApart) taken = true;
            }
            if (!taken) result.peaks.push_back(candidate.peak);
        }

        // the peak at 0.5 (the last bins) against the encoder's at 0.25
        auto last_bins = [](const std::vector<double> & strength) {
            return strength.size() < 2 ? 0.0 : std::max(strength[strength.size() - 1], strength[strength.size() - 2]);
        };
        result.half_strength_x = last_bins(sx);
        result.half_strength_y = last_bins(sy);
        result.quarter_strength_x = strongest_near(sx, nx, 0.25, tolerance);
        result.quarter_strength_y = strongest_near(sy, ny, 0.25, tolerance);
        result.half_present = result.half_strength_x >= kLatticeMinStrength && result.half_strength_y >= kLatticeMinStrength;
        result.half_peak = result.half_present && result.half_strength_x >= kHalfPeakShare * result.quarter_strength_x
                        && result.half_strength_y >= kHalfPeakShare * result.quarter_strength_y;
    }

    // ---- picture entering at the edge of a pan
    result.pan_frames_right = _pan_frames[0];
    result.pan_frames_left = _pan_frames[1];
    const bool pan = _pan_frames[0] + _pan_frames[1] >= kMinPanFrames;
    if (pan)
    {
        for (size_t age = 0; age < kResolutionAges; ++age)
        {
            if (_columns[0][age] + _columns[1][age] == 0) continue;
            std::vector<double> & contrast = result.contrast[age];
            contrast.assign(kResolutionBins, 0.0);
            for (size_t b = 0; b < kResolutionBins; ++b)
            {
                const double entering = _entering[0][age][b] + _entering[1][age][b];
                const double leaving = _leaving[0][age][b] + _leaving[1][age][b];
                contrast[b] = leaving > 0.0 ? entering / leaving : 0.0;
            }
            // coarse detail is all there in a single frame: the first bin is the unit. Power -> contrast
            const double unit = contrast[0];
            for (double & value : contrast) value = unit > 0.0 ? std::sqrt(value / unit) : 0.0;
        }
        const std::vector<double> & newest = result.contrast[0];
        if (!newest.empty())
        {
            for (size_t b = 1; b < kResolutionBins; ++b)
            {
                if (newest[b] < 0.5 && newest[b - 1] >= 0.5)
                {
                    const double t = (newest[b - 1] - 0.5) / (newest[b - 1] - newest[b]);
                    result.half_contrast_frequency = bin_frequency(static_cast<int>(b) - 1) + t / kResolutionSegment;
                    break;
                }
            }
            size_t lowest = 0, highest = 0;
            for (size_t b = 1; b < kResolutionBins; ++b)
            {
                if (newest[b] < newest[lowest]) lowest = b;
                if (newest[b] > newest[highest]) highest = b;
            }
            result.lowest_contrast = newest[lowest];
            result.lowest_contrast_frequency = bin_frequency(static_cast<int>(lowest));
            result.highest_contrast = newest[highest];
            result.highest_contrast_frequency = bin_frequency(static_cast<int>(highest));
            for (int age = kResolutionAges - 1; age > 0; --age)
            {
                if (result.contrast[static_cast<size_t>(age)].empty()) continue;
                result.recovered_contrast = result.contrast[static_cast<size_t>(age)][lowest];
                result.recovered_age = age;
                // an upward fan: more contrast than the leaving picture, and less of it the older the columns are
                result.fan_up = result.half_contrast_frequency == 0.0 && result.highest_contrast >= kFanMinContrast && age >= 3
                             && _pan_frames[0] >= kMinPanFrames / 2 && _pan_frames[1] >= kMinPanFrames / 2
                             && result.contrast[static_cast<size_t>(age)][highest] - 1.0 <= kFanSettled * (result.highest_contrast - 1.0);
                break;
            }
        }
    }

    // ---- stair steps
    result.stairs_entering = _stairs[0];
    result.stairs_leaving = _stairs[1];
    {
        long long high[2] = { 0, 0 }, total[2] = { 0, 0 };
        double sum = 0.0;
        long long counted = 0;
        for (int height = 1; height <= kResolutionStairHeights; ++height)
        {
            for (int side = 0; side < 2; ++side)
            {
                const long long steps = _stairs[static_cast<size_t>(side)][static_cast<size_t>(height)];
                total[side] += steps;
                if (height >= 2) high[side] += steps;
            }
            if (height >= 2 && height < kResolutionStairHeights)
            {
                sum += static_cast<double>(height) * static_cast<double>(_stairs[0][static_cast<size_t>(height)]);
                counted += _stairs[0][static_cast<size_t>(height)];
            }
        }
        result.stair_steps_entering = total[0];
        result.stair_steps_leaving = total[1];
        result.stair_high_entering = total[0] > 0 ? static_cast<double>(high[0]) / static_cast<double>(total[0]) : 0.0;
        result.stair_high_leaving = total[1] > 0 ? static_cast<double>(high[1]) / static_cast<double>(total[1]) : 0.0;
        result.stair_height = counted > 0 ? sum / static_cast<double>(counted) : 0.0;
        result.stairs = pan && total[0] >= kStairMinSteps && total[1] >= kStairMinSettledSteps && result.stair_height > 0.0
                     && result.stair_high_entering >= kStairHighShare && result.stair_high_leaving <= kStairSettled;
        if (result.stairs) result.stair_scale = 1.0 / result.stair_height;
    }

    // ---- the candidates
    // a single frame resolves up to half its sampling rate: where the contrast of the entering picture is down
    // to one half is taken as that limit (it was 0.167 for a picture rendered at 1 / 3, whose limit is 0.167)
    const double pan_scale = result.half_contrast_frequency > 0.0 ? std::min(1.0, 2.0 * result.half_contrast_frequency) : 0.0;
    typedef Summary::Candidate Candidate;
    auto add = [&result](double scale, Candidate::Confidence confidence, Candidate::Evidence evidence, int peak) {
        Candidate candidate;
        candidate.scale = scale;
        candidate.confidence = confidence;
        candidate.evidence = evidence;
        candidate.peak = peak;
        result.candidates.push_back(candidate);
    };
    bool first_grid = true;
    for (size_t i = 0; i < result.peaks.size(); ++i)
    {
        const Summary::Peak & peak = result.peaks[i];
        // the scale of the pair that fits the pan first, if there is a pan that says something
        double a = peak.frequency, b = 1.0 - peak.frequency;
        if (pan_scale > 0.0 && std::abs(std::log(b / pan_scale)) < std::abs(std::log(a / pan_scale))) std::swap(a, b);
        const bool picked = pan_scale > 0.0 && std::abs(std::log(b / pan_scale)) - std::abs(std::log(a / pan_scale)) >= kPanMinMargin;
        // with an upward fan the larger scale was the true one in every recording tried: it goes first
        if (result.fan_up && a < b) std::swap(a, b);
        // stair steps: the scale of the two that is clearly nearer to what they read, if it fits. The steps are
        // whole pixels and only those of 2 and more are read, so they read 50% for anything from 50% to about 60%:
        // a pair like 42% / 58% is left alone
        const double smaller = std::min(peak.frequency, 1.0 - peak.frequency);
        const double to_smaller = result.stairs ? std::abs(std::log(result.stair_scale / smaller)) : 0.0;
        const double to_larger = result.stairs ? std::abs(std::log(result.stair_scale / (1.0 - smaller))) : 0.0;
        const bool stepped = !picked && result.stairs && std::min(to_smaller, to_larger) <= kStairFit
                          && std::abs(to_smaller - to_larger) >= kPanMinMargin;
        if (stepped)
        {
            a = to_smaller < to_larger ? smaller : 1.0 - smaller;
            b = 1.0 - a;
        }
        const int index = static_cast<int>(i);
        if (peak.axes != Summary::Peak::Both)
        {
            add(a, Candidate::Low, Candidate::GridOneAxis, index);
            add(b, Candidate::Low, Candidate::GridOneAxis, index);
        }
        else if (!first_grid)
        {
            add(a, Candidate::Low, Candidate::Grid, index);
            add(b, Candidate::Low, Candidate::Grid, index);
        }
        else if (picked)
        {
            add(a, Candidate::High, Candidate::GridAndPan, index);
            add(b, Candidate::Low, Candidate::GridOtherScale, index);
        }
        else if (stepped)
        {
            add(a, Candidate::High, Candidate::GridAndStairs, index);
            add(b, Candidate::Low, Candidate::GridAgainstStairs, index);
        }
        else if (result.fan_up)
        {
            add(a, Candidate::Medium, Candidate::GridAndFan, index);
            add(b, Candidate::Low, Candidate::GridAgainstFan, index);
        }
        else
        {
            add(a, Candidate::Medium, Candidate::Grid, index);
            add(b, Candidate::Medium, Candidate::Grid, index);
        }
        if (peak.axes == Summary::Peak::Both) first_grid = false;
    }
    // without a grid in both axes: the peak at 0.5 and the stair steps, together if they mean the same
    const bool steps_alone = result.stairs && first_grid;
    if (result.half_peak && steps_alone && std::abs(std::log(result.stair_scale / 0.5)) <= kStairHalf)
    {
        add(0.5, Candidate::Medium, Candidate::HalfAndStairs, -1);
    }
    else
    {
        if (steps_alone) add(result.stair_scale, Candidate::Low, Candidate::Stairs, -1);
        if (result.half_peak) add(0.5, Candidate::Low, Candidate::HalfPeak, -1);
    }
    if (pan_scale > 0.0 && first_grid) add(pan_scale, Candidate::Low, Candidate::Pan, -1);
    std::stable_sort(result.candidates.begin(), result.candidates.end(), [](const Candidate & a, const Candidate & b) {
        return a.confidence > b.confidence;
    });
    return result;
}

void ResolutionAnalyzer::write_data(std::ostream & out, const std::string & video) const
{
    auto number = [](double value) {
        char text[32];
        std::snprintf(text, sizeof(text), "%.6g", value);
        return std::string(text);
    };
    out << "# trdrop++ resolution data 1\n";
    out << "# video: " << video << "\n";
    out << "# width: " << _layout.width << "\n# height: " << _layout.height << "\n# frames: " << _frames << "\n";
    out << "# grid_frames: " << _lattice_frames << "\n# grid_rows: " << _layout.lattice_rows << "\n";
    out << "# pieces: " << _chunks << "\n# piece_frames: " << kLatticeChunk << "\n# piece_strength: " << number(kLatticeChunkStrength) << "\n";
    out << "# block_grid: " << number(kCodecGrid) << "\n";
    out << "# pan_rows_from: " << number(_layout.rows_from) << "\n# pan_rows_to: " << number(_layout.rows_to) << "\n";
    out << "# pan_frames_right: " << _pan_frames[0] << "\n# pan_frames_left: " << _pan_frames[1] << "\n";
    for (int age = 0; age < kResolutionAges; ++age) out << "# age_" << age << ": " << kResolutionAgeNames[age] << "\n";
    out << "# stair_heights: " << kResolutionStairHeights << "\n";
    out << "curve,direction,age,frequency,value\n";

    // pixel grid: the mean magnitude per frame, that relative to the background around it, the share of the pieces
    auto grid = [&](const char * axis, const LatticeAxis & data, double n) {
        if (data.spectrum.empty() || _lattice_frames == 0) return;
        const std::vector<double> strength = relative_strengths(data.spectrum);
        for (size_t k = 0; k < data.spectrum.size(); ++k)
        {
            const std::string frequency = number(static_cast<double>(k) / n);
            out << "grid_" << axis << "_spectrum,,," << frequency << "," << number(data.spectrum[k] / static_cast<double>(_lattice_frames)) << "\n";
            out << "grid_" << axis << "_strength,,," << frequency << "," << number(strength[k]) << "\n";
            if (_chunks >= kLatticeMinChunks)
            {
                out << "grid_" << axis << "_share,,," << frequency << "," << number(static_cast<double>(data.hits[k]) / static_cast<double>(_chunks)) << "\n";
            }
        }
    };
    grid("columns", _grid_x, static_cast<double>(_layout.width));
    grid("rows", _grid_y, static_cast<double>(_layout.lattice_rows));

    // stair steps: how many of every height (in the frequency column; the last one: or higher)
    for (int side = 0; side < 2; ++side)
    {
        for (int height = 1; height <= kResolutionStairHeights; ++height)
        {
            out << (side == 0 ? "stairs_entering" : "stairs_leaving") << ",both,," << height << ","
                << _stairs[static_cast<size_t>(side)][static_cast<size_t>(height)] << "\n";
        }
    }

    // pan: the mean power of a column of the entering and of the leaving edge, and the contrast between them
    static const char * const directions[3] = { "right", "left", "both" };
    for (int d = 0; d < 3; ++d)
    {
        for (size_t age = 0; age < kResolutionAges; ++age)
        {
            const long long columns = d == 2 ? _columns[0][age] + _columns[1][age] : _columns[static_cast<size_t>(d)][age];
            if (columns == 0) continue;
            double entering[kResolutionBins], leaving[kResolutionBins];
            for (size_t b = 0; b < kResolutionBins; ++b)
            {
                entering[b] = d == 2 ? _entering[0][age][b] + _entering[1][age][b] : _entering[static_cast<size_t>(d)][age][b];
                leaving[b] = d == 2 ? _leaving[0][age][b] + _leaving[1][age][b] : _leaving[static_cast<size_t>(d)][age][b];
            }
            const double unit = leaving[0] > 0.0 ? entering[0] / leaving[0] : 0.0;
            for (size_t b = 0; b < kResolutionBins; ++b)
            {
                const std::string tail = std::string(",") + directions[d] + "," + std::to_string(age) + "," + number(bin_frequency(static_cast<int>(b))) + ",";
                out << "pan_entering" << tail << number(entering[b] / static_cast<double>(columns)) << "\n";
                out << "pan_leaving" << tail << number(leaving[b] / static_cast<double>(columns)) << "\n";
                const double ratio = leaving[b] > 0.0 ? entering[b] / leaving[b] : 0.0;
                out << "pan_contrast" << tail << number(unit > 0.0 ? std::sqrt(ratio / unit) : 0.0) << "\n";
            }
        }
    }
}

} // namespace trdrop
