#include "resolution.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>

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

ResolutionLayout resolution_layout(int width, int height)
{
    ResolutionLayout layout;
    if (width < 8 * kResolutionSegment || height < 4 * kResolutionSegment) return layout;
    layout.width = width;
    layout.height = height;
    layout.lattice_rows = height * 88 / 100;
    layout.edge = std::min(kResolutionEdgeColumns, width / 8);
    layout.row0 = (height * 9 + 50) / 100;
    const int row1 = (height * 74 + 50) / 100;
    layout.segments = (row1 - layout.row0 - kResolutionSegment) / kResolutionHop + 1;
    return layout;
}

void measure_resolution(const cv::Mat & luma, bool edges, ResolutionFrame & out)
{
    CV_Assert(luma.type() == CV_8UC1);
    const ResolutionLayout layout = resolution_layout(luma.cols, luma.rows);
    out.layout = layout;
    out.edges = false;
    out.column_sums.clear();
    out.row_sums.clear();
    out.edge_power.clear();
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

void ResolutionAnalyzer::_add_lattice(const ResolutionFrame & frame)
{
    // the profile without its slow variation, windowed: magnitude spectrum
    auto add = [this](const std::vector<uint32_t> & sums, std::vector<double> & spectrum, std::vector<float> & window) {
        const int n = static_cast<int>(sums.size());
        if (n < 4 * kLatticeDetrend) return;
        if (static_cast<int>(window.size()) != n)
        {
            window.resize(static_cast<size_t>(n));
            const double pi = 3.14159265358979323846;
            for (int i = 0; i < n; ++i) window[static_cast<size_t>(i)] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * pi * i / (n - 1)));
            spectrum.assign(static_cast<size_t>(n / 2 + 1), 0.0);
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
            profile[i] = static_cast<float>(value - mean) * window[static_cast<size_t>(i)];
        }
        cv::dft(_profile, _transformed, cv::DFT_COMPLEX_OUTPUT);
        const cv::Vec2f * bins = _transformed.ptr<cv::Vec2f>(0);
        for (size_t k = 0; k < spectrum.size(); ++k)
        {
            spectrum[k] += std::sqrt(static_cast<double>(bins[k][0]) * bins[k][0] + static_cast<double>(bins[k][1]) * bins[k][1]);
        }
    };
    add(frame.column_sums, _spectrum_x, _window_x);
    add(frame.row_sums, _spectrum_y, _window_y);
    ++_lattice_frames;
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
}

ResolutionAnalyzer::Summary ResolutionAnalyzer::summary() const
{
    Summary result;
    result.width = _layout.width;
    result.height = _layout.height;
    result.frames = _frames;

    // ---- pixel grid: peaks relative to the background around them
    if (_lattice_frames > 0 && !_spectrum_x.empty() && !_spectrum_y.empty())
    {
        auto strengths = [](const std::vector<double> & spectrum) {
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
        };
        const std::vector<double> sx = strengths(_spectrum_x), sy = strengths(_spectrum_y);
        // bin k of n / 2 + 1 bins: k / n cycles per pixel
        const double nx = static_cast<double>(_layout.width), ny = static_cast<double>(_layout.lattice_rows);
        const double tolerance = 1.5 / nx + 1.5 / ny;
        double best = 0.0, best_codec = 0.0;
        for (size_t k = 0; k < sx.size(); ++k)
        {
            const double frequency = static_cast<double>(k) / nx;
            if (frequency < kLatticeMinFrequency) continue;
            // the same frequency in the rows
            const int first = std::max(0, static_cast<int>(std::ceil((frequency - tolerance) * ny)));
            const int last = std::min(static_cast<int>(sy.size()) - 1, static_cast<int>(std::floor((frequency + tolerance) * ny)));
            double rows = 0.0;
            for (int j = first; j <= last; ++j) rows = std::max(rows, sy[static_cast<size_t>(j)]);
            const double strength = std::min(sx[k], rows);
            const double on_grid = frequency * kCodecGrid;
            const bool codec = std::abs(on_grid - std::round(on_grid)) <= tolerance * kCodecGrid;
            if (codec)
            {
                if (strength > best_codec)
                {
                    best_codec = strength;
                    result.codec_frequency = frequency;
                    result.codec_strength = strength;
                }
            }
            else if (strength > best)
            {
                best = strength;
                result.lattice_frequency = frequency;
                result.lattice_strength_x = sx[k];
                result.lattice_strength_y = rows;
                // between the bins: the top of the parabola through the peak and its neighbours
                if (k > 0 && k + 1 < sx.size())
                {
                    const double curvature = sx[k - 1] - 2.0 * sx[k] + sx[k + 1];
                    if (curvature < 0.0) result.lattice_frequency += std::min(0.5, std::max(-0.5, 0.5 * (sx[k - 1] - sx[k + 1]) / curvature)) / nx;
                }
            }
        }
        result.lattice = best >= kLatticeMinStrength;
        result.lattice_on_codec_grid = !result.lattice && best_codec >= kLatticeMinStrength;
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
            size_t lowest = 0;
            for (size_t b = 1; b < kResolutionBins; ++b)
            {
                if (newest[b] < newest[lowest]) lowest = b;
            }
            result.lowest_contrast = newest[lowest];
            result.lowest_contrast_frequency = bin_frequency(static_cast<int>(lowest));
            for (int age = kResolutionAges - 1; age > 0; --age)
            {
                if (result.contrast[static_cast<size_t>(age)].empty()) continue;
                result.recovered_contrast = result.contrast[static_cast<size_t>(age)][lowest];
                result.recovered_age = age;
                break;
            }
        }
    }

    // ---- the estimate
    // a single frame resolves up to half its sampling rate: where the contrast of the entering picture is down
    // to one half is taken as that limit (it was 0.167 for a picture rendered at 1 / 3, whose limit is 0.167)
    const double pan_scale = result.half_contrast_frequency > 0.0 ? std::min(1.0, 2.0 * result.half_contrast_frequency) : 0.0;
    if (result.lattice)
    {
        const double a = result.lattice_frequency, b = 1.0 - result.lattice_frequency;
        result.scale = a;
        result.other_scale = b;
        result.source = Summary::Lattice;
        if (pan_scale > 0.0)
        {
            if (std::abs(std::log(b / pan_scale)) < std::abs(std::log(a / pan_scale))) std::swap(result.scale, result.other_scale);
            result.source = Summary::LatticeAndPan;
        }
    }
    else if (pan_scale > 0.0)
    {
        result.scale = pan_scale;
        result.source = Summary::Pan;
    }
    return result;
}

} // namespace trdrop
