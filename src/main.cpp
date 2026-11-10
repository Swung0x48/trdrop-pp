// trdrop-cli: command line version of trdrop (https://github.com/cirquit/trdrop)
// Analyzes raw (uncompressed) captures and writes the per frame framerate / frametime / tear data.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "diff_engine.h"
#include "frame_analyzer.h"
#include "frame_pipeline.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#include "options.h"

namespace trdrop {
namespace {

constexpr const char * kVersion = "1.1.1-cli";

void print_usage(std::ostream & out)
{
    out <<
R"(trdrop-pp - raw video analysis (framerate, frametime and tear detection)

usage: trdrop-pp [options] <video> [<video> [<video>]]

Compares up to 3 videos frame by frame and writes one CSV row per frame.

output:
  -o, --output <file>          per frame CSV (default: stdout)
      --tears <file>           additional CSV with one row per detected tear
      --no-summary             don't print the summary to stderr
  -q, --quiet                  no progress output

analysis (single value for all videos, or comma separated per video, e.g. 0,5,0):
      --pixel-diff <0-255>     grayscale difference a pixel may have without counting as
                               different (default: 0, lossy codecs: 4)
      --tolerance-pct <%>      for lossy codecs (H.264, HEVC, ...), whose duplicate frames differ
                               by compression noise: a frame only counts as new if more than
                               this percentage of its pixels is different, a row with at most
                               this percentage counts as unchanged
                               (default: 0; lossy codecs: automatic, at least 0.5 and 5% of
                               what new frames changed in the last 2 seconds)
      --dismiss-tear-pct <%>   tears smaller than this percentage of the frame height are
                               dismissed (default: 10)
      --source-fps <fps>       override the recorded framerate of the container
      --max-frames <n>         stop after n frames

motion (smoothness of the animation, needs coherent motion like a moving camera):
      --motion                 measure the motion between the frames and how evenly the
                               animation advances: animation step, animation error and the
                               perceived smoothness. On the GPU with Vulkan Video decoding,
                               otherwise on the CPU
      --motion-window <ms>     the motion of a frame is compared to the average motion in
                               this window around it (default: 400, growing up to 1000 until
                               it holds 10 new frames)

backend:
      --vkdevice <n>           Vulkan device used for the per pixel work (default: the first
                               discrete GPU, otherwise device 0)
      --no-vk                  don't use Vulkan, analyze with OpenCV on the CPU
      --list-vkdevices         list the Vulkan devices and exit
      --decode <mode>          auto (default): the fastest of the modes below for the videos
                               (vulkan for HEVC / AV1 / ..., ffmpeg for H.264 on CPUs with
                               16+ threads and for YUV formats without Vulkan decoding,
                               opencv for RGB sources), the choice is printed
                               vulkan: decode on the GPU (FFmpeg Vulkan Video), compare the
                                 Y, U and V planes on the GPU, frames never leave the GPU
                               ffmpeg: decode with FFmpeg on the CPU, compare the Y, U and V
                                 planes on the CPU (same results as vulkan)
                               opencv: decode with OpenCV, compare BGR -> grayscale (the
                                 original behaviour, also used with --no-vk)
                               YUV and grayscale differ slightly, so the results of
                               vulkan / ffmpeg differ slightly from opencv

  -h, --help                   show this help
  -v, --version                show the version

per frame columns (vN = N-th input video):
  frame            frame index, starting at 0
  vN_diff          difference to the previous frame: 0 = duplicate, 1 = new frame,
                   (0,1) = torn frame (1 - torn part); empty for frame 0
  vN_fps           sum of vN_diff over the last second (ceil(recorded fps) frames)
  vN_frametime_ms  how long the last completed frame was shown
  vN_tears         amount of tears detected in this frame

additional per frame columns with --motion:
  vN_anim_step      how far the animation advanced since the previous frame, in display
                    intervals (1 = the average around the frame); empty without usable motion
  vN_anim_error_ms  new frames: animation step since the previous new frame minus the time
                    that frame was displayed
  vN_perceived_fps  perceived smoothness over the last second: the framerate of evenly paced
                    motion that looks as smooth, new frames per second / (1 + cv(step)^2)

tear columns:
  frame, video, row_a, row_b, video_height   (the tear is between row_a and row_b)
)";
#ifndef USE_VULKAN
    out << "\nthis build has no Vulkan support (configure with -DUSE_VULKAN=ON), --no-vk is accepted and ignored\n";
#endif
#ifndef USE_FFMPEG
    out << "\nthis build has no FFmpeg support (configure with -DUSE_FFMPEG=ON), only --decode opencv is available\n";
#endif
}

std::vector<std::string> split(const std::string & text, char delimiter)
{
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string part;
    while (std::getline(stream, part, delimiter)) parts.push_back(part);
    if (!text.empty() && text.back() == delimiter) parts.push_back("");
    return parts;
}

double parse_double(const std::string & text, const std::string & option)
{
    try
    {
        size_t consumed = 0;
        const double value = std::stod(text, &consumed);
        if (consumed == text.size() && std::isfinite(value)) return value;
    }
    catch (const std::exception &) { }
    throw std::invalid_argument("invalid value '" + text + "' for " + option);
}

long long parse_integer(const std::string & text, const std::string & option)
{
    try
    {
        size_t consumed = 0;
        const long long value = std::stoll(text, &consumed);
        if (consumed == text.size()) return value;
    }
    catch (const std::exception &) { }
    throw std::invalid_argument("invalid value '" + text + "' for " + option);
}

//! a value list for the per video options, either a single value or one per video
struct PerVideoValue {
    std::string option;
    std::vector<std::string> values;
};

template <typename Setter>
void apply_per_video(const PerVideoValue & value, std::vector<VideoOptions> & videos, Setter setter)
{
    if (value.values.empty()) return;
    if (value.values.size() != 1 && value.values.size() != videos.size())
    {
        throw std::invalid_argument(value.option + " expects 1 or " + std::to_string(videos.size()) + " comma separated values");
    }
    for (size_t i = 0; i < videos.size(); ++i)
    {
        const std::string & text = value.values.size() == 1 ? value.values[0] : value.values[i];
        if (!text.empty()) setter(videos[i], text);
    }
}

//! returns false if the program should exit without analyzing (help / version)
bool parse_arguments(int argc, char ** argv, Options & options)
{
    PerVideoValue pixel_diff { "--pixel-diff", {} };
    PerVideoValue dismiss    { "--dismiss-tear-pct", {} };
    PerVideoValue tolerance  { "--tolerance-pct", {} };
    PerVideoValue source_fps { "--source-fps", {} };

    bool only_inputs = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            return argv[++i];
        };
        if (only_inputs || arg.empty() || arg[0] != '-' || arg == "-")
        {
            options.inputs.push_back(arg);
        }
        else if (arg == "--") { only_inputs = true; }
        else if (arg == "-h" || arg == "--help") { print_usage(std::cout); return false; }
        else if (arg == "-v" || arg == "--version") { std::cout << "trdrop-pp " << kVersion << "\n"; return false; }
        else if (arg == "-o" || arg == "--output") { options.output_path = next(); }
        else if (arg == "--tears") { options.tears_path = next(); }
        else if (arg == "--no-summary") { options.print_summary = false; }
        else if (arg == "-q" || arg == "--quiet") { options.quiet = true; }
        else if (arg == "--pixel-diff") { pixel_diff.values = split(next(), ','); }
        else if (arg == "--dismiss-tear-pct") { dismiss.values = split(next(), ','); }
        else if (arg == "--tolerance-pct") { tolerance.values = split(next(), ','); }
        else if (arg == "--source-fps") { source_fps.values = split(next(), ','); }
        else if (arg == "--no-vk") { options.use_vulkan = false; }
        else if (arg == "--motion") { options.motion = true; }
        else if (arg == "--motion-window")
        {
            options.motion_window_ms = parse_double(next(), arg);
            options.motion = true;
            if (options.motion_window_ms < 50.0 || options.motion_window_ms > 5000.0)
            {
                throw std::invalid_argument("--motion-window must be between 50 and 5000 ms");
            }
        }
        else if (arg == "--decode")
        {
            options.decode = next();
            if (options.decode != "auto" && options.decode != "vulkan" && options.decode != "ffmpeg" && options.decode != "opencv")
            {
                throw std::invalid_argument("--decode expects auto, vulkan, ffmpeg or opencv");
            }
#ifndef USE_FFMPEG
            if (options.decode == "vulkan" || options.decode == "ffmpeg")
            {
                throw std::invalid_argument("--decode " + options.decode + ": this build has no FFmpeg support (configure with -DUSE_FFMPEG=ON)");
            }
#endif
#ifndef USE_VULKAN
            if (options.decode == "vulkan") throw std::invalid_argument("--decode vulkan: this build has no Vulkan support (configure with -DUSE_VULKAN=ON)");
#endif
        }
        else if (arg == "--vkdevice")
        {
            const long long value = parse_integer(next(), arg);
#ifndef USE_VULKAN
            (void)value;
            throw std::invalid_argument("--vkdevice: this build has no Vulkan support (configure with -DUSE_VULKAN=ON)");
#else
            if (value < 0 || value > 1000) throw std::invalid_argument("--vkdevice must be >= 0");
            options.vk_device = static_cast<int>(value);
#endif
        }
        else if (arg == "--list-vkdevices")
        {
#ifndef USE_VULKAN
            throw std::invalid_argument("--list-vkdevices: this build has no Vulkan support (configure with -DUSE_VULKAN=ON)");
#else
            const std::vector<VulkanDeviceInfo> devices = list_vulkan_devices();
            if (devices.empty()) std::cout << "no Vulkan devices found\n";
            for (size_t d = 0; d < devices.size(); ++d)
            {
                std::cout << d << ": " << devices[d].name << " (" << devices[d].type << ")\n";
            }
            return false;
#endif
        }
        else if (arg == "--max-frames")
        {
            options.max_frames = parse_integer(next(), arg);
            if (options.max_frames < 0) throw std::invalid_argument("--max-frames must be >= 0");
        }
        else throw std::invalid_argument("unknown option '" + arg + "' (see --help)");
    }

    if (!options.use_vulkan && options.vk_device >= 0) throw std::invalid_argument("--vkdevice and --no-vk can't be combined");
    if (!options.use_vulkan && options.decode == "vulkan") throw std::invalid_argument("--decode vulkan and --no-vk can't be combined");
    if (options.inputs.empty()) throw std::invalid_argument("no input video given (see --help)");
    if (options.inputs.size() > static_cast<size_t>(kMaxVideoCount))
    {
        throw std::invalid_argument("at most " + std::to_string(kMaxVideoCount) + " videos can be compared");
    }
    options.videos.assign(options.inputs.size(), VideoOptions());

    apply_per_video(pixel_diff, options.videos, [](VideoOptions & v, const std::string & text) {
        const long long value = parse_integer(text, "--pixel-diff");
        if (value < 0 || value > 255) throw std::invalid_argument("--pixel-diff must be between 0 and 255");
        v.pixel_difference = static_cast<int>(value);
        v.pixel_difference_set = true;
    });
    apply_per_video(tolerance, options.videos, [](VideoOptions & v, const std::string & text) {
        const double value = parse_double(text, "--tolerance-pct");
        if (value < 0.0 || value >= 100.0) throw std::invalid_argument("--tolerance-pct must be >= 0 and < 100");
        v.tolerance_pct = value;
    });
    apply_per_video(dismiss, options.videos, [](VideoOptions & v, const std::string & text) {
        const double value = parse_double(text, "--dismiss-tear-pct");
        if (value < 0.0 || value > 100.0) throw std::invalid_argument("--dismiss-tear-pct must be between 0 and 100");
        v.dismiss_tear_percentage = value;
    });
    apply_per_video(source_fps, options.videos, [](VideoOptions & v, const std::string & text) {
        const double value = parse_double(text, "--source-fps");
        if (value <= 0.0) throw std::invalid_argument("--source-fps must be > 0");
        v.source_fps_override = value;
    });
    return true;
}

std::string format_number(double value)
{
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.6g", value);
    return buffer;
}

//! per video statistics for the summary
struct VideoStats {
    long long compared_frames = 0;
    long long new_frames = 0;
    long long duplicate_frames = 0;
    long long torn_frames = 0;
    long long tear_count = 0;
    double diff_sum = 0.0;
    // rolling fps / frametime after the first full window
    long long window_samples = 0;
    double fps_sum = 0.0;
    double fps_min = std::numeric_limits<double>::infinity();
    double fps_max = 0.0;
    long long frametime_samples = 0;
    double frametime_sum = 0.0;
    double frametime_max = 0.0;
};

class ProgressPrinter {
public:
    explicit ProgressPrinter(bool enabled) : _enabled(enabled), _last(std::chrono::steady_clock::now()) { }
    void update(const FramePipeline & sources, bool force = false)
    {
        if (!_enabled) return;
        const auto now = std::chrono::steady_clock::now();
        if (!force && now - _last < std::chrono::milliseconds(500)) return;
        _last = now;
        const double progress = sources.progress();
        if (progress >= 0.0)
        {
            std::fprintf(stderr, "\rframe %lld (%.1f%%)   ", sources.frames_read(), progress * 100.0);
        }
        else
        {
            std::fprintf(stderr, "\rframe %lld   ", sources.frames_read());
        }
        std::fflush(stderr);
        _printed = true;
    }
    void finish()
    {
        if (_printed) std::fprintf(stderr, "\n");
    }

private:
    bool _enabled;
    bool _printed = false;
    std::chrono::steady_clock::time_point _last;
};

//! OpenCV decoding with the Vulkan engine (if enabled and available) or the CPU engine
std::unique_ptr<FramePipeline> make_opencv_decode_pipeline(const Options & options)
{
    std::unique_ptr<DiffEngine> engine;
#ifdef USE_VULKAN
    if (options.use_vulkan)
    {
        try
        {
            engine = make_vulkan_engine(options.vk_device, options.inputs.size());
        }
        catch (const std::exception & e)
        {
            // an explicitly requested device has to work, otherwise fall back to the CPU
            if (options.vk_device >= 0) throw;
            std::fprintf(stderr, "trdrop-pp: Vulkan not available (%s), using the CPU\n", e.what());
        }
    }
#endif
    if (!engine) engine = make_cpu_engine(options.inputs.size());
    return make_opencv_pipeline(options.inputs, std::move(engine));
}

std::unique_ptr<FramePipeline> make_pipeline(const Options & options)
{
#if defined(USE_FFMPEG) && defined(USE_VULKAN)
    if (options.decode == "vulkan") return make_vulkan_video_pipeline(options.inputs, options.vk_device);
#endif
#ifdef USE_FFMPEG
    if (options.decode == "ffmpeg") return make_ffmpeg_cpu_pipeline(options.inputs);
    // auto: the fastest decoder for the inputs; --no-vk keeps the original OpenCV path
    if (options.decode == "auto" && options.use_vulkan)
    {
        auto note = [&](const std::string & what, const std::string & why) {
            if (!options.quiet) std::fprintf(stderr, "trdrop-pp: decoding with %s (%s)\n", what.c_str(), why.c_str());
        };
#ifdef USE_VULKAN
        const bool vulkan_available = true;
#else
        const bool vulkan_available = false;
#endif
        // with --motion the GPU also measures the motion, which is much slower on the CPU
        const DecoderChoice choice = choose_decoder(options.inputs, vulkan_available, options.motion);
        std::string fallback_reason = choice.reason;
#ifdef USE_VULKAN
        if (choice.decoder == "vulkan")
        {
            try
            {
                return make_vulkan_video_pipeline(options.inputs, options.vk_device);
            }
            catch (const std::exception & e)
            {
                // e.g. no Vulkan driver or the device can't decode this profile
                fallback_reason = std::string("no Vulkan Video decoding: ") + e.what();
            }
        }
#endif
        if (choice.decoder == "vulkan" || choice.decoder == "ffmpeg")
        {
            try
            {
                std::unique_ptr<FramePipeline> pipeline = make_ffmpeg_cpu_pipeline(options.inputs);
                note("FFmpeg on the CPU", fallback_reason);
                return pipeline;
            }
            catch (const std::exception & e)
            {
                fallback_reason = e.what();
            }
        }
        note("OpenCV", fallback_reason);
    }
#endif
    return make_opencv_decode_pipeline(options);
}

int run(const Options & options)
{
    std::unique_ptr<FramePipeline> pipeline_owner = make_pipeline(options);
    FramePipeline & pipeline = *pipeline_owner;
    if (options.motion) pipeline.enable_motion();
    const FramePipeline & sources = pipeline;
    const size_t video_count = sources.count();

    // the framerate window has the size of the recorded framerate, as we can't detect a higher framerate
    std::vector<int> window_sizes;
    std::vector<double> recorded_framerates;
    for (size_t i = 0; i < video_count; ++i)
    {
        const double recorded = options.videos[i].source_fps_override > 0.0
                              ? options.videos[i].source_fps_override
                              : sources.recorded_framerate(i);
        if (!(recorded > 0.0) || !std::isfinite(recorded))
        {
            throw std::runtime_error("could not read the framerate of '" + options.inputs[i] + "', use --source-fps");
        }
        recorded_framerates.push_back(recorded);
        window_sizes.push_back(static_cast<int>(std::ceil(recorded - 1e-9)));
    }

    // lossy codecs: duplicate frames differ by compression noise, tolerate it unless told otherwise
    std::vector<VideoOptions> video_options = options.videos;
    std::vector<std::string> automatic(video_count);
    for (size_t i = 0; i < video_count; ++i)
    {
        VideoOptions & v = video_options[i];
        const bool lossy = sources.lossy_codec(i);
        if (lossy && !v.pixel_difference_set)
        {
            v.pixel_difference = kLossyPixelDifference;
            automatic[i] = "pixel-diff";
        }
        if (v.tolerance_pct < 0.0)
        {
            v.tolerance_pct = lossy ? kLossyTolerancePct : 0.0;
            v.tolerance_auto = lossy;
            if (lossy) automatic[i] += automatic[i].empty() ? "tolerance-pct" : " and tolerance-pct";
        }
        if (!automatic[i].empty()) automatic[i] = ", " + automatic[i] + " set for the lossy codec";
    }

    if (!options.quiet)
    {
        for (size_t i = 0; i < video_count; ++i)
        {
            std::fprintf(stderr, "v%zu: %s (%dx%d, %s, %s fps, %lld frames reported, pixel-diff %d, tolerance-pct %s, dismiss-tear-pct %s%s)\n"
                       , i + 1, options.inputs[i].c_str()
                       , sources.frame_width(i), sources.frame_height(i), sources.codec_name(i).c_str()
                       , format_number(recorded_framerates[i]).c_str(), sources.reported_frame_count(i)
                       , video_options[i].pixel_difference
                       , (video_options[i].tolerance_auto ? "auto" : format_number(video_options[i].tolerance_pct)).c_str()
                       , format_number(video_options[i].dismiss_tear_percentage).c_str()
                       , automatic[i].c_str());
        }
    }

    if (!options.quiet) std::fprintf(stderr, "backend: %s\n", pipeline.description().c_str());

    // outputs
    std::ofstream output_file;
    std::ostream * output = &std::cout;
    if (options.output_path != "-")
    {
        output_file.open(options.output_path, std::ios::out | std::ios::trunc);
        if (!output_file) throw std::runtime_error("could not open '" + options.output_path + "' for writing");
        output = &output_file;
    }
    std::ofstream tears_file;
    if (!options.tears_path.empty())
    {
        tears_file.open(options.tears_path, std::ios::out | std::ios::trunc);
        if (!tears_file) throw std::runtime_error("could not open '" + options.tears_path + "' for writing");
        tears_file << "frame,video,row_a,row_b,video_height\n";
    }

    *output << "frame";
    for (size_t i = 0; i < video_count; ++i)
    {
        const std::string prefix = "v" + std::to_string(i + 1) + "_";
        *output << ',' << prefix << "diff" << ',' << prefix << "fps" << ',' << prefix << "frametime_ms" << ',' << prefix << "tears";
        if (options.motion) *output << ',' << prefix << "anim_step" << ',' << prefix << "anim_error_ms" << ',' << prefix << "perceived_fps";
    }
    *output << '\n';

    FrameAnalyzer analyzer(window_sizes);
    std::vector<VideoStats> stats(video_count);
    ProgressPrinter progress(!options.quiet);
    std::vector<size_t> tears_per_video(video_count);
    std::vector<RowCounts> row_counts;
    std::string line;

    // --motion: the motion results of a frame need the frames after it, the rows wait for them
    struct PendingRow {
        long long frame = 0;
        std::vector<std::string> columns; // per video, each starts with a comma
    };
    std::deque<PendingRow> pending_rows;
    std::vector<MotionAnalyzer> motion_analyzers;
    if (options.motion)
    {
        for (size_t i = 0; i < video_count; ++i) motion_analyzers.emplace_back(recorded_framerates[i], options.motion_window_ms);
    }
    auto write_ready_rows = [&]() {
        while (!pending_rows.empty())
        {
            for (const MotionAnalyzer & motion : motion_analyzers)
            {
                if (!motion.ready()) return;
            }
            PendingRow & row = pending_rows.front();
            line = std::to_string(row.frame);
            for (size_t i = 0; i < video_count; ++i)
            {
                MotionSample sample;
                motion_analyzers[i].pop(sample);
                line += row.columns[i];
                line += ',';
                if (sample.measurable) line += format_number(sample.step);
                line += ',';
                if (sample.has_error) line += format_number(sample.error_ms);
                line += ',';
                if (sample.has_perceived_fps) line += format_number(sample.perceived_fps);
            }
            line += '\n';
            *output << line;
            pending_rows.pop_front();
        }
    };

    using Clock = std::chrono::steady_clock;
    const Clock::time_point start_time = Clock::now();
    Clock::duration analysis_time {};

    // takes the oldest result of the pipeline and writes its row
    long long frame = 0;
    auto emit_frame = [&]() {
        pipeline.collect(row_counts);
        const Clock::time_point analyzer_start = Clock::now();
        analyzer.process(row_counts, video_options);
        analysis_time += Clock::now() - analyzer_start;

        const std::vector<double> framerates = analyzer.framerates();
        const std::vector<double> frametimes = analyzer.frametimes();
        const std::vector<double> & differences = analyzer.last_differences();

        std::fill(tears_per_video.begin(), tears_per_video.end(), 0);
        for (const Tear & tear : analyzer.tears())
        {
            tears_per_video[tear.video_index] += 1;
            if (tears_file.is_open())
            {
                tears_file << frame << ',' << (tear.video_index + 1) << ',' << tear.index_a << ',' << tear.index_b
                           << ',' << tear.original_video_height << '\n';
            }
        }

        PendingRow row;
        row.frame = frame;
        row.columns.resize(video_count);
        for (size_t i = 0; i < video_count; ++i)
        {
            std::string & column = row.columns[i];
            column = ',';
            if (analyzer.has_difference()) column += format_number(differences[i]);
            column += ',' + format_number(framerates[i]);
            column += ',' + format_number(frametimes[i]);
            column += ',' + std::to_string(tears_per_video[i]);
            if (options.motion)
            {
                motion_analyzers[i].push(row_counts[i].motion, analyzer.has_difference() && differences[i] == 0.0
                                       , analyzer.has_difference() ? analyzer.last_changed_shares()[i] : 1.0);
            }

            if (!analyzer.has_difference()) continue;
            VideoStats & s = stats[i];
            const double diff = differences[i];
            s.compared_frames += 1;
            s.diff_sum += diff;
            if (diff == 0.0) s.duplicate_frames += 1;
            else if (diff >= 1.0) s.new_frames += 1;
            else s.torn_frames += 1;
            s.tear_count += static_cast<long long>(tears_per_video[i]);
            // the rolling values are only meaningful once the window is filled
            if (s.compared_frames >= window_sizes[i])
            {
                s.window_samples += 1;
                s.fps_sum += framerates[i];
                s.fps_min = std::min(s.fps_min, framerates[i]);
                s.fps_max = std::max(s.fps_max, framerates[i]);
                if (frametimes[i] > 0.0)
                {
                    s.frametime_samples += 1;
                    s.frametime_sum += frametimes[i];
                    s.frametime_max = std::max(s.frametime_max, frametimes[i]);
                }
            }
        }
        if (options.motion)
        {
            pending_rows.push_back(std::move(row));
            write_ready_rows();
        }
        else
        {
            line = std::to_string(row.frame);
            for (const std::string & column : row.columns) line += column;
            line += '\n';
            *output << line;
        }
        ++frame;
    };

    // the pipeline may work on a set of frames while the next one is decoded
    long long submitted = 0;
    while ((options.max_frames == 0 || submitted < options.max_frames) && pipeline.submit_next(video_options))
    {
        ++submitted;
        if (pipeline.pending() >= pipeline.max_pending()) emit_frame();
        progress.update(sources);
    }
    while (pipeline.pending() > 0) emit_frame();
    for (MotionAnalyzer & motion : motion_analyzers) motion.finish();
    write_ready_rows();
    progress.update(sources, true);
    progress.finish();
    output->flush();
    if (!*output) throw std::runtime_error("failed writing the output");

    const double total_ms = std::chrono::duration<double, std::milli>(Clock::now() - start_time).count();
    // decoding and comparison may run fused on the GPU, then only the total is meaningful
    const double pipeline_analysis_s = pipeline.analysis_seconds();
    const double analysis_ms = std::chrono::duration<double, std::milli>(analysis_time).count() + pipeline_analysis_s * 1000.0;

    if (options.print_summary)
    {
        std::fprintf(stderr, "analyzed %lld frames in %.2f s (%.1f fps)"
                   , frame, total_ms / 1000.0, total_ms > 0.0 ? static_cast<double>(frame) / (total_ms / 1000.0) : 0.0);
        if (pipeline_analysis_s >= 0.0)
        {
            std::fprintf(stderr, " | analysis %.1f ms (%.3f ms/frame)", analysis_ms, frame > 0 ? analysis_ms / static_cast<double>(frame) : 0.0);
        }
        std::fprintf(stderr, " | %s\n", pipeline.description().c_str());
        for (size_t i = 0; i < video_count; ++i)
        {
            const VideoStats & s = stats[i];
            const double effective_fps = s.compared_frames > 0 ? s.diff_sum / s.compared_frames * recorded_framerates[i] : 0.0;
            std::fprintf(stderr, "v%zu: effective %.2f fps | new %lld, duplicate %lld, torn %lld (%lld tears)"
                       , i + 1, effective_fps, s.new_frames, s.duplicate_frames, s.torn_frames, s.tear_count);
            if (s.window_samples > 0)
            {
                std::fprintf(stderr, " | fps avg %.2f min %.2f max %.2f", s.fps_sum / s.window_samples, s.fps_min, s.fps_max);
            }
            if (s.frametime_samples > 0)
            {
                std::fprintf(stderr, " | frametime avg %.2f ms max %.2f ms", s.frametime_sum / s.frametime_samples, s.frametime_max);
            }
            std::fprintf(stderr, "\n");
        }
        for (size_t i = 0; i < motion_analyzers.size(); ++i)
        {
            const MotionAnalyzer::Summary m = motion_analyzers[i].summary();
            // share of the new frames whose motion could be measured (duplicates are easy and say little)
            const double measurable = m.new_frames > 0 ? 100.0 * static_cast<double>(m.measurable_new_frames) / static_cast<double>(m.new_frames) : 0.0;
            std::fprintf(stderr, "v%zu motion: measured %lld of %lld new frames (%.1f%%)", i + 1, m.measurable_new_frames, m.new_frames, measurable);
            if (m.animation_frames > 0)
            {
                // per second values are robust against single long steps, short videos only have the overall value
                if (m.perceived_fps_median > 0.0)
                {
                    std::fprintf(stderr, " | %.2f new frames/s | perceived smoothness %.2f fps (10%% low %.2f, whole video %.2f)"
                               , m.motion_fps, m.perceived_fps_median, m.perceived_fps_low, m.perceived_fps);
                }
                else
                {
                    std::fprintf(stderr, " | %.2f new frames/s | perceived smoothness %.2f fps", m.motion_fps, m.perceived_fps);
                }
                std::fprintf(stderr, " | animation error median %.2f ms avg %.2f ms p95 %.2f ms max %.2f ms"
                           , m.median_abs_error_ms, m.mean_abs_error_ms, m.p95_abs_error_ms, m.max_abs_error_ms);
            }
            else
            {
                std::fprintf(stderr, " | not enough coherent motion to measure the animation");
            }
            std::fprintf(stderr, "\n");
        }
    }
    return 0;
}

} // namespace
} // namespace trdrop

int main(int argc, char ** argv)
{
#ifdef _WIN32
    // the manifest makes UTF-8 the process code page, the console has to print it as UTF-8 too
    SetConsoleOutputCP(CP_UTF8);
#endif
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    try
    {
        trdrop::Options options;
        if (!trdrop::parse_arguments(argc, argv, options)) return 0;
        return trdrop::run(options);
    }
    catch (const std::invalid_argument & e)
    {
        std::fprintf(stderr, "trdrop-pp: %s\n", e.what());
        return 2;
    }
    catch (const std::exception & e)
    {
        std::fprintf(stderr, "\ntrdrop-pp: error: %s\n", e.what());
        return 1;
    }
}
