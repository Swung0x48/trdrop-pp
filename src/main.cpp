// trdrop-cli: command line version of trdrop (https://github.com/cirquit/trdrop)
// Analyzes raw (uncompressed) captures and writes the per frame framerate / frametime / tear data.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
                               different (default: 0)
      --dismiss-tear-pct <%>   tears smaller than this percentage of the frame height are
                               dismissed (default: 10)
      --source-fps <fps>       override the recorded framerate of the container
      --max-frames <n>         stop after n frames

backend:
      --vkdevice <n>           Vulkan device used for the per pixel work (default: the first
                               discrete GPU, otherwise device 0)
      --no-vk                  don't use Vulkan, analyze with OpenCV on the CPU
      --list-vkdevices         list the Vulkan devices and exit
      --decode <mode>          auto (default): vulkan if every video can be decoded that way,
                               otherwise opencv
                               vulkan: decode on the GPU (FFmpeg Vulkan Video), compare the
                                 luma (Y) plane on the GPU, frames never leave the GPU
                               ffmpeg: decode with FFmpeg on the CPU, compare the luma plane
                                 on the CPU (same results as vulkan)
                               opencv: decode with OpenCV, compare BGR -> grayscale
                               luma and grayscale differ slightly, so the results of
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
        else if (arg == "--source-fps") { source_fps.values = split(next(), ','); }
        else if (arg == "--no-vk") { options.use_vulkan = false; }
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
    if (options.decode == "auto" && options.use_vulkan)
    {
        try
        {
            return make_vulkan_video_pipeline(options.inputs, options.vk_device);
        }
        catch (const std::exception & e)
        {
            if (!options.quiet) std::fprintf(stderr, "trdrop-pp: no Vulkan Video decoding (%s), decoding with OpenCV\n", e.what());
        }
    }
#endif
#ifdef USE_FFMPEG
    if (options.decode == "ffmpeg") return make_ffmpeg_cpu_pipeline(options.inputs);
#endif
    return make_opencv_decode_pipeline(options);
}

int run(const Options & options)
{
    std::unique_ptr<FramePipeline> pipeline_owner = make_pipeline(options);
    FramePipeline & pipeline = *pipeline_owner;
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

    if (!options.quiet)
    {
        for (size_t i = 0; i < video_count; ++i)
        {
            std::fprintf(stderr, "v%zu: %s (%dx%d, %s fps, %lld frames reported, pixel-diff %d, dismiss-tear-pct %s)\n"
                       , i + 1, options.inputs[i].c_str()
                       , sources.frame_width(i), sources.frame_height(i)
                       , format_number(recorded_framerates[i]).c_str(), sources.reported_frame_count(i)
                       , options.videos[i].pixel_difference
                       , format_number(options.videos[i].dismiss_tear_percentage).c_str());
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
    }
    *output << '\n';

    FrameAnalyzer analyzer(window_sizes);
    std::vector<VideoStats> stats(video_count);
    ProgressPrinter progress(!options.quiet);
    std::vector<size_t> tears_per_video(video_count);
    std::vector<RowCounts> row_counts;
    std::string line;

    using Clock = std::chrono::steady_clock;
    const Clock::time_point start_time = Clock::now();
    Clock::duration analysis_time {};

    // takes the oldest result of the pipeline and writes its row
    long long frame = 0;
    auto emit_frame = [&]() {
        pipeline.collect(row_counts);
        const Clock::time_point analyzer_start = Clock::now();
        analyzer.process(row_counts, options.videos);
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

        line = std::to_string(frame);
        for (size_t i = 0; i < video_count; ++i)
        {
            line += ',';
            if (analyzer.has_difference()) line += format_number(differences[i]);
            line += ',' + format_number(framerates[i]);
            line += ',' + format_number(frametimes[i]);
            line += ',' + std::to_string(tears_per_video[i]);

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
        line += '\n';
        *output << line;
        ++frame;
    };

    // the pipeline may work on a set of frames while the next one is decoded
    long long submitted = 0;
    while ((options.max_frames == 0 || submitted < options.max_frames) && pipeline.submit_next(options.videos))
    {
        ++submitted;
        if (pipeline.pending() >= pipeline.max_pending()) emit_frame();
        progress.update(sources);
    }
    while (pipeline.pending() > 0) emit_frame();
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
    }
    return 0;
}

} // namespace
} // namespace trdrop

int main(int argc, char ** argv)
{
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
