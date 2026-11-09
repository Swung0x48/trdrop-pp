// Generates small lossless test videos with a known "real" framerate / tear pattern.
//
//   trdrop_make_fixtures <output-dir>
//
// 30in60.avi   60 fps container, a new frame every 2nd frame             -> 30 fps, 33.3 ms
// stutter.avi  60 fps container, frames held 1,2,3,2,1,... frames           -> 30 fps, 16.7-50 ms
// tear.avi     60 fps container, 30 fps content, every new frame first
//              appears torn (top 40% new, bottom 60% old)                   -> 30 fps, diffs 0.6/0.4
// 60.avi       60 fps container, every frame new                            -> 60 fps, 16.7 ms

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

const cv::Size kSize(320, 180);
constexpr double kFps = 60.0;

cv::Mat content(int index)
{
    // random noise, so every row of two different contents differs
    cv::Mat image(kSize, CV_8UC3);
    cv::RNG rng(static_cast<uint64_t>(index) * 7919u + 17u);
    rng.fill(image, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(256));
    return image;
}

cv::Mat torn(const cv::Mat & new_frame, const cv::Mat & old_frame, double new_part)
{
    cv::Mat image = old_frame.clone();
    const int rows = static_cast<int>(new_frame.rows * new_part);
    new_frame.rowRange(0, rows).copyTo(image.rowRange(0, rows));
    return image;
}

bool write_video(const std::filesystem::path & path, const std::vector<cv::Mat> & frames)
{
    const int codecs[] = {
        cv::VideoWriter::fourcc('F', 'F', 'V', '1'),
        cv::VideoWriter::fourcc('H', 'F', 'Y', 'U'),
        0, // uncompressed
    };
    for (int codec : codecs)
    {
        cv::VideoWriter writer(path.string(), cv::CAP_FFMPEG, codec, kFps, kSize, true);
        if (!writer.isOpened()) continue;
        for (const cv::Mat & frame : frames) writer.write(frame);
        writer.release();
        std::printf("wrote %s (%zu frames)\n", path.string().c_str(), frames.size());
        return true;
    }
    std::fprintf(stderr, "could not open a lossless video writer for %s\n", path.string().c_str());
    return false;
}

} // namespace

int main(int argc, char ** argv)
{
    if (argc != 2)
    {
        std::fprintf(stderr, "usage: %s <output-dir>\n", argv[0]);
        return 2;
    }
    const std::filesystem::path dir = argv[1];
    std::filesystem::create_directories(dir);
    bool ok = true;

    {
        std::vector<cv::Mat> frames;
        for (int i = 0; i < 240; ++i) frames.push_back(content(i / 2));
        ok &= write_video(dir / "30in60.avi", frames);
    }
    {
        const int holds[] = { 1, 2, 3, 2 };
        std::vector<cv::Mat> frames;
        for (int c = 0; frames.size() < 240; ++c)
        {
            for (int k = 0; k < holds[c % 4] && frames.size() < 240; ++k) frames.push_back(content(1000 + c));
        }
        ok &= write_video(dir / "stutter.avi", frames);
    }
    {
        std::vector<cv::Mat> frames;
        frames.push_back(content(2000));
        frames.push_back(content(2000));
        for (int c = 1; frames.size() < 240; ++c)
        {
            frames.push_back(torn(content(2000 + c), content(2000 + c - 1), 0.4));
            frames.push_back(content(2000 + c));
        }
        ok &= write_video(dir / "tear.avi", frames);
    }
    {
        std::vector<cv::Mat> frames;
        for (int i = 0; i < 240; ++i) frames.push_back(content(3000 + i));
        ok &= write_video(dir / "60.avi", frames);
    }
    return ok ? 0 : 1;
}
