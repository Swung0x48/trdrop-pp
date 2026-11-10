// Generates small lossless test videos with a known "real" framerate / tear pattern.
//
//   trdrop_make_fixtures <output-dir>
//
// 30in60.avi   60 fps container, a new frame every 2nd frame             -> 30 fps, 33.3 ms
// stutter.avi  60 fps container, frames held 1,2,3,2,1,... frames           -> 30 fps, 16.7-50 ms
// tear.avi     60 fps container, 30 fps content, every new frame first
//              appears torn (top 40% new, bottom 60% old)                   -> 30 fps, diffs 0.6/0.4
// 60.avi       60 fps container, every frame new                            -> 60 fps, 16.7 ms
//
// Fixtures for --motion (640x360, 60 fps, a texture moving with a known animation time per frame):
// motion_even.avi    pan, every frame advances by one interval             -> step 1, error 0, perceived 60 fps
// motion_30in60.avi  pan, a new frame every 2nd frame                      -> step 2, error 0, perceived 30 fps
// motion_uneven.avi  pan, the frames advance by 0.5 / 1.5 intervals        -> error -8.3 / +8.3 ms, perceived 48 fps
// motion_hitch.avi   pan, frames 100-104 freeze and the time is lost       -> one error of -83.3 ms,
//                    frames 170-173 freeze and frame 174 catches up        -> step 5 after 83.3 ms, error 0
// motion_zoom.avi    forward motion (zoom), every frame advances evenly    -> step 1, error 0, perceived 60 fps

#include <cmath>
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

//! a texture with structure at several scales and many corners, so its motion can be tracked
cv::Mat motion_texture(int width, int height)
{
    cv::RNG rng(4242);
    cv::Mat sum(height, width, CV_32FC3, cv::Scalar::all(0));
    for (int structure : { 8, 24, 72 })
    {
        cv::Mat noise(height / structure + 2, width / structure + 2, CV_32FC3);
        rng.fill(noise, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(1));
        cv::Mat layer;
        cv::resize(noise, layer, cv::Size(width, height), 0, 0, cv::INTER_CUBIC);
        sum += layer;
    }
    cv::Mat image;
    sum.convertTo(image, CV_8UC3, 255.0 / 3.0);
    for (int i = 0; i < width * height / 4000; ++i)
    {
        const cv::Point center(rng.uniform(0, width), rng.uniform(0, height));
        const cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        const int size = rng.uniform(6, 28);
        if (i % 2 == 0) cv::rectangle(image, cv::Rect(center.x - size / 2, center.y - size / 2, size, size), color, cv::FILLED);
        else cv::circle(image, center, size / 2, color, cv::FILLED);
    }
    // band limited, so sub pixel positions are well defined
    cv::GaussianBlur(image, image, cv::Size(5, 5), 1.0);
    return image;
}

const cv::Size kMotionSize(640, 360);

//! the texture panned to the animation time `time` (in display intervals)
cv::Mat pan_frame(const cv::Mat & texture, double time)
{
    const double matrix[] = { 1.0, 0.0, 20.0 + 5.3 * time, 0.0, 1.0, 20.0 + 1.7 * time };
    cv::Mat frame;
    cv::warpAffine(texture, frame, cv::Mat(2, 3, CV_64F, const_cast<double *>(matrix)), kMotionSize, cv::INTER_CUBIC | cv::WARP_INVERSE_MAP);
    return frame;
}

//! the camera moves forward: the texture is scaled around the center by 0.5% per interval
cv::Mat zoom_frame(const cv::Mat & texture, double time)
{
    const double scale = 1.6 * std::exp(-0.005 * time);
    const double matrix[] = { scale, 0.0, texture.cols / 2.0 - scale * kMotionSize.width / 2.0
                            , 0.0, scale, texture.rows / 2.0 - scale * kMotionSize.height / 2.0 };
    cv::Mat frame;
    cv::warpAffine(texture, frame, cv::Mat(2, 3, CV_64F, const_cast<double *>(matrix)), kMotionSize, cv::INTER_CUBIC | cv::WARP_INVERSE_MAP);
    return frame;
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
        cv::VideoWriter writer(path.string(), cv::CAP_FFMPEG, codec, kFps, frames.front().size(), true);
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
    // --motion fixtures
    {
        const cv::Mat texture = motion_texture(2000, 800);
        const int count = 240;
        std::vector<cv::Mat> even, half, uneven, hitch;
        for (int i = 0; i < count; ++i)
        {
            even.push_back(pan_frame(texture, i));
            half.push_back(pan_frame(texture, 2 * (i / 2)));
            uneven.push_back(pan_frame(texture, i % 2 == 0 ? i : i - 0.5));
            // frames 100-104 show frame 99 and the time is lost, frames 170-173 show frame 169 and frame 174 catches up
            double time = i < 100 ? i : (i < 105 ? 99 : i - 5);
            if (i >= 170 && i < 174) time = 169 - 5;
            hitch.push_back(pan_frame(texture, time));
        }
        ok &= write_video(dir / "motion_even.avi", even);
        ok &= write_video(dir / "motion_30in60.avi", half);
        ok &= write_video(dir / "motion_uneven.avi", uneven);
        ok &= write_video(dir / "motion_hitch.avi", hitch);

        const cv::Mat large = motion_texture(1400, 800);
        std::vector<cv::Mat> zoom;
        for (int i = 0; i < count; ++i) zoom.push_back(zoom_frame(large, i));
        ok &= write_video(dir / "motion_zoom.avi", zoom);
    }
    return ok ? 0 : 1;
}
