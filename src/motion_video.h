#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "motion.h"
#include "video_writer.h"

namespace trdrop {

//! what the analysis says about a frame of the motion video
struct MotionVideoFrame {
    long long frame = 0;
    //! the frame was compared with the previous one ...
    bool compared = false;
    //! ... and is a duplicate of it / shows a part of the previous frame (torn)
    bool duplicate = false;
    bool torn = false;
    //! framerate over the last second
    double framerate = 0.0;
    MotionSample sample;
};

//! Writes a copy of a video with the motion the analysis works with drawn on it, at the resolution the motion
//! is measured at:
//!   arrow    a point of the grid that was tracked: from where it was in the previous frame to where it is now,
//!            colored by its length. On a duplicate frame the arrows of the last new frame stay, dimmed
//!   red x    a point whose match doesn't come back, it is not used
//!   nothing  no texture at the point
//! and the values of the frame in the upper left corner. The numbers have fixed places, so the text stands still.
class MotionVideo {
public:
    //! frame_width, frame_height: size of the frames of the video, fps: its recorded framerate,
    //! arrow_scale: the arrows are drawn this many times longer than the motion.
    //! throws std::runtime_error if the file can't be written
    MotionVideo(const std::string & path, int frame_width, int frame_height, double fps, double arrow_scale);

    const std::string & path() const { return _path; }
    cv::Size size() const { return _size; }
    //! the encoder
    std::string description() const { return _writer->description(); }
    long long frames() const { return _frames; }

    void write(const MotionVideoFrame & values, const MotionDetail & detail);
    void finish() { _writer->finish(); }

private:
    //! cursor of a line of text: draws into `target` if given and returns the x position after the text
    int _label(cv::Mat * target, int x, int line, const std::string & text) const;
    //! a number in a field of `cells` digits, right aligned, every character in a cell of its own
    int _number(cv::Mat * target, int x, int line, int cells, const std::string & text) const;
    //! the four lines of values, returns the x position where the longest line ends
    int _text(cv::Mat * target, const MotionVideoFrame & values, int used, int rejected, bool counted) const;
    void _draw_points(cv::Mat & picture, const std::vector<MotionPoint> & points, bool dimmed) const;

    std::string _path;
    cv::Size _size;
    double _arrow_scale;
    std::unique_ptr<VideoWriter> _writer;
    long long _frames = 0;

    // text layout
    double _font_scale = 0.6;
    int _digit_width = 0;
    int _space = 0;
    int _line_height = 0;
    int _state_width = 0;
    int _box_width = 0;

    //! colors of the arrows by length (BGR)
    std::vector<cv::Scalar> _colors;
    //! the points of the last new frame
    std::vector<MotionPoint> _held;
    cv::Mat _picture;
};

} // namespace trdrop
