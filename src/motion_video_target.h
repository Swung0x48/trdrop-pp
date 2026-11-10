#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "motion.h"

namespace trdrop {

//! a line with round ends in the motion video (a dot if both ends are the same), in pixels of the picture
struct MotionVideoSegment {
    float x0 = 0.0f;
    float y0 = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    //! half of its width
    float radius = 0.5f;
    uint8_t blue = 0;
    uint8_t green = 0;
    uint8_t red = 0;
};

//! what is drawn on the picture of a frame, in this order
struct MotionVideoScene {
    //! brightness of the picture under everything else
    float dim = 1.0f;
    std::vector<MotionVideoSegment> segments;
    //! the text box: the picture in it is darkened by box_dim and gets the text in white. Empty: no text
    cv::Rect box;
    float box_dim = 1.0f;
    //! coverage of the text (0 - 255), of the size of the box
    cv::Mat text;
};

//! Where the frames of a motion video go: puts the scene on the picture of the frame and encodes it.
//! On the CPU (MotionDetail::picture) or, by a pipeline that has the frames there, on the GPU.
class MotionVideoTarget {
public:
    virtual ~MotionVideoTarget() = default;

    //! encoder, and where the picture is drawn
    virtual std::string description() const = 0;
    virtual void write(const MotionDetail & detail, const MotionVideoScene & scene) = 0;
    //! writes what is left, the file is complete afterwards
    virtual void finish() = 0;
};

} // namespace trdrop
