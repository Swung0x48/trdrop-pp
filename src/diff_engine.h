#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "motion.h"
#include "options.h"
#include "resolution.h"

namespace trdrop {

//! result of comparing one frame of a video with its previous frame
struct RowCounts {
    //! false if there was nothing to compare with (first frame or the resolution changed)
    bool compared = false;
    int cols = 0;
    //! per row: amount of pixels whose grayscale difference is > pixel_difference
    std::vector<uint32_t> counts;
    //! how the frame is coded (FFmpeg decoders), the automatic tolerance of lossy codecs depends on it
    FrameCoding coding = FrameCoding::Unknown;
    //! motion from the previous frame, only if the pipeline measures it (FramePipeline::enable_motion)
    MotionField motion;
    //! the picture and the tracked points for the motion video (FramePipeline::enable_motion_detail)
    std::shared_ptr<MotionDetail> detail;
    //! what the render resolution is estimated from, only if the pipeline measures it (FramePipeline::enable_resolution)
    std::shared_ptr<ResolutionFrame> resolution;
};

//! Computes the per pixel part of the analysis: grayscale conversion, the difference to the
//! previous frame, the pixel difference threshold and the per row sum of different pixels.
//!
//! Usage per set of frames: decode into next_frames(), call submit(). A backend may keep
//! up to max_pending() submissions in flight, collect() returns them in submission order.
class DiffEngine {
public:
    virtual ~DiffEngine() = default;

    //! human readable backend name for the input info
    virtual std::string description() const = 0;

    //! the frames of the next submit() have to be decoded into these Mats (one per video).
    //! A backend may hand out headers to GPU visible memory, so the decoder writes directly into it.
    //! Reallocating / replacing them is allowed (the backend copies then).
    virtual std::vector<cv::Mat> & next_frames() = 0;

    //! starts comparing next_frames() to the frames of the previous submit()
    virtual void submit(const std::vector<VideoOptions> & video_options) = 0;

    //! submissions which were not collected yet
    virtual size_t pending() const = 0;
    //! collect() has to be called once pending() reaches this value
    virtual size_t max_pending() const = 0;

    //! waits for the oldest pending submission, one entry per video
    virtual void collect(std::vector<RowCounts> & out) = 0;
};

//! the OpenCV (CPU) implementation
std::unique_ptr<DiffEngine> make_cpu_engine(size_t video_count);

#ifdef USE_VULKAN
struct VulkanDeviceInfo {
    std::string name;
    std::string type;
};

//! all Vulkan devices in the order used by --vkdevice, throws if Vulkan is not available
std::vector<VulkanDeviceInfo> list_vulkan_devices();

//! the device index --vkdevice resolves to (device_index < 0: automatic) and its name,
//! throws std::runtime_error if Vulkan or the device is not available
int resolve_vulkan_device(int device_index, std::string & name);

//! device_index < 0: pick automatically (first discrete GPU, otherwise device 0).
//! throws std::runtime_error if Vulkan or the device is not available
std::unique_ptr<DiffEngine> make_vulkan_engine(int device_index, size_t video_count);
#endif

} // namespace trdrop
