// Android: the analysis on the GPU, from the hardware decoder to the hardware encoder.
//
// MediaCodec (through FFmpeg) decodes every frame into a hardware buffer, which Vulkan imports without a copy.
// ahb_import.comp reads its planes into the luma / chroma buffers of the shaders the Vulkan Video pipeline of
// the desktop uses, which then run unchanged: yuv_diff.comp compares the frame with the previous one,
// motion_pyramid.comp and motion_lk.comp track the grid of points. Only the per row counts and the tracked
// points are read back.
//
// Motion video: overlay_base.comp keeps a scaled down picture of every frame on the GPU until the analysis
// has its values, overlay_draw.comp draws the scene on it, nv12_to_rgba.comp converts the result for the
// input surface of a MediaCodec H.264 encoder, which a Vulkan swapchain presents it to. The encoded video is
// written with the NDK's muxer.
//
// Like the Vulkan Video pipeline, several sets of frames are in flight (kSlots): the next frames are decoded
// and submitted while the GPU works on the earlier ones, and nothing waits for a drawn picture. A GPU that
// is only given one frame at a time idles between them and is clocked down for it.

#include "frame_pipeline.h"

#define VK_USE_PLATFORM_ANDROID_KHR
#include <vulkan/vulkan.h>

#include <android/hardware_buffer.h>
#include <android/native_window.h>
#include <media/NdkImageReader.h>
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaFormat.h>
#include <media/NdkMediaMuxer.h>

#include <fcntl.h>
#include <unistd.h>

extern "C" {
#include <libavcodec/mediacodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_mediacodec.h>
}

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "ffmpeg/ffmpeg_decoder.h"
#include "vulkan/vk_motion_shared.h"

namespace trdrop {

namespace {

const uint32_t kYuvDiffSpirv[] = {
#include "yuv_diff.spv.inc"
};
const uint32_t kMotionPyramidSpirv[] = {
#include "motion_pyramid.spv.inc"
};
const uint32_t kMotionLkSpirv[] = {
#include "motion_lk.spv.inc"
};
const uint32_t kOverlayBaseSpirv[] = {
#include "overlay_base.spv.inc"
};
const uint32_t kOverlayDrawSpirv[] = {
#include "overlay_draw.spv.inc"
};
const uint32_t kAhbImportSpirv[] = {
#include "ahb_import.spv.inc"
};
const uint32_t kNv12ToRgbaSpirv[] = {
#include "nv12_to_rgba.spv.inc"
};

//! must match ahb_import.comp
struct ImportPush {
    uint32_t width;
    uint32_t height;
    uint32_t row_words;
    uint32_t crop_left;
    uint32_t crop_top;
    uint32_t buffer_width;
    uint32_t buffer_height;
};

//! must match nv12_to_rgba.comp
struct RgbaPush {
    uint32_t width;
    uint32_t height;
    uint32_t row_words;
};

//! sets of frames in flight
constexpr size_t kSlots = 3;
//! motion video pictures being drawn at the same time
constexpr size_t kDrawSlots = 3;
//! bits per pixel and frame of the motion video, at 60 fps (as on the desktop)
constexpr double kBitsPerPixel = 0.07;
constexpr int32_t kColorFormatSurface = 0x7F000789;

using Clock = std::chrono::steady_clock;

//! adds the time it lives to a sum
struct Timer {
    explicit Timer(double & sum) : _sum(sum) { }
    ~Timer() { _sum += std::chrono::duration<double, std::milli>(Clock::now() - _start).count(); }
    double & _sum;
    Clock::time_point _start = Clock::now();
};

void check(VkResult result, const char * what)
{
    if (result != VK_SUCCESS) throw std::runtime_error(std::string("Vulkan: ") + what + " failed (" + std::to_string(result) + ")");
}

class MediaCodecVulkanPipeline final : public FramePipeline {
public:
    explicit MediaCodecVulkanPipeline(const std::vector<std::string> & paths)
        : _videos(paths.size())
    {
        // TRDROP_GPU_PROFILE: every step of the analysis is run on its own and timed (slower, for tuning)
        _profile = std::getenv("TRDROP_GPU_PROFILE") != nullptr;
        try
        {
            _create_device();
            _diff = _create_compute(kYuvDiffSpirv, sizeof(kYuvDiffSpirv), 5, sizeof(PushConstants), VK_NULL_HANDLE);
            _pyramid = _create_compute(kMotionPyramidSpirv, sizeof(kMotionPyramidSpirv), 2, sizeof(PyramidPush), VK_NULL_HANDLE);
            _lk = _create_compute(kMotionLkSpirv, sizeof(kMotionLkSpirv), 3, sizeof(LkPush), VK_NULL_HANDLE);
            _overlay_base = _create_compute(kOverlayBaseSpirv, sizeof(kOverlayBaseSpirv), 3, sizeof(OverlayBasePush), VK_NULL_HANDLE);
            _overlay_draw = _create_compute(kOverlayDrawSpirv, sizeof(kOverlayDrawSpirv), 5, sizeof(OverlayDrawPush), VK_NULL_HANDLE);
            _rgba = _create_compute(kNv12ToRgbaSpirv, sizeof(kNv12ToRgbaSpirv), 2, sizeof(RgbaPush), VK_NULL_HANDLE);
            _create_slots();
            for (size_t i = 0; i < paths.size(); ++i) _open_video(_videos[i], paths[i]);
            // the first frames: checks that the device decodes them into buffers Vulkan can read; they are
            // the first submission
            for (size_t i = 0; i < paths.size(); ++i)
            {
                if (!_next_frame(i, _slots[0].frames[i])) throw std::runtime_error("'" + paths[i] + "' contains no frames");
            }
            _first_ready = true;
        }
        catch (...)
        {
            _release();
            throw;
        }
    }

    ~MediaCodecVulkanPipeline() override { _release(); }

    MediaCodecVulkanPipeline(const MediaCodecVulkanPipeline &) = delete;
    MediaCodecVulkanPipeline & operator=(const MediaCodecVulkanPipeline &) = delete;

    std::string description() const override
    {
        return "mediacodec decode + vulkan analysis (" + _device_name + "), yuv" + (_motion ? ", motion on the GPU" : "");
    }

    void enable_motion() override { _motion = true; }
    //! the points of every frame are kept; the pictures stay on the GPU (start_motion_video)
    void enable_motion_detail() override { _motion_detail = true; }

    //! The frames never reach the memory, so there is no drawing on the CPU to fall back to: if the video can't
    //! be made on the GPU the analysis stops, it can be run on the CPU instead (--decode ffmpeg)
    std::unique_ptr<MotionVideoTarget> start_motion_video(size_t index, const std::string & path, double fps, std::string & reason) override
    {
        Video & video = _videos.at(index);
        reason.clear();
        try
        {
            _start_encoder(video, path, fps);
        }
        catch (const std::exception & e)
        {
            _stop_encoder(video, false);
            throw std::runtime_error(std::string("the motion video can't be made on the GPU (") + e.what() + "), analyze on the CPU instead");
        }
        return std::make_unique<GpuTarget>(*this, index);
    }

    bool submit_next(const std::vector<VideoOptions> & video_options) override
    {
        Slot & slot = _slots[_submitted % kSlots];
        if (slot.busy) throw std::logic_error("mediacodec pipeline: submit_next() called while the slot is in flight");
        if (_first_ready)
        {
            _first_ready = false;
        }
        else
        {
            for (size_t i = 0; i < _videos.size(); ++i)
            {
                if (_next_frame(i, slot.frames[i])) continue;
                for (Frame & frame : slot.frames) _drop_frame(frame);
                return false;
            }
        }

        check(vkResetCommandBuffer(slot.command_buffer, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo begin { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(slot.command_buffer, &begin), "vkBeginCommandBuffer");
        // what the frames before wrote (the planes and pyramids of the previous frame) is read
        _barrier(slot.command_buffer);
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            _record_frame(slot.command_buffer, _videos[i], slot.frames[i], std::max(0, video_options[i].pixel_difference));
        }
        check(vkEndCommandBuffer(slot.command_buffer), "vkEndCommandBuffer");
        VkSubmitInfo submit { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &slot.command_buffer;
        check(vkResetFences(_device, 1, &slot.fence), "vkResetFences");
        check(vkQueueSubmit(_queue, 1, &submit, slot.fence), "vkQueueSubmit");

        slot.busy = true;
        ++_submitted;
        ++_frames_read;
        for (Video & video : _videos)
        {
            video.has_previous = true;
            video.current = 1 - video.current;
        }
        return true;
    }

    size_t pending() const override { return _submitted - _collected; }
    size_t max_pending() const override { return kSlots; }

    void collect(std::vector<RowCounts> & out) override
    {
        Slot & slot = _slots[_collected % kSlots];
        if (!slot.busy) throw std::logic_error("mediacodec pipeline: nothing to collect");
        {
            const Timer timer(_wait_ms);
            check(vkWaitForFences(_device, 1, &slot.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        }
        out.resize(_videos.size());
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            Frame & frame = slot.frames[i];
            RowCounts & result = out[i];
            result.compared = frame.compared;
            result.cols = frame.width;
            result.counts.clear();
            if (frame.compared)
            {
                const uint32_t * counts = static_cast<const uint32_t *>(frame.row_counts.mapped);
                result.counts.assign(counts, counts + frame.height);
            }
            result.detail.reset();
            if (_motion_detail)
            {
                result.detail = std::make_shared<MotionDetail>();
                result.detail->gpu_picture = frame.picture;
            }
            if (_motion) _read_motion(frame, result.motion, _motion_detail ? &result.detail->points : nullptr);
            // the hardware buffer goes back to the decoder
            _drop_frame(frame);
        }
        slot.busy = false;
        ++_collected;
    }

    double analysis_seconds() const override { return -1.0; }

    size_t count() const override { return _videos.size(); }
    double recorded_framerate(size_t index) const override { return _videos.at(index).decoder->framerate(); }
    long long reported_frame_count(size_t index) const override { return _videos.at(index).decoder->frame_count(); }
    int frame_width(size_t index) const override { return _videos.at(index).decoder->width(); }
    int frame_height(size_t index) const override { return _videos.at(index).decoder->height(); }
    long long frames_read() const override { return _frames_read; }
    std::string codec_name(size_t index) const override { return _videos.at(index).decoder->codec_name(); }
    bool lossy_codec(size_t index) const override { return _videos.at(index).decoder->lossy(); }

private:
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        void * mapped = nullptr;
    };

    struct Compute {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
    };

    //! a frame of a video in a slot: its hardware buffer as an image, and where its results arrive
    struct Frame {
        AImage * image = nullptr;
        VkImage vk_image = VK_NULL_HANDLE;
        VkDeviceMemory vk_memory = VK_NULL_HANDLE;
        VkImageView vk_view = VK_NULL_HANDLE;
        ImportPush import_push {};
        Buffer row_counts;
        Buffer motion_points;
        VkDescriptorSet import_set = VK_NULL_HANDLE;
        VkDescriptorSet diff_set = VK_NULL_HANDLE;
        VkDescriptorSet pyramid_set = VK_NULL_HANDLE;
        VkDescriptorSet lk_set = VK_NULL_HANDLE;
        VkDescriptorSet base_set = VK_NULL_HANDLE;
        int width = 0;
        int height = 0;
        bool compared = false;
        bool motion_estimated = false;
        MotionLayout motion;
        //! motion video: the picture of the frame (index into Encoder::pictures, -1: none)
        int picture = -1;
    };

    struct Slot {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool busy = false;
        std::vector<Frame> frames;
    };

    //! a motion video picture being drawn: the scene, the drawn picture and its RGBA version
    struct DrawSlot {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore acquired = VK_NULL_HANDLE;
        VkSemaphore rendered = VK_NULL_HANDLE;
        bool busy = false;
        Buffer segments;
        Buffer tiles;
        Buffer text;
        Buffer drawn;
        Buffer rgba;
        VkDescriptorSet draw_set = VK_NULL_HANDLE;
        VkDescriptorSet rgba_set = VK_NULL_HANDLE;
    };

    //! the motion video of a video: a hardware encoder fed through a swapchain on its input surface
    struct Encoder {
        bool active = false;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t row_words = 0;
        VkDeviceSize picture_bytes = 0;
        double fps = 0.0;
        AMediaCodec * codec = nullptr;
        ANativeWindow * window = nullptr;
        AMediaMuxer * muxer = nullptr;
        int file = -1;
        ssize_t track = -1;
        bool muxing = false;
        long long written = 0;
        long long presented = 0;
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VkSwapchainKHR swapchain = VK_NULL_HANDLE;
        std::vector<VkImage> images;
        //! pictures of the frames which wait for their values, and the ones not in use
        std::vector<Buffer> pictures;
        std::vector<int> free_pictures;
        std::string path;
    };

    struct Video {
        // decoding: FFmpeg's MediaCodec decoder renders into the window of the reader
        AImageReader * reader = nullptr;
        AVBufferRef * device = nullptr;
        std::unique_ptr<ffmpeg::VideoDecoder> decoder;
        AVFrame * frame = nullptr;
        // per format of the hardware buffers: the conversion and the pipeline that reads them
        uint64_t external_format = 0;
        VkSamplerYcbcrConversion conversion = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        Compute import;
        // ping pong buffers of the planes and the pyramids
        Buffer luma[2];
        Buffer chroma[2];
        Buffer pyramid[2];
        int current = 0;
        int width = 0;
        int height = 0;
        bool has_previous = false;
        Encoder encoder;
    };

    //! the motion video of one of the videos, drawn and encoded by this pipeline
    class GpuTarget final : public MotionVideoTarget {
    public:
        GpuTarget(MediaCodecVulkanPipeline & pipeline, size_t index) : _pipeline(pipeline), _index(index) { }
        std::string description() const override { return "h264 (MediaCodec), drawn on the GPU"; }
        void write(const MotionDetail & detail, const MotionVideoScene & scene) override { _pipeline._draw(_pipeline._videos[_index], detail.gpu_picture, scene); }
        void finish() override { _pipeline._stop_encoder(_pipeline._videos[_index], true); }

    private:
        MediaCodecVulkanPipeline & _pipeline;
        size_t _index;
    };

    // ---- device

    void _create_device()
    {
        VkApplicationInfo application { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        application.pApplicationName = "trdrop++";
        application.pEngineName = "trdrop++";
        application.apiVersion = VK_API_VERSION_1_1;
        const char * instance_extensions[] = { "VK_KHR_surface", "VK_KHR_android_surface" };
        VkInstanceCreateInfo instance_info { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        instance_info.pApplicationInfo = &application;
        instance_info.enabledExtensionCount = 2;
        instance_info.ppEnabledExtensionNames = instance_extensions;
        check(vkCreateInstance(&instance_info, nullptr, &_instance), "vkCreateInstance");

        uint32_t count = 0;
        vkEnumeratePhysicalDevices(_instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(_instance, &count, devices.data());
        if (devices.empty()) throw std::runtime_error("no Vulkan device");
        _physical = devices[0];
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(_physical, &properties);
        _device_name = properties.deviceName;
        if (properties.apiVersion < VK_API_VERSION_1_1) throw std::runtime_error("Vulkan 1.1 is needed");
        if (properties.limits.maxComputeWorkGroupSize[0] < kMotionGroupSize || properties.limits.maxComputeWorkGroupInvocations < kMotionGroupSize)
        {
            throw std::runtime_error("the GPU can't run the motion analysis");
        }
        vkGetPhysicalDeviceMemoryProperties(_physical, &_memory_properties);

        // extensions: hardware buffers as images, a swapchain on the surface of the encoder
        vkEnumerateDeviceExtensionProperties(_physical, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> available(count);
        vkEnumerateDeviceExtensionProperties(_physical, nullptr, &count, available.data());
        auto has = [&](const char * name) {
            for (const VkExtensionProperties & extension : available)
            {
                if (std::strcmp(extension.extensionName, name) == 0) return true;
            }
            return false;
        };
        std::vector<const char *> extensions;
        for (const char * name : { "VK_ANDROID_external_memory_android_hardware_buffer", "VK_EXT_queue_family_foreign", "VK_KHR_swapchain" })
        {
            if (!has(name)) throw std::runtime_error(std::string("the GPU driver has no ") + name);
            extensions.push_back(name);
        }
        _display_timing = has("VK_GOOGLE_display_timing");
        if (_display_timing) extensions.push_back("VK_GOOGLE_display_timing");

        vkGetPhysicalDeviceQueueFamilyProperties(_physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(_physical, &count, families.data());
        _queue_family = UINT32_MAX;
        for (uint32_t i = 0; i < count && _queue_family == UINT32_MAX; ++i)
        {
            // graphics queues can present to a surface
            const VkQueueFlags wanted = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT;
            if ((families[i].queueFlags & wanted) == wanted) _queue_family = i;
        }
        if (_queue_family == UINT32_MAX) throw std::runtime_error("the GPU has no queue for compute and presentation");

        VkPhysicalDeviceSamplerYcbcrConversionFeatures ycbcr { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES };
        VkPhysicalDeviceFeatures2 features { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        features.pNext = &ycbcr;
        auto get_features = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(vkGetInstanceProcAddr(_instance, "vkGetPhysicalDeviceFeatures2"));
        if (!get_features) throw std::runtime_error("Vulkan 1.1 is needed");
        get_features(_physical, &features);
        if (!ycbcr.samplerYcbcrConversion) throw std::runtime_error("the GPU can't sample YCbCr images");
        ycbcr.pNext = nullptr;

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        queue_info.queueFamilyIndex = _queue_family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        VkDeviceCreateInfo device_info { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        device_info.pNext = &ycbcr;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        device_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        device_info.ppEnabledExtensionNames = extensions.data();
        check(vkCreateDevice(_physical, &device_info, nullptr, &_device), "vkCreateDevice");
        vkGetDeviceQueue(_device, _queue_family, 0, &_queue);

        // not in the Vulkan library of old Android versions the app still runs on: looked up
#define TRDROP_LOAD(name) \
        _fn.name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(_device, #name)); \
        if (!_fn.name) throw std::runtime_error("missing Vulkan function " #name);
        TRDROP_LOAD(vkCreateSamplerYcbcrConversion)
        TRDROP_LOAD(vkDestroySamplerYcbcrConversion)
        TRDROP_LOAD(vkGetAndroidHardwareBufferPropertiesANDROID)
#undef TRDROP_LOAD

        VkCommandPoolCreateInfo pool_info { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = _queue_family;
        check(vkCreateCommandPool(_device, &pool_info, nullptr, &_command_pool), "vkCreateCommandPool");
        VkFenceCreateInfo fence_info { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        check(vkCreateFence(_device, &fence_info, nullptr, &_profile_fence), "vkCreateFence");
    }

    VkCommandBuffer _allocate_command_buffer()
    {
        VkCommandBufferAllocateInfo info { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        info.commandPool = _command_pool;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = 1;
        VkCommandBuffer buffer = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(_device, &info, &buffer), "vkAllocateCommandBuffers");
        return buffer;
    }

    void _create_slots()
    {
        const uint32_t frames = static_cast<uint32_t>(kSlots * _videos.size());
        const uint32_t draws = static_cast<uint32_t>(kDrawSlots);
        // per frame of a slot: the import (an image and 2 buffers), the comparison, the pyramid, the tracking
        // and the picture; per picture being drawn: the drawing and the conversion
        const VkDescriptorPoolSize sizes[2] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (2 + 5 + 2 + 3 + 3) * frames + (5 + 2) * draws },
            // an image with several planes may count as several descriptors
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 * frames },
        };
        VkDescriptorPoolCreateInfo pool_info { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 5 * frames + 2 * draws;
        pool_info.poolSizeCount = 2;
        pool_info.pPoolSizes = sizes;
        check(vkCreateDescriptorPool(_device, &pool_info, nullptr, &_descriptor_pool), "vkCreateDescriptorPool");

        VkFenceCreateInfo fence_info { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VkSemaphoreCreateInfo semaphore_info { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        for (Slot & slot : _slots)
        {
            slot.command_buffer = _allocate_command_buffer();
            check(vkCreateFence(_device, &fence_info, nullptr, &slot.fence), "vkCreateFence");
            slot.frames.resize(_videos.size());
            for (Frame & frame : slot.frames)
            {
                frame.diff_set = _allocate_set(_diff);
                frame.pyramid_set = _allocate_set(_pyramid);
                frame.lk_set = _allocate_set(_lk);
                frame.base_set = _allocate_set(_overlay_base);
            }
        }
        for (DrawSlot & slot : _draw_slots)
        {
            slot.command_buffer = _allocate_command_buffer();
            check(vkCreateFence(_device, &fence_info, nullptr, &slot.fence), "vkCreateFence");
            check(vkCreateSemaphore(_device, &semaphore_info, nullptr, &slot.acquired), "vkCreateSemaphore");
            check(vkCreateSemaphore(_device, &semaphore_info, nullptr, &slot.rendered), "vkCreateSemaphore");
            slot.draw_set = _allocate_set(_overlay_draw);
            slot.rgba_set = _allocate_set(_rgba);
        }
    }

    VkDescriptorSet _allocate_set(const Compute & compute)
    {
        VkDescriptorSetAllocateInfo info { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        info.descriptorPool = _descriptor_pool;
        info.descriptorSetCount = 1;
        info.pSetLayouts = &compute.set_layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        check(vkAllocateDescriptorSets(_device, &info, &set), "vkAllocateDescriptorSets");
        return set;
    }

    //! a compute pipeline with `buffers` storage buffers in one set and push constants; with a sampler the
    //! first binding is an image with that (immutable) sampler
    Compute _create_compute(const uint32_t * code, size_t code_bytes, uint32_t buffers, uint32_t push_bytes, VkSampler sampler)
    {
        Compute compute;
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        if (sampler)
        {
            VkDescriptorSetLayoutBinding binding {};
            binding.binding = 0;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            binding.pImmutableSamplers = &sampler;
            bindings.push_back(binding);
        }
        for (uint32_t b = 0; b < buffers; ++b)
        {
            VkDescriptorSetLayoutBinding binding {};
            binding.binding = static_cast<uint32_t>(bindings.size());
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings.push_back(binding);
        }
        VkDescriptorSetLayoutCreateInfo set_layout_info { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        set_layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        set_layout_info.pBindings = bindings.data();
        check(vkCreateDescriptorSetLayout(_device, &set_layout_info, nullptr, &compute.set_layout), "vkCreateDescriptorSetLayout");
        VkPushConstantRange push_range { VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes };
        VkPipelineLayoutCreateInfo layout_info { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &compute.set_layout;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push_range;
        check(vkCreatePipelineLayout(_device, &layout_info, nullptr, &compute.layout), "vkCreatePipelineLayout");
        VkShaderModuleCreateInfo module_info { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        module_info.codeSize = code_bytes;
        module_info.pCode = code;
        VkShaderModule module = VK_NULL_HANDLE;
        check(vkCreateShaderModule(_device, &module_info, nullptr, &module), "vkCreateShaderModule");
        VkComputePipelineCreateInfo pipeline_info { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = compute.layout;
        const VkResult result = vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &compute.pipeline);
        vkDestroyShaderModule(_device, module, nullptr);
        check(result, "vkCreateComputePipelines");
        return compute;
    }

    void _destroy_compute(Compute & compute)
    {
        if (compute.pipeline) vkDestroyPipeline(_device, compute.pipeline, nullptr);
        if (compute.layout) vkDestroyPipelineLayout(_device, compute.layout, nullptr);
        if (compute.set_layout) vkDestroyDescriptorSetLayout(_device, compute.set_layout, nullptr);
        compute = Compute();
    }

    //! Contents are lost if the buffer has to grow. shared: the buffer may be in use by frames in flight,
    //! they are waited for before it is replaced
    void _ensure_buffer(Buffer & buffer, VkDeviceSize bytes, VkBufferUsageFlags usage, bool host, bool shared = false)
    {
        if (buffer.size >= bytes) return;
        if (shared && buffer.buffer) vkQueueWaitIdle(_queue);
        _destroy_buffer(buffer);
        buffer.size = std::max<VkDeviceSize>(bytes, 64);
        VkBufferCreateInfo buffer_info { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        buffer_info.size = buffer.size;
        buffer_info.usage = usage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(_device, &buffer_info, nullptr, &buffer.buffer), "vkCreateBuffer");
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(_device, buffer.buffer, &requirements);
        // host: readable and writable without flushing; otherwise memory of the GPU if there is such a thing
        const VkMemoryPropertyFlags wanted[2] = {
            host ? VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
                 : VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
            host ? VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) : VkMemoryPropertyFlags(0),
        };
        uint32_t type = UINT32_MAX;
        for (const VkMemoryPropertyFlags flags : wanted)
        {
            for (uint32_t i = 0; i < _memory_properties.memoryTypeCount && type == UINT32_MAX; ++i)
            {
                if ((requirements.memoryTypeBits & (1u << i)) && (_memory_properties.memoryTypes[i].propertyFlags & flags) == flags) type = i;
            }
        }
        if (type == UINT32_MAX) throw std::runtime_error("Vulkan: no suitable memory");
        VkMemoryAllocateInfo allocate { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = type;
        check(vkAllocateMemory(_device, &allocate, nullptr, &buffer.memory), "vkAllocateMemory");
        check(vkBindBufferMemory(_device, buffer.buffer, buffer.memory, 0), "vkBindBufferMemory");
        if (host) check(vkMapMemory(_device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped), "vkMapMemory");
    }

    void _destroy_buffer(Buffer & buffer)
    {
        if (buffer.buffer) vkDestroyBuffer(_device, buffer.buffer, nullptr);
        if (buffer.memory) vkFreeMemory(_device, buffer.memory, nullptr);
        buffer = Buffer();
    }

    void _write_set(VkDescriptorSet set, uint32_t first_binding, std::initializer_list<VkBuffer> buffers)
    {
        std::vector<VkDescriptorBufferInfo> infos;
        for (VkBuffer buffer : buffers) infos.push_back({ buffer, 0, VK_WHOLE_SIZE });
        std::vector<VkWriteDescriptorSet> writes(infos.size());
        for (size_t b = 0; b < infos.size(); ++b)
        {
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = set;
            writes[b].dstBinding = first_binding + static_cast<uint32_t>(b);
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &infos[b];
        }
        vkUpdateDescriptorSets(_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    //! what an earlier step (of this command buffer or of one submitted before) wrote is read by the next one
    void _barrier(VkCommandBuffer command_buffer)
    {
        VkMemoryBarrier barrier { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    // ---- decoding

    void _open_video(Video & video, const std::string & path)
    {
        const ffmpeg::Probe probe = ffmpeg::probe(path);
        if (!probe.opened) throw std::runtime_error(probe.error);
        if (probe.format != AV_PIX_FMT_NONE && !ffmpeg::is_luma_format(probe.format))
        {
            throw std::runtime_error("pixel format " + ffmpeg::format_name(probe.format) + " is no 8-bit YUV format");
        }
        // the window the decoder renders into: frames arrive as hardware buffers the GPU can sample
        {
            AVFormatContext * format = nullptr;
            if (ffmpeg::open_input(&format, path) < 0) throw std::runtime_error("could not open video '" + path + "'");
            const int stream = avformat_find_stream_info(format, nullptr) < 0 ? -1 : av_find_best_stream(format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            if (stream >= 0)
            {
                video.width = format->streams[stream]->codecpar->width;
                video.height = format->streams[stream]->codecpar->height;
            }
            avformat_close_input(&format);
            if (stream < 0 || video.width <= 0 || video.height <= 0) throw std::runtime_error("no video stream in '" + path + "'");
        }
        if (video.width % 4 != 0 || video.height % 2 != 0) throw std::runtime_error("the width of the video is no multiple of 4");
        // the frames in flight keep their buffers until they are collected
        if (AImageReader_newWithUsage(video.width, video.height, AIMAGE_FORMAT_PRIVATE, AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE,
                                      static_cast<int32_t>(kSlots) + 2, &video.reader) != AMEDIA_OK)
        {
            throw std::runtime_error("no image reader for the decoder");
        }
        ANativeWindow * window = nullptr;
        if (AImageReader_getWindow(video.reader, &window) != AMEDIA_OK || !window) throw std::runtime_error("no window for the decoder");
        video.device = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_MEDIACODEC);
        if (!video.device) throw std::runtime_error("out of memory");
        AVHWDeviceContext * device = reinterpret_cast<AVHWDeviceContext *>(video.device->data);
        static_cast<AVMediaCodecDeviceContext *>(device->hwctx)->native_window = window;
        const int result = av_hwdevice_ctx_init(video.device);
        if (result < 0) throw std::runtime_error("no MediaCodec device (" + ffmpeg::error_string(result) + ")");
        video.decoder = std::make_unique<ffmpeg::VideoDecoder>(path, video.device);
        video.frame = av_frame_alloc();
        if (!video.frame) throw std::runtime_error("out of memory");
    }

    //! decodes the next frame of the video and imports its hardware buffer, false at the end
    bool _next_frame(size_t index, Frame & frame)
    {
        Video & video = _videos[index];
        _drop_frame(frame);
        const Timer timer(_decode_ms);
        if (!video.decoder->next(video.frame)) return false;
        if (video.frame->format != AV_PIX_FMT_MEDIACODEC || !video.frame->data[3]) throw std::runtime_error("the decoder returned no hardware frame");
        // the frame goes to the window of the reader, where it arrives a moment later
        const int rendered = av_mediacodec_release_buffer(reinterpret_cast<AVMediaCodecBuffer *>(video.frame->data[3]), 1);
        av_frame_unref(video.frame);
        if (rendered < 0) throw std::runtime_error("the decoder could not output the frame (" + ffmpeg::error_string(rendered) + ")");
        for (int attempt = 0; !frame.image; ++attempt)
        {
            const media_status_t status = AImageReader_acquireNextImage(video.reader, &frame.image);
            if (status == AMEDIA_OK && frame.image) break;
            frame.image = nullptr;
            if (status != AMEDIA_IMGREADER_NO_BUFFER_AVAILABLE || attempt > 20000) throw std::runtime_error("the decoded frame didn't arrive");
            usleep(100);
        }
        AHardwareBuffer * buffer = nullptr;
        if (AImage_getHardwareBuffer(frame.image, &buffer) != AMEDIA_OK || !buffer) throw std::runtime_error("the decoded frame has no hardware buffer");
        const Timer import_timer(_import_ms);
        _import(index, frame, buffer);
        return true;
    }

    void _drop_frame(Frame & frame)
    {
        if (frame.vk_view) vkDestroyImageView(_device, frame.vk_view, nullptr);
        if (frame.vk_image) vkDestroyImage(_device, frame.vk_image, nullptr);
        if (frame.vk_memory) vkFreeMemory(_device, frame.vk_memory, nullptr);
        frame.vk_view = VK_NULL_HANDLE;
        frame.vk_image = VK_NULL_HANDLE;
        frame.vk_memory = VK_NULL_HANDLE;
        if (frame.image) AImage_delete(frame.image);
        frame.image = nullptr;
    }

    //! the hardware buffer as an image that ahb_import.comp samples
    void _import(size_t index, Frame & frame, AHardwareBuffer * buffer)
    {
        Video & video = _videos[index];
        AHardwareBuffer_Desc description {};
        AHardwareBuffer_describe(buffer, &description);
        AImageCropRect crop {};
        if (AImage_getCropRect(frame.image, &crop) != AMEDIA_OK) crop = AImageCropRect { 0, 0, video.width, video.height };
        if (crop.right - crop.left < video.width || crop.bottom - crop.top < video.height)
        {
            crop.left = 0;
            crop.top = 0;
        }

        VkAndroidHardwareBufferFormatPropertiesANDROID format { VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID };
        VkAndroidHardwareBufferPropertiesANDROID properties { VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID };
        properties.pNext = &format;
        check(_fn.vkGetAndroidHardwareBufferPropertiesANDROID(_device, buffer, &properties), "vkGetAndroidHardwareBufferPropertiesANDROID");
        if (format.externalFormat == 0) throw std::runtime_error("the decoder's buffers have no format Vulkan can sample");

        if (!video.conversion || video.external_format != format.externalFormat)
        {
            // the conversion leaves the values as they are stored: no matrix, no range expansion, no filtering
            vkDeviceWaitIdle(_device);
            for (Slot & slot : _slots)
            {
                Frame & other = slot.frames[index];
                if (other.import_set) vkFreeDescriptorSets(_device, _descriptor_pool, 1, &other.import_set);
                other.import_set = VK_NULL_HANDLE;
            }
            _destroy_compute(video.import);
            if (video.sampler) vkDestroySampler(_device, video.sampler, nullptr);
            if (video.conversion) _fn.vkDestroySamplerYcbcrConversion(_device, video.conversion, nullptr);
            video.sampler = VK_NULL_HANDLE;
            video.conversion = VK_NULL_HANDLE;
            video.external_format = format.externalFormat;

            VkExternalFormatANDROID external { VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID };
            external.externalFormat = format.externalFormat;
            VkSamplerYcbcrConversionCreateInfo conversion_info { VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO };
            conversion_info.pNext = &external;
            conversion_info.format = VK_FORMAT_UNDEFINED;
            conversion_info.ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_IDENTITY;
            conversion_info.ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_FULL;
            conversion_info.components = format.samplerYcbcrConversionComponents;
            conversion_info.xChromaOffset = format.suggestedXChromaOffset;
            conversion_info.yChromaOffset = format.suggestedYChromaOffset;
            conversion_info.chromaFilter = VK_FILTER_NEAREST;
            check(_fn.vkCreateSamplerYcbcrConversion(_device, &conversion_info, nullptr, &video.conversion), "vkCreateSamplerYcbcrConversion");
            VkSamplerYcbcrConversionInfo conversion { VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO };
            conversion.conversion = video.conversion;
            VkSamplerCreateInfo sampler_info { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
            sampler_info.pNext = &conversion;
            sampler_info.magFilter = VK_FILTER_NEAREST;
            sampler_info.minFilter = VK_FILTER_NEAREST;
            sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            check(vkCreateSampler(_device, &sampler_info, nullptr, &video.sampler), "vkCreateSampler");
            video.import = _create_compute(kAhbImportSpirv, sizeof(kAhbImportSpirv), 2, sizeof(ImportPush), video.sampler);
        }
        if (!frame.import_set) frame.import_set = _allocate_set(video.import);

        VkExternalFormatANDROID external { VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID };
        external.externalFormat = format.externalFormat;
        VkExternalMemoryImageCreateInfo external_memory { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO };
        external_memory.pNext = &external;
        external_memory.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
        VkImageCreateInfo image_info { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        image_info.pNext = &external_memory;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = VK_FORMAT_UNDEFINED;
        image_info.extent = { description.width, description.height, 1 };
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        check(vkCreateImage(_device, &image_info, nullptr, &frame.vk_image), "vkCreateImage");

        VkImportAndroidHardwareBufferInfoANDROID import { VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID };
        import.buffer = buffer;
        VkMemoryDedicatedAllocateInfo dedicated { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
        dedicated.pNext = &import;
        dedicated.image = frame.vk_image;
        VkMemoryAllocateInfo allocate { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocate.pNext = &dedicated;
        allocate.allocationSize = properties.allocationSize;
        allocate.memoryTypeIndex = 0;
        while (allocate.memoryTypeIndex < 32 && !(properties.memoryTypeBits & (1u << allocate.memoryTypeIndex))) ++allocate.memoryTypeIndex;
        check(vkAllocateMemory(_device, &allocate, nullptr, &frame.vk_memory), "vkAllocateMemory (hardware buffer)");
        check(vkBindImageMemory(_device, frame.vk_image, frame.vk_memory, 0), "vkBindImageMemory");

        VkSamplerYcbcrConversionInfo conversion { VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO };
        conversion.conversion = video.conversion;
        VkImageViewCreateInfo view_info { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        view_info.pNext = &conversion;
        view_info.image = frame.vk_image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = VK_FORMAT_UNDEFINED;
        view_info.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        check(vkCreateImageView(_device, &view_info, nullptr, &frame.vk_view), "vkCreateImageView");

        frame.import_push.width = static_cast<uint32_t>(video.width);
        frame.import_push.height = static_cast<uint32_t>(video.height);
        frame.import_push.row_words = static_cast<uint32_t>(video.width) / 4;
        frame.import_push.crop_left = static_cast<uint32_t>(std::max(0, crop.left));
        frame.import_push.crop_top = static_cast<uint32_t>(std::max(0, crop.top));
        frame.import_push.buffer_width = description.width;
        frame.import_push.buffer_height = description.height;
    }

    // ---- analysis of a frame

    //! The frames in flight share the ping pong buffers of the video: the queue runs the submissions in order,
    //! so a frame finds the planes and the pyramid of the one before it
    void _record_frame(VkCommandBuffer command_buffer, Video & video, Frame & frame, int threshold)
    {
        const uint32_t width = static_cast<uint32_t>(video.width);
        const uint32_t height = static_cast<uint32_t>(video.height);
        const uint32_t row_words = width / 4;
        // the planes as the Vulkan Video pipeline lays them out: bytes of luma, UV pairs of chroma
        const VkDeviceSize luma_bytes = static_cast<VkDeviceSize>(row_words) * 4 * height;
        const VkDeviceSize chroma_bytes = static_cast<VkDeviceSize>(row_words) * 4 * (height / 2);
        frame.motion = motion_layout(video.width, video.height);
        frame.width = video.width;
        frame.height = video.height;
        for (int b = 0; b < 2; ++b)
        {
            _ensure_buffer(video.luma[b], luma_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, true);
            _ensure_buffer(video.chroma[b], chroma_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, true);
            if (_motion) _ensure_buffer(video.pyramid[b], frame.motion.bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, true);
        }
        _ensure_buffer(frame.row_counts, static_cast<VkDeviceSize>(height) * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
        if (_motion) _ensure_buffer(frame.motion_points, kMotionPoints * 4 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
        const int current = video.current;
        const int previous = 1 - current;
        frame.compared = video.has_previous;
        frame.motion_estimated = _motion && video.has_previous;

        // the hardware buffer becomes readable for this queue
        VkImageMemoryBarrier acquire { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        acquire.srcAccessMask = 0;
        acquire.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        acquire.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        acquire.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        acquire.srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
        acquire.dstQueueFamilyIndex = _queue_family;
        acquire.image = frame.vk_image;
        acquire.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &acquire);

        VkDescriptorImageInfo image { VK_NULL_HANDLE, frame.vk_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet image_write { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        image_write.dstSet = frame.import_set;
        image_write.dstBinding = 0;
        image_write.descriptorCount = 1;
        image_write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        image_write.pImageInfo = &image;
        vkUpdateDescriptorSets(_device, 1, &image_write, 0, nullptr);
        _write_set(frame.import_set, 1, { video.luma[current].buffer, video.chroma[current].buffer });
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, video.import.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, video.import.layout, 0, 1, &frame.import_set, 0, nullptr);
        vkCmdPushConstants(command_buffer, video.import.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ImportPush), &frame.import_push);
        vkCmdDispatch(command_buffer, (row_words + 7) / 8, (height / 2 + 7) / 8, 1);
        _barrier(command_buffer);
        _profile_step(command_buffer, 0);

        if (frame.compared)
        {
            _write_set(frame.diff_set, 0, { video.luma[current].buffer, video.luma[previous].buffer, video.chroma[current].buffer,
                                            video.chroma[previous].buffer, frame.row_counts.buffer });
            // 4:2:0 with interleaved chroma, as yuv_diff.comp names it
            const PushConstants push { width, row_words, static_cast<uint32_t>(threshold), 1, 1, 1, row_words * 4, 0 };
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _diff.pipeline);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _diff.layout, 0, 1, &frame.diff_set, 0, nullptr);
            vkCmdPushConstants(command_buffer, _diff.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(command_buffer, height, 1, 1);
            _profile_step(command_buffer, 1);
        }

        frame.picture = -1;
        Encoder & encoder = video.encoder;
        if (encoder.active)
        {
            // a picture on the GPU for the frame, until it is drawn
            if (encoder.free_pictures.empty())
            {
                encoder.pictures.emplace_back();
                _ensure_buffer(encoder.pictures.back(), encoder.picture_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
                encoder.free_pictures.push_back(static_cast<int>(encoder.pictures.size()) - 1);
            }
            frame.picture = encoder.free_pictures.back();
            encoder.free_pictures.pop_back();
            _write_set(frame.base_set, 0, { video.luma[current].buffer, video.chroma[current].buffer,
                                            encoder.pictures[static_cast<size_t>(frame.picture)].buffer });
            OverlayBasePush push {};
            push.width = encoder.width;
            push.height = encoder.height;
            push.row_words = encoder.row_words;
            push.scale = static_cast<uint32_t>(motion_scale(video.width, video.height));
            push.luma_row_words = row_words;
            push.chroma_mode = 1;
            push.chroma_row_bytes = row_words * 4;
            push.v_offset = 0;
            push.chroma_step_x = push.scale;
            push.chroma_step_y = push.scale;
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _overlay_base.pipeline);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _overlay_base.layout, 0, 1, &frame.base_set, 0, nullptr);
            vkCmdPushConstants(command_buffer, _overlay_base.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(command_buffer, (encoder.row_words + 7) / 8, (encoder.height / 2 + 7) / 8, 1);
            _profile_step(command_buffer, 2);
        }

        if (!_motion) return;
        const MotionLayout & layout = frame.motion;
        _write_set(frame.pyramid_set, 0, { video.luma[current].buffer, video.pyramid[current].buffer });
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pyramid.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pyramid.layout, 0, 1, &frame.pyramid_set, 0, nullptr);
        for (uint32_t level = 0; level < layout.levels; ++level)
        {
            PyramidPush push {};
            push.dst_width = layout.width[level];
            push.dst_height = layout.height[level];
            push.dst_offset = layout.offset[level];
            if (level == 0)
            {
                push.mode = 0;
                push.factor = layout.scale;
                push.src_width = width;
                push.src_height = height;
                push.src_base = row_words;
            }
            else
            {
                push.mode = 1;
                push.factor = 2;
                push.src_width = layout.width[level - 1];
                push.src_height = layout.height[level - 1];
                push.src_base = layout.offset[level - 1];
            }
            vkCmdPushConstants(command_buffer, _pyramid.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(command_buffer, (push.dst_width + 15) / 16, (push.dst_height + 15) / 16, 1);
            // the next level / the tracking reads this level
            _barrier(command_buffer);
        }
        _profile_step(command_buffer, 3);
        if (!frame.motion_estimated) return;
        _write_set(frame.lk_set, 0, { video.pyramid[previous].buffer, video.pyramid[current].buffer, frame.motion_points.buffer });
        LkPush push {};
        for (uint32_t level = 0; level < layout.levels; ++level)
        {
            push.level_width[level] = layout.width[level];
            push.level_height[level] = layout.height[level];
            push.level_offset[level] = layout.offset[level];
        }
        push.levels = layout.levels;
        push.grid_x = kMotionPointsX;
        push.grid_y = kMotionPointsY;
        push.iterations = kMotionIterations;
        push.max_residual = kMaxResidual;
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _lk.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _lk.layout, 0, 1, &frame.lk_set, 0, nullptr);
        vkCmdPushConstants(command_buffer, _lk.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(command_buffer, kMotionPoints, 1, 1);
        _profile_step(command_buffer, 4);
    }

    //! profiling: runs what the command buffer holds so far and adds the time to the step
    void _profile_step(VkCommandBuffer command_buffer, int step)
    {
        if (!_profile) return;
        check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
        {
            // without the frames submitted before
            vkQueueWaitIdle(_queue);
            const Timer timer(_step_ms[step]);
            VkSubmitInfo submit { VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command_buffer;
            check(vkResetFences(_device, 1, &_profile_fence), "vkResetFences");
            check(vkQueueSubmit(_queue, 1, &submit, _profile_fence), "vkQueueSubmit");
            check(vkWaitForFences(_device, 1, &_profile_fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        }
        check(vkResetCommandBuffer(command_buffer, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo begin { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(command_buffer, &begin), "vkBeginCommandBuffer");
    }

    //! the tracked points of a frame as the median motion per grid cell, and one by one if `detail` is given
    void _read_motion(const Frame & frame, MotionField & field, std::vector<MotionPoint> * detail)
    {
        field = MotionField();
        field.pixel_scale = static_cast<float>(frame.motion.scale);
        if (detail) detail->clear();
        if (!frame.motion_estimated) return;
        field.estimated = true;
        _binner.clear();
        const float * points = static_cast<const float *>(frame.motion_points.mapped);
        for (int y = 0; y < kMotionPointsY; ++y)
        {
            for (int x = 0; x < kMotionPointsX; ++x)
            {
                const float * point = points + 4 * (y * kMotionPointsX + x);
                // (dx, dy, texture, distance by which the way back misses the point)
                const bool texture = point[2] >= kMotionMinTexture && std::isfinite(point[0]) && std::isfinite(point[1]);
                const bool returns = texture && point[3] <= kMotionMaxBackError;
                if (detail)
                {
                    MotionPoint tracked;
                    tracked.x = (static_cast<float>(x) + 0.5f) * static_cast<float>(frame.motion.width[0]) / kMotionPointsX - 0.5f;
                    tracked.y = (static_cast<float>(y) + 0.5f) * static_cast<float>(frame.motion.height[0]) / kMotionPointsY - 0.5f;
                    tracked.state = returns ? MotionPoint::Used : texture ? MotionPoint::Rejected : MotionPoint::NoTexture;
                    if (texture)
                    {
                        tracked.dx = point[0];
                        tracked.dy = point[1];
                    }
                    detail->push_back(tracked);
                }
                if (!returns) continue;
                _binner.add((static_cast<float>(x) + 0.5f) / kMotionPointsX, (static_cast<float>(y) + 0.5f) / kMotionPointsY, point[0], point[1]);
            }
        }
        _binner.finish(field, kMotionMinCellPoints, field.pixel_scale);
    }

    // ---- motion video

    void _start_encoder(Video & video, const std::string & path, double fps)
    {
        Encoder & encoder = video.encoder;
        const cv::Size size = motion_picture_size(video.width, video.height);
        encoder.width = static_cast<uint32_t>(size.width);
        encoder.height = static_cast<uint32_t>(size.height);
        encoder.row_words = (encoder.width + 3) / 4;
        encoder.picture_bytes = static_cast<VkDeviceSize>(encoder.row_words) * 4 * (encoder.height + encoder.height / 2);
        encoder.fps = fps;
        encoder.path = path;

        encoder.codec = AMediaCodec_createEncoderByType("video/avc");
        if (!encoder.codec) throw std::runtime_error("no H.264 encoder");
        AMediaFormat * format = AMediaFormat_new();
        AMediaFormat_setString(format, AMEDIAFORMAT_KEY_MIME, "video/avc");
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_WIDTH, size.width);
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_HEIGHT, size.height);
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_COLOR_FORMAT, kColorFormatSurface);
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_BIT_RATE,
                              static_cast<int32_t>(kBitsPerPixel * size.width * size.height * 60.0 * std::sqrt(std::max(1.0, fps) / 60.0)));
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_FRAME_RATE, std::max(1, static_cast<int>(std::lround(fps))));
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_I_FRAME_INTERVAL, 1);
        // the pictures are converted with BT.709, limited range (nv12_to_rgba.comp)
        AMediaFormat_setInt32(format, "color-standard", 1);
        AMediaFormat_setInt32(format, "color-range", 2);
        AMediaFormat_setInt32(format, "color-transfer", 3);
        const media_status_t configured = AMediaCodec_configure(encoder.codec, format, nullptr, nullptr, AMEDIACODEC_CONFIGURE_FLAG_ENCODE);
        AMediaFormat_delete(format);
        if (configured != AMEDIA_OK) throw std::runtime_error("the H.264 encoder doesn't take " + std::to_string(size.width) + "x" + std::to_string(size.height));
        if (AMediaCodec_createInputSurface(encoder.codec, &encoder.window) != AMEDIA_OK || !encoder.window)
        {
            throw std::runtime_error("the H.264 encoder has no input surface");
        }
        if (AMediaCodec_start(encoder.codec) != AMEDIA_OK) throw std::runtime_error("the H.264 encoder didn't start");

        encoder.file = open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
        if (encoder.file < 0) throw std::runtime_error("could not open '" + path + "' for writing");
        encoder.muxer = AMediaMuxer_new(encoder.file, AMEDIAMUXER_OUTPUT_FORMAT_MPEG_4);
        if (!encoder.muxer) throw std::runtime_error("no MP4 writer");

        // a swapchain on the input surface of the encoder: what is presented gets encoded
        VkAndroidSurfaceCreateInfoKHR surface_info { VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR };
        surface_info.window = encoder.window;
        check(vkCreateAndroidSurfaceKHR(_instance, &surface_info, nullptr, &encoder.surface), "vkCreateAndroidSurfaceKHR");
        VkBool32 supported = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(_physical, _queue_family, encoder.surface, &supported);
        if (!supported) throw std::runtime_error("the GPU can't present to the encoder");
        VkSurfaceCapabilitiesKHR capabilities {};
        check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(_physical, encoder.surface, &capabilities), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
        if (!(capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) throw std::runtime_error("the surface of the encoder can't be copied to");
        uint32_t count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(_physical, encoder.surface, &count, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(_physical, encoder.surface, &count, formats.data());
        const VkSurfaceFormatKHR * chosen = nullptr;
        for (const VkSurfaceFormatKHR & candidate : formats)
        {
            if (candidate.format == VK_FORMAT_R8G8B8A8_UNORM) chosen = &candidate;
        }
        if (!chosen) throw std::runtime_error("the surface of the encoder takes no RGBA pictures");
        VkSwapchainCreateInfoKHR swapchain_info { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
        swapchain_info.surface = encoder.surface;
        // a picture per draw slot may be on its way to the encoder
        swapchain_info.minImageCount = std::max(capabilities.minImageCount + 1, static_cast<uint32_t>(kDrawSlots));
        if (capabilities.maxImageCount > 0) swapchain_info.minImageCount = std::min(swapchain_info.minImageCount, capabilities.maxImageCount);
        swapchain_info.imageFormat = chosen->format;
        swapchain_info.imageColorSpace = chosen->colorSpace;
        swapchain_info.imageExtent = { encoder.width, encoder.height };
        swapchain_info.imageArrayLayers = 1;
        swapchain_info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        swapchain_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        swapchain_info.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
        swapchain_info.compositeAlpha = (capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) ? VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
                                                                                                                       : VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        swapchain_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        swapchain_info.clipped = VK_FALSE;
        check(vkCreateSwapchainKHR(_device, &swapchain_info, nullptr, &encoder.swapchain), "vkCreateSwapchainKHR");
        vkGetSwapchainImagesKHR(_device, encoder.swapchain, &count, nullptr);
        encoder.images.resize(count);
        vkGetSwapchainImagesKHR(_device, encoder.swapchain, &count, encoder.images.data());
        encoder.active = true;
    }

    //! takes what the encoder has finished and writes it to the file; wait: until the end of its stream
    void _drain(Encoder & encoder, bool wait)
    {
        for (int idle = 0; idle < 2000;)
        {
            AMediaCodecBufferInfo info {};
            const ssize_t index = AMediaCodec_dequeueOutputBuffer(encoder.codec, &info, wait ? 5000 : 0);
            if (index == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED)
            {
                AMediaFormat * format = AMediaCodec_getOutputFormat(encoder.codec);
                encoder.track = AMediaMuxer_addTrack(encoder.muxer, format);
                AMediaFormat_delete(format);
                if (encoder.track < 0 || AMediaMuxer_start(encoder.muxer) != AMEDIA_OK) throw std::runtime_error("could not write '" + encoder.path + "'");
                encoder.muxing = true;
                continue;
            }
            if (index < 0)
            {
                if (!wait) return;
                ++idle;
                continue;
            }
            idle = 0;
            size_t size = 0;
            uint8_t * data = AMediaCodec_getOutputBuffer(encoder.codec, static_cast<size_t>(index), &size);
            const bool config = (info.flags & AMEDIACODEC_BUFFER_FLAG_CODEC_CONFIG) != 0;
            if (data && info.size > 0 && !config && encoder.muxing)
            {
                // the pictures are presented as fast as they are drawn: their place in the video is their number
                info.presentationTimeUs = static_cast<int64_t>(std::llround(static_cast<double>(encoder.written) * 1000000.0 / encoder.fps));
                if (AMediaMuxer_writeSampleData(encoder.muxer, static_cast<size_t>(encoder.track), data, &info) != AMEDIA_OK)
                {
                    AMediaCodec_releaseOutputBuffer(encoder.codec, static_cast<size_t>(index), false);
                    throw std::runtime_error("writing '" + encoder.path + "' failed");
                }
                ++encoder.written;
            }
            const bool end = (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) != 0;
            AMediaCodec_releaseOutputBuffer(encoder.codec, static_cast<size_t>(index), false);
            if (end) return;
        }
        throw std::runtime_error("the H.264 encoder didn't finish");
    }

    //! Draws the scene on the picture a frame left on the GPU and presents the result to the encoder. Doesn't
    //! wait for it: the picture is drawn while the next frames are decoded and analyzed
    void _draw(Video & video, int picture, const MotionVideoScene & scene)
    {
        Encoder & encoder = video.encoder;
        if (!encoder.active || picture < 0 || picture >= static_cast<int>(encoder.pictures.size()))
        {
            throw std::logic_error("motion video: the frame has no picture on the GPU");
        }
        const Timer timer(_video_ms);
        DrawSlot & slot = _draw_slots[_draws % kDrawSlots];
        if (slot.busy)
        {
            const Timer wait_timer(_draw_wait_ms);
            check(vkWaitForFences(_device, 1, &slot.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
            slot.busy = false;
        }
        build_overlay_upload(scene, encoder.width, encoder.height, encoder.row_words, true, _overlay_upload);
        auto upload = [&](Buffer & buffer, const void * data, size_t bytes) {
            _ensure_buffer(buffer, std::max<size_t>(bytes, 64), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
            if (bytes > 0) std::memcpy(buffer.mapped, data, bytes);
        };
        upload(slot.segments, _overlay_upload.segments.data(), _overlay_upload.segments.size() * sizeof(float));
        upload(slot.tiles, _overlay_upload.tiles.data(), _overlay_upload.tiles.size() * sizeof(uint32_t));
        upload(slot.text, _overlay_upload.text.data(), _overlay_upload.text.size());
        _ensure_buffer(slot.drawn, encoder.picture_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
        _ensure_buffer(slot.rgba, static_cast<VkDeviceSize>(encoder.width) * encoder.height * 4,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false);
        _write_set(slot.draw_set, 0, { encoder.pictures[static_cast<size_t>(picture)].buffer, slot.segments.buffer, slot.tiles.buffer,
                                       slot.text.buffer, slot.drawn.buffer });
        _write_set(slot.rgba_set, 0, { slot.drawn.buffer, slot.rgba.buffer });

        uint32_t image_index = 0;
        {
            // the encoder hands a picture back when it has taken one
            const Timer encoder_timer(_encoder_ms);
            check(vkAcquireNextImageKHR(_device, encoder.swapchain, UINT64_MAX, slot.acquired, VK_NULL_HANDLE, &image_index), "vkAcquireNextImageKHR");
        }
        const VkImage image = encoder.images[image_index];
        const VkCommandBuffer command_buffer = slot.command_buffer;

        check(vkResetCommandBuffer(command_buffer, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo begin { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(command_buffer, &begin), "vkBeginCommandBuffer");
        // the picture was written by the submission of its frame
        _barrier(command_buffer);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _overlay_draw.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _overlay_draw.layout, 0, 1, &slot.draw_set, 0, nullptr);
        vkCmdPushConstants(command_buffer, _overlay_draw.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(OverlayDrawPush), &_overlay_upload.push);
        vkCmdDispatch(command_buffer, (encoder.row_words + 7) / 8, (encoder.height / 2 + 7) / 8, 1);
        _barrier(command_buffer);
        const RgbaPush rgba_push { encoder.width, encoder.height, encoder.row_words };
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _rgba.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _rgba.layout, 0, 1, &slot.rgba_set, 0, nullptr);
        vkCmdPushConstants(command_buffer, _rgba.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rgba_push), &rgba_push);
        vkCmdDispatch(command_buffer, (encoder.width + 7) / 8, (encoder.height + 7) / 8, 1);
        _barrier(command_buffer);

        VkImageMemoryBarrier to_copy { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        to_copy.srcAccessMask = 0;
        to_copy.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_copy.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_copy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_copy.image = image;
        to_copy.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_copy);
        VkBufferImageCopy region {};
        region.bufferRowLength = encoder.width;
        region.bufferImageHeight = encoder.height;
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageExtent = { encoder.width, encoder.height, 1 };
        vkCmdCopyBufferToImage(command_buffer, slot.rgba.buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        VkImageMemoryBarrier to_present = to_copy;
        to_present.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_present.dstAccessMask = 0;
        to_present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_present);
        check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

        const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo submit { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &slot.acquired;
        submit.pWaitDstStageMask = &wait_stage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_buffer;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &slot.rendered;
        check(vkResetFences(_device, 1, &slot.fence), "vkResetFences");
        check(vkQueueSubmit(_queue, 1, &submit, slot.fence), "vkQueueSubmit");
        slot.busy = true;
        ++_draws;

        // the time of the picture in the video: the encoder spends its bits by it
        VkPresentTimeGOOGLE time {};
        time.presentID = static_cast<uint32_t>(encoder.presented);
        time.desiredPresentTime = static_cast<uint64_t>(std::llround(static_cast<double>(encoder.presented + 1) * 1e9 / encoder.fps));
        VkPresentTimesInfoGOOGLE times { VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE };
        times.swapchainCount = 1;
        times.pTimes = &time;
        VkPresentInfoKHR present { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        if (_display_timing) present.pNext = &times;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &slot.rendered;
        present.swapchainCount = 1;
        present.pSwapchains = &encoder.swapchain;
        present.pImageIndices = &image_index;
        VkResult presented = VK_SUCCESS;
        {
            const Timer encoder_timer(_encoder_ms);
            presented = vkQueuePresentKHR(_queue, &present);
        }
        if (presented != VK_SUCCESS && presented != VK_SUBOPTIMAL_KHR) check(presented, "vkQueuePresentKHR");
        ++encoder.presented;
        // a later frame may use the picture again: its submission comes after this one
        encoder.free_pictures.push_back(picture);
        _drain(encoder, false);
    }

    //! finish: the video is completed; otherwise everything is just released
    void _stop_encoder(Video & video, bool finish)
    {
        Encoder & encoder = video.encoder;
        if (_device) vkDeviceWaitIdle(_device);
        for (DrawSlot & slot : _draw_slots) slot.busy = false;
        if (finish && encoder.active && encoder.codec)
        {
            AMediaCodec_signalEndOfInputStream(encoder.codec);
            _drain(encoder, true);
            if (encoder.written != encoder.presented)
            {
                std::fprintf(stderr, "trdrop-pp: the encoder wrote %lld of %lld pictures of the motion video\n", encoder.written, encoder.presented);
            }
        }
        encoder.active = false;
        if (encoder.swapchain) vkDestroySwapchainKHR(_device, encoder.swapchain, nullptr);
        if (encoder.surface) vkDestroySurfaceKHR(_instance, encoder.surface, nullptr);
        encoder.swapchain = VK_NULL_HANDLE;
        encoder.surface = VK_NULL_HANDLE;
        encoder.images.clear();
        if (encoder.muxer)
        {
            if (encoder.muxing) AMediaMuxer_stop(encoder.muxer);
            AMediaMuxer_delete(encoder.muxer);
        }
        encoder.muxer = nullptr;
        encoder.muxing = false;
        if (encoder.codec)
        {
            AMediaCodec_stop(encoder.codec);
            AMediaCodec_delete(encoder.codec);
        }
        encoder.codec = nullptr;
        if (encoder.window) ANativeWindow_release(encoder.window);
        encoder.window = nullptr;
        if (encoder.file >= 0) close(encoder.file);
        encoder.file = -1;
        if (_device)
        {
            for (Buffer & buffer : encoder.pictures) _destroy_buffer(buffer);
        }
        encoder.pictures.clear();
        encoder.free_pictures.clear();
    }

    void _release()
    {
        if (_frames_read > 0)
        {
            // where the main thread waited (decoding includes the import, which is listed again on its own)
            const double frames = static_cast<double>(_frames_read);
            std::fprintf(stderr, "gpu pipeline, per frame: waiting for the decoder %.2f ms (of it import %.2f ms), for the GPU %.2f ms,"
                                 " motion video %.2f ms (of it waiting for the GPU %.2f ms, for the encoder %.2f ms)\n",
                         _decode_ms / frames, _import_ms / frames, _wait_ms / frames, _video_ms / frames, _draw_wait_ms / frames, _encoder_ms / frames);
            if (_profile)
            {
                std::fprintf(stderr, "gpu pipeline, steps per frame: reading the hardware buffer %.2f ms, comparison %.2f ms, picture %.2f ms,"
                                     " pyramid %.2f ms, tracking %.2f ms\n",
                             _step_ms[0] / frames, _step_ms[1] / frames, _step_ms[2] / frames, _step_ms[3] / frames, _step_ms[4] / frames);
            }
        }
        if (_device) vkDeviceWaitIdle(_device);
        for (Video & video : _videos) _stop_encoder(video, false);
        for (Slot & slot : _slots)
        {
            for (Frame & frame : slot.frames)
            {
                if (_device)
                {
                    _drop_frame(frame);
                    _destroy_buffer(frame.row_counts);
                    _destroy_buffer(frame.motion_points);
                }
                else if (frame.image)
                {
                    AImage_delete(frame.image);
                    frame.image = nullptr;
                }
            }
            if (_device && slot.fence) vkDestroyFence(_device, slot.fence, nullptr);
            slot.fence = VK_NULL_HANDLE;
        }
        for (DrawSlot & slot : _draw_slots)
        {
            if (!_device) continue;
            for (Buffer * buffer : { &slot.segments, &slot.tiles, &slot.text, &slot.drawn, &slot.rgba }) _destroy_buffer(*buffer);
            if (slot.fence) vkDestroyFence(_device, slot.fence, nullptr);
            if (slot.acquired) vkDestroySemaphore(_device, slot.acquired, nullptr);
            if (slot.rendered) vkDestroySemaphore(_device, slot.rendered, nullptr);
            slot = DrawSlot();
        }
        for (Video & video : _videos)
        {
            if (_device)
            {
                _destroy_compute(video.import);
                if (video.sampler) vkDestroySampler(_device, video.sampler, nullptr);
                if (video.conversion) _fn.vkDestroySamplerYcbcrConversion(_device, video.conversion, nullptr);
                video.sampler = VK_NULL_HANDLE;
                video.conversion = VK_NULL_HANDLE;
                for (Buffer * buffer : { &video.luma[0], &video.luma[1], &video.chroma[0], &video.chroma[1], &video.pyramid[0], &video.pyramid[1] })
                {
                    _destroy_buffer(*buffer);
                }
            }
            av_frame_free(&video.frame);
            // the decoder renders into the window of the reader: it goes first
            video.decoder.reset();
            av_buffer_unref(&video.device);
            if (video.reader) AImageReader_delete(video.reader);
            video.reader = nullptr;
        }
        if (_device)
        {
            for (Compute * compute : { &_diff, &_pyramid, &_lk, &_overlay_base, &_overlay_draw, &_rgba }) _destroy_compute(*compute);
            if (_descriptor_pool) vkDestroyDescriptorPool(_device, _descriptor_pool, nullptr);
            if (_profile_fence) vkDestroyFence(_device, _profile_fence, nullptr);
            if (_command_pool) vkDestroyCommandPool(_device, _command_pool, nullptr);
            vkDestroyDevice(_device, nullptr);
            _device = VK_NULL_HANDLE;
        }
        if (_instance) vkDestroyInstance(_instance, nullptr);
        _instance = VK_NULL_HANDLE;
    }

    VkInstance _instance = VK_NULL_HANDLE;
    VkPhysicalDevice _physical = VK_NULL_HANDLE;
    VkDevice _device = VK_NULL_HANDLE;
    VkQueue _queue = VK_NULL_HANDLE;
    uint32_t _queue_family = 0;
    std::string _device_name;
    VkPhysicalDeviceMemoryProperties _memory_properties {};
    bool _display_timing = false;
    struct {
        PFN_vkCreateSamplerYcbcrConversion vkCreateSamplerYcbcrConversion = nullptr;
        PFN_vkDestroySamplerYcbcrConversion vkDestroySamplerYcbcrConversion = nullptr;
        PFN_vkGetAndroidHardwareBufferPropertiesANDROID vkGetAndroidHardwareBufferPropertiesANDROID = nullptr;
    } _fn;
    VkCommandPool _command_pool = VK_NULL_HANDLE;
    VkFence _profile_fence = VK_NULL_HANDLE;
    VkDescriptorPool _descriptor_pool = VK_NULL_HANDLE;
    Compute _diff;
    Compute _pyramid;
    Compute _lk;
    Compute _overlay_base;
    Compute _overlay_draw;
    Compute _rgba;

    std::vector<Video> _videos;
    std::array<Slot, kSlots> _slots;
    std::array<DrawSlot, kDrawSlots> _draw_slots;
    size_t _submitted = 0;
    size_t _collected = 0;
    size_t _draws = 0;
    bool _first_ready = false;
    bool _motion = false;
    bool _motion_detail = false;
    MotionCellBinner _binner;
    OverlayUpload _overlay_upload;
    long long _frames_read = 0;
    double _decode_ms = 0.0;
    double _import_ms = 0.0;
    double _wait_ms = 0.0;
    double _video_ms = 0.0;
    double _draw_wait_ms = 0.0;
    double _encoder_ms = 0.0;
    bool _profile = false;
    double _step_ms[5] = {};
};

} // namespace

std::unique_ptr<FramePipeline> make_mediacodec_vulkan_pipeline(const std::vector<std::string> & paths)
{
    return std::make_unique<MediaCodecVulkanPipeline>(paths);
}

} // namespace trdrop
