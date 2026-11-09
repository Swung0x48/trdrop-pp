// Vulkan Video pipeline: FFmpeg decodes on the GPU (Vulkan hwaccel), the luma (Y) plane of every
// decoded image is copied into a buffer on the GPU and compared with the previous one by
// luma_diff.comp. Only the per row counts are read back, the frames never leave the GPU.
//
// The Vulkan objects live on FFmpeg's VkDevice. Every decoded image is synchronized through its
// timeline semaphore (AVVkFrame::sem / sem_value), as required by FFmpeg.

#include "frame_pipeline.h"

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/pixdesc.h>
#include <libavutil/version.h>
}

#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "ffmpeg/ffmpeg_decoder.h"

#if LIBAVUTIL_VERSION_MAJOR < 60
#error "the Vulkan Video pipeline needs FFmpeg 8.0 or newer"
#endif

namespace trdrop {

namespace {

const uint32_t kLumaDiffSpirv[] = {
#include "luma_diff.spv.inc"
};

//! sets of frames in flight
constexpr size_t kSlots = 3;

struct PushConstants {
    uint32_t width;
    uint32_t row_words;
    uint32_t threshold;
};

#define TRDROP_VKV_DEVICE_FUNCTIONS(X) \
    X(vkGetDeviceQueue) X(vkGetDeviceQueue2) \
    X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkResetCommandBuffer) \
    X(vkCmdPipelineBarrier2) X(vkCmdCopyImageToBuffer) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) \
    X(vkCmdPushConstants) X(vkCmdDispatch) X(vkQueueSubmit2) \
    X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences) \
    X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) X(vkMapMemory) X(vkUnmapMemory) \
    X(vkInvalidateMappedMemoryRanges) \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) \
    X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) \
    X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) X(vkCreateComputePipelines) X(vkDestroyPipeline) \
    X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) \
    X(vkDeviceWaitIdle)

void check(VkResult result, const char * what)
{
    if (result != VK_SUCCESS)
    {
        throw std::runtime_error(std::string(what) + " failed (VkResult " + std::to_string(static_cast<int>(result)) + ")");
    }
}

class VulkanVideoPipeline final : public FramePipeline {
public:
    VulkanVideoPipeline(const std::vector<std::string> & paths, int device_index)
        : _videos(paths.size())
    {
        try
        {
            // cheap checks first, creating the device and decoding a frame takes a while
            for (const std::string & path : paths)
            {
                const std::string reason = ffmpeg::vulkan_decode_blocker(path);
                if (!reason.empty()) throw std::runtime_error(reason);
            }
            _device_index = resolve_vulkan_device(device_index, _device_name);
            const int created = av_hwdevice_ctx_create(&_hw_device, AV_HWDEVICE_TYPE_VULKAN,
                                                       std::to_string(_device_index).c_str(), nullptr, 0);
            if (created < 0) throw std::runtime_error("FFmpeg could not create a Vulkan device (" + ffmpeg::error_string(created) + ")");
            _load_device();
            _create_pipeline();
            _create_slots();
            for (size_t i = 0; i < paths.size(); ++i)
            {
                _videos[i].decoder = std::make_unique<ffmpeg::VideoDecoder>(paths[i], _hw_device, static_cast<int>(kSlots) + 1);
                _videos[i].first_frame = av_frame_alloc();
                if (!_videos[i].first_frame) throw std::runtime_error("out of memory");
            }
            // decode the first frames: checks that the GPU decodes them into a supported format,
            // they are the first submission
            for (size_t i = 0; i < paths.size(); ++i)
            {
                if (!_videos[i].decoder->next(_videos[i].first_frame)) throw std::runtime_error("'" + paths[i] + "' contains no frames");
                _check_frame(_videos[i].first_frame, *_videos[i].decoder);
            }
        }
        catch (...)
        {
            _release();
            throw;
        }
    }

    ~VulkanVideoPipeline() override { _release(); }

    VulkanVideoPipeline(const VulkanVideoPipeline &) = delete;
    VulkanVideoPipeline & operator=(const VulkanVideoPipeline &) = delete;

    std::string description() const override
    {
        return "vulkan video decode + analysis (" + _device_name + ", device " + std::to_string(_device_index) + "), luma";
    }

    bool submit_next(const std::vector<VideoOptions> & video_options) override
    {
        Slot & slot = _slots[_submitted % kSlots];
        if (slot.busy) throw std::logic_error("vulkan video pipeline: submit_next() called while the slot is in flight");

        // decode the next frame of every video into the slot
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            VideoState & video = _videos[i];
            AVFrame * frame = slot.videos[i].frame;
            if (video.first_frame)
            {
                av_frame_move_ref(frame, video.first_frame);
                av_frame_free(&video.first_frame);
            }
            else if (!video.decoder->next(frame))
            {
                for (SlotVideo & v : slot.videos) av_frame_unref(v.frame);
                return false;
            }
            _check_frame(frame, *video.decoder);
        }

        // buffers / descriptors per video
        std::array<PushConstants, kMaxVideoCount> parameters {};
        std::array<VkImageMemoryBarrier2, kMaxVideoCount> image_barriers {};
        std::array<VkSemaphoreSubmitInfo, kMaxVideoCount> waits {};
        std::array<VkSemaphoreSubmitInfo, kMaxVideoCount> signals {};
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            VideoState & video = _videos[i];
            SlotVideo & slot_video = slot.videos[i];
            const AVFrame * frame = slot_video.frame;
            const int width = frame->width;
            const int height = frame->height;
            if (static_cast<uint32_t>(height) > _max_workgroups) throw std::runtime_error("frame height exceeds the device limit");

            const uint32_t row_words = static_cast<uint32_t>((width + 3) / 4);
            const VkDeviceSize luma_bytes = static_cast<VkDeviceSize>(row_words) * 4 * static_cast<VkDeviceSize>(height);
            const bool compare = video.has_previous && video.width == width && video.height == height;
            if (video.luma[0].size < luma_bytes)
            {
                // the buffers may still be used by submissions in flight
                check(_fn.vkDeviceWaitIdle(_device), "vkDeviceWaitIdle");
                for (Buffer & luma : video.luma)
                {
                    _destroy_buffer(luma);
                    luma = _create_buffer(luma_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                          { VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0 });
                }
            }
            const VkDeviceSize row_bytes = static_cast<VkDeviceSize>(height) * sizeof(uint32_t);
            if (slot_video.row_counts.size < row_bytes)
            {
                _destroy_buffer(slot_video.row_counts);
                slot_video.row_counts = _create_buffer(row_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, {
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                });
            }
            video.width = width;
            video.height = height;
            slot_video.luma = &video.luma[video.current];

            const std::array<VkDescriptorBufferInfo, 3> buffers = {{
                { video.luma[video.current].buffer, 0, VK_WHOLE_SIZE },
                { video.luma[1 - video.current].buffer, 0, VK_WHOLE_SIZE },
                { slot_video.row_counts.buffer, 0, VK_WHOLE_SIZE },
            }};
            std::array<VkWriteDescriptorSet, 3> writes {};
            for (uint32_t b = 0; b < 3; ++b)
            {
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = slot_video.descriptor_set;
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1;
                writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[b].pBufferInfo = &buffers[b];
            }
            _fn.vkUpdateDescriptorSets(_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

            parameters[i] = { static_cast<uint32_t>(width), row_words, static_cast<uint32_t>(std::max(0, video_options[i].pixel_difference)) };
            slot_video.compared = compare;
            slot_video.height = height;
            slot_video.width = width;
        }

        // lock the decoded images while their state is read and updated
        for (SlotVideo & slot_video : slot.videos) _lock(slot_video.frame, true);

        check(_fn.vkResetCommandBuffer(slot.command_buffer, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo begin { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(_fn.vkBeginCommandBuffer(slot.command_buffer, &begin), "vkBeginCommandBuffer");

        // earlier submissions read / wrote the luma buffers, the decoded images become copy sources
        VkMemoryBarrier2 before { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        before.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
        before.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        before.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        before.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            const AVVkFrame * vkf = reinterpret_cast<const AVVkFrame *>(slot.videos[i].frame->data[0]);
            VkImageMemoryBarrier2 & barrier = image_barriers[i];
            barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            barrier.srcAccessMask = vkf->access[0];
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            barrier.oldLayout = vkf->layout[0];
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcQueueFamilyIndex = vkf->queue_family[0];
            barrier.dstQueueFamilyIndex = vkf->queue_family[0] == VK_QUEUE_FAMILY_IGNORED ? VK_QUEUE_FAMILY_IGNORED : _queue_family;
            barrier.image = vkf->img[0];
            barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        }
        VkDependencyInfo dependency { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &before;
        dependency.imageMemoryBarrierCount = static_cast<uint32_t>(_videos.size());
        dependency.pImageMemoryBarriers = image_barriers.data();
        _fn.vkCmdPipelineBarrier2(slot.command_buffer, &dependency);

        for (size_t i = 0; i < _videos.size(); ++i)
        {
            const AVFrame * frame = slot.videos[i].frame;
            const AVVkFrame * vkf = reinterpret_cast<const AVVkFrame *>(frame->data[0]);
            // one VkImage per frame (multiplane format) or one per plane
            const bool multiplane = vkf->img[1] == VK_NULL_HANDLE;
            VkBufferImageCopy region {};
            region.bufferRowLength = parameters[i].row_words * 4;
            region.bufferImageHeight = static_cast<uint32_t>(frame->height);
            region.imageSubresource = { static_cast<VkImageAspectFlags>(multiplane ? VK_IMAGE_ASPECT_PLANE_0_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1 };
            // hardware frames keep the top / left cropping in the frame properties
            region.imageOffset = { static_cast<int32_t>(frame->crop_left), static_cast<int32_t>(frame->crop_top), 0 };
            region.imageExtent = { static_cast<uint32_t>(frame->width), static_cast<uint32_t>(frame->height), 1 };
            _fn.vkCmdCopyImageToBuffer(slot.command_buffer, vkf->img[0], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       slot.videos[i].luma->buffer, 1, &region);
        }

        VkMemoryBarrier2 copied { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        copied.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        copied.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        copied.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        copied.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo copied_dependency { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        copied_dependency.memoryBarrierCount = 1;
        copied_dependency.pMemoryBarriers = &copied;
        _fn.vkCmdPipelineBarrier2(slot.command_buffer, &copied_dependency);

        _fn.vkCmdBindPipeline(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            if (!slot.videos[i].compared) continue;
            _fn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline_layout,
                                        0, 1, &slot.videos[i].descriptor_set, 0, nullptr);
            _fn.vkCmdPushConstants(slot.command_buffer, _pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &parameters[i]);
            _fn.vkCmdDispatch(slot.command_buffer, static_cast<uint32_t>(slot.videos[i].height), 1, 1);
        }

        VkMemoryBarrier2 host { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        host.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        host.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        host.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        host.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        VkDependencyInfo host_dependency { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        host_dependency.memoryBarrierCount = 1;
        host_dependency.pMemoryBarriers = &host;
        _fn.vkCmdPipelineBarrier2(slot.command_buffer, &host_dependency);
        check(_fn.vkEndCommandBuffer(slot.command_buffer), "vkEndCommandBuffer");

        // wait for the decoder's work on each image and signal our use of it
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            const AVVkFrame * vkf = reinterpret_cast<const AVVkFrame *>(slot.videos[i].frame->data[0]);
            waits[i] = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
            waits[i].semaphore = vkf->sem[0];
            waits[i].value = vkf->sem_value[0];
            waits[i].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            signals[i] = waits[i];
            signals[i].value = vkf->sem_value[0] + 1;
        }
        VkCommandBufferSubmitInfo command_info { VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        command_info.commandBuffer = slot.command_buffer;
        VkSubmitInfo2 submit { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submit.waitSemaphoreInfoCount = static_cast<uint32_t>(_videos.size());
        submit.pWaitSemaphoreInfos = waits.data();
        submit.commandBufferInfoCount = 1;
        submit.pCommandBufferInfos = &command_info;
        submit.signalSemaphoreInfoCount = static_cast<uint32_t>(_videos.size());
        submit.pSignalSemaphoreInfos = signals.data();

        VkResult result = _fn.vkResetFences(_device, 1, &slot.fence);
        if (result == VK_SUCCESS)
        {
            _lock_queue(true);
            result = _fn.vkQueueSubmit2(_queue, 1, &submit, slot.fence);
            _lock_queue(false);
        }
        if (result == VK_SUCCESS)
        {
            for (size_t i = 0; i < _videos.size(); ++i)
            {
                AVVkFrame * vkf = reinterpret_cast<AVVkFrame *>(slot.videos[i].frame->data[0]);
                vkf->sem_value[0] += 1;
                vkf->layout[0] = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                vkf->access[0] = VK_ACCESS_2_TRANSFER_READ_BIT;
                vkf->queue_family[0] = image_barriers[i].dstQueueFamilyIndex;
            }
        }
        for (SlotVideo & slot_video : slot.videos) _lock(slot_video.frame, false);
        check(result, "vkQueueSubmit2");

        slot.busy = true;
        ++_submitted;
        ++_frames_read;
        for (VideoState & video : _videos)
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
        if (!slot.busy) throw std::logic_error("vulkan video pipeline: nothing to collect");
        check(_fn.vkWaitForFences(_device, 1, &slot.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        out.resize(_videos.size());
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            SlotVideo & slot_video = slot.videos[i];
            RowCounts & result = out[i];
            result.compared = slot_video.compared;
            result.cols = slot_video.width;
            result.counts.clear();
            if (slot_video.compared)
            {
                if (!slot_video.row_counts.coherent)
                {
                    VkMappedMemoryRange range { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
                    range.memory = slot_video.row_counts.memory;
                    range.size = VK_WHOLE_SIZE;
                    check(_fn.vkInvalidateMappedMemoryRanges(_device, 1, &range), "vkInvalidateMappedMemoryRanges");
                }
                const uint32_t * counts = static_cast<const uint32_t *>(slot_video.row_counts.mapped);
                result.counts.assign(counts, counts + slot_video.height);
            }
            // the decoded image can be reused by the decoder
            av_frame_unref(slot_video.frame);
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

private:
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        void * mapped = nullptr;
        bool coherent = true;
    };

    struct VideoState {
        std::unique_ptr<ffmpeg::VideoDecoder> decoder;
        //! decoded while probing, the first submission
        AVFrame * first_frame = nullptr;
        //! luma ping pong buffers
        Buffer luma[2];
        int current = 0;
        int width = 0;
        int height = 0;
        bool has_previous = false;
    };

    struct SlotVideo {
        AVFrame * frame = nullptr;
        Buffer row_counts;
        //! luma buffer the frame is copied to
        const Buffer * luma = nullptr;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        bool compared = false;
        int width = 0;
        int height = 0;
    };

    struct Slot {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool busy = false;
        std::vector<SlotVideo> videos;
    };

    static void _check_frame(const AVFrame * frame, const ffmpeg::VideoDecoder & decoder)
    {
        if (frame->format != AV_PIX_FMT_VULKAN || !frame->hw_frames_ctx) throw std::runtime_error("no Vulkan decoding for " + decoder.codec_name());
        const AVHWFramesContext * frames = reinterpret_cast<const AVHWFramesContext *>(frame->hw_frames_ctx->data);
        if (!ffmpeg::is_luma_format(frames->sw_format))
        {
            const char * name = av_get_pix_fmt_name(frames->sw_format);
            throw std::runtime_error(std::string("pixel format ") + (name ? name : "unknown") + " has no 8-bit luma plane");
        }
    }

    void _lock(AVFrame * frame, bool lock)
    {
        AVHWFramesContext * frames = reinterpret_cast<AVHWFramesContext *>(frame->hw_frames_ctx->data);
        AVVulkanFramesContext * vk_frames = static_cast<AVVulkanFramesContext *>(frames->hwctx);
        AVVkFrame * vkf = reinterpret_cast<AVVkFrame *>(frame->data[0]);
        if (lock) vk_frames->lock_frame(frames, vkf);
        else vk_frames->unlock_frame(frames, vkf);
    }

    void _lock_queue(bool lock)
    {
#if !defined(FF_API_VULKAN_SYNC_QUEUES) || FF_API_VULKAN_SYNC_QUEUES
        // deprecated in newer FFmpeg versions (internally synchronized queues), required by older ones
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        AVHWDeviceContext * device = reinterpret_cast<AVHWDeviceContext *>(_hw_device->data);
        if (lock && _vk->lock_queue) _vk->lock_queue(device, _queue_family, 0);
        if (!lock && _vk->unlock_queue) _vk->unlock_queue(device, _queue_family, 0);
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#else
        (void)lock;
#endif
    }

    void _load_device()
    {
        AVHWDeviceContext * device = reinterpret_cast<AVHWDeviceContext *>(_hw_device->data);
        _vk = static_cast<AVVulkanDeviceContext *>(device->hwctx);
        _device = _vk->act_dev;
        PFN_vkGetInstanceProcAddr get_instance_proc_addr = _vk->get_proc_addr;
        if (!get_instance_proc_addr) throw std::runtime_error("FFmpeg's Vulkan device has no vkGetInstanceProcAddr");
        auto get_device_proc_addr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(get_instance_proc_addr(_vk->inst, "vkGetDeviceProcAddr"));
        auto get_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(get_instance_proc_addr(_vk->inst, "vkGetPhysicalDeviceProperties"));
        auto get_memory_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
            get_instance_proc_addr(_vk->inst, "vkGetPhysicalDeviceMemoryProperties"));
        if (!get_device_proc_addr || !get_properties || !get_memory_properties) throw std::runtime_error("missing Vulkan instance functions");
#define TRDROP_VKV_LOAD(name) \
        _fn.name = reinterpret_cast<PFN_##name>(get_device_proc_addr(_device, #name)); \
        if (!_fn.name) throw std::runtime_error("missing Vulkan function " #name);
        TRDROP_VKV_DEVICE_FUNCTIONS(TRDROP_VKV_LOAD)
#undef TRDROP_VKV_LOAD

        VkPhysicalDeviceProperties properties;
        get_properties(_vk->phys_dev, &properties);
        if (_device_name != properties.deviceName)
        {
            std::fprintf(stderr, "trdrop-pp: FFmpeg opened '%s' as Vulkan device %d (expected '%s')\n",
                         properties.deviceName, _device_index, _device_name.c_str());
            _device_name = properties.deviceName;
        }
        _max_workgroups = properties.limits.maxComputeWorkGroupCount[0];
        get_memory_properties(_vk->phys_dev, &_memory_properties);

        // a compute queue of the ones FFmpeg created
        bool found = false;
        for (int i = 0; i < _vk->nb_qf && !found; ++i)
        {
            if (_vk->qf[i].flags & VK_QUEUE_COMPUTE_BIT)
            {
                _queue_family = static_cast<uint32_t>(_vk->qf[i].idx);
                found = true;
            }
        }
        if (!found) throw std::runtime_error("FFmpeg's Vulkan device has no compute queue");
#if LIBAVUTIL_VERSION_MAJOR >= 61
        VkDeviceQueueInfo2 queue_info { VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2 };
        queue_info.flags = _vk->queue_flags;
        queue_info.queueFamilyIndex = _queue_family;
        queue_info.queueIndex = 0;
        _fn.vkGetDeviceQueue2(_device, &queue_info, &_queue);
#else
        _fn.vkGetDeviceQueue(_device, _queue_family, 0, &_queue);
#endif
    }

    void _create_pipeline()
    {
        std::array<VkDescriptorSetLayoutBinding, 3> bindings {};
        for (uint32_t b = 0; b < 3; ++b)
        {
            bindings[b].binding = b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo set_layout_info { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        set_layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        set_layout_info.pBindings = bindings.data();
        check(_fn.vkCreateDescriptorSetLayout(_device, &set_layout_info, nullptr, &_set_layout), "vkCreateDescriptorSetLayout");

        VkPushConstantRange push_range { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants) };
        VkPipelineLayoutCreateInfo layout_info { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &_set_layout;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push_range;
        check(_fn.vkCreatePipelineLayout(_device, &layout_info, nullptr, &_pipeline_layout), "vkCreatePipelineLayout");

        VkShaderModuleCreateInfo module_info { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        module_info.codeSize = sizeof(kLumaDiffSpirv);
        module_info.pCode = kLumaDiffSpirv;
        VkShaderModule module = VK_NULL_HANDLE;
        check(_fn.vkCreateShaderModule(_device, &module_info, nullptr, &module), "vkCreateShaderModule");
        VkComputePipelineCreateInfo pipeline_info { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = _pipeline_layout;
        const VkResult result = _fn.vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &_pipeline);
        _fn.vkDestroyShaderModule(_device, module, nullptr);
        check(result, "vkCreateComputePipelines");
    }

    void _create_slots()
    {
        const uint32_t set_count = static_cast<uint32_t>(kSlots * _videos.size());
        VkDescriptorPoolSize pool_size { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * set_count };
        VkDescriptorPoolCreateInfo pool_info { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        pool_info.maxSets = set_count;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        check(_fn.vkCreateDescriptorPool(_device, &pool_info, nullptr, &_descriptor_pool), "vkCreateDescriptorPool");

        VkCommandPoolCreateInfo command_pool_info { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        command_pool_info.queueFamilyIndex = _queue_family;
        check(_fn.vkCreateCommandPool(_device, &command_pool_info, nullptr, &_command_pool), "vkCreateCommandPool");

        for (Slot & slot : _slots)
        {
            slot.videos.resize(_videos.size());
            VkCommandBufferAllocateInfo command_info { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            command_info.commandPool = _command_pool;
            command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            command_info.commandBufferCount = 1;
            check(_fn.vkAllocateCommandBuffers(_device, &command_info, &slot.command_buffer), "vkAllocateCommandBuffers");
            VkFenceCreateInfo fence_info { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
            check(_fn.vkCreateFence(_device, &fence_info, nullptr, &slot.fence), "vkCreateFence");
            for (SlotVideo & video : slot.videos)
            {
                video.frame = av_frame_alloc();
                if (!video.frame) throw std::runtime_error("out of memory");
                VkDescriptorSetAllocateInfo set_info { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
                set_info.descriptorPool = _descriptor_pool;
                set_info.descriptorSetCount = 1;
                set_info.pSetLayouts = &_set_layout;
                check(_fn.vkAllocateDescriptorSets(_device, &set_info, &video.descriptor_set), "vkAllocateDescriptorSets");
            }
        }
    }

    Buffer _create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, std::initializer_list<VkMemoryPropertyFlags> preferred)
    {
        Buffer result;
        result.size = size;
        VkBufferCreateInfo buffer_info { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        buffer_info.size = size;
        buffer_info.usage = usage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(_fn.vkCreateBuffer(_device, &buffer_info, nullptr, &result.buffer), "vkCreateBuffer");
        try
        {
            VkMemoryRequirements requirements;
            _fn.vkGetBufferMemoryRequirements(_device, result.buffer, &requirements);
            uint32_t type_index = UINT32_MAX;
            for (VkMemoryPropertyFlags flags : preferred)
            {
                for (uint32_t i = 0; i < _memory_properties.memoryTypeCount && type_index == UINT32_MAX; ++i)
                {
                    if ((requirements.memoryTypeBits & (1u << i)) == 0) continue;
                    if ((_memory_properties.memoryTypes[i].propertyFlags & flags) == flags) type_index = i;
                }
                if (type_index != UINT32_MAX) break;
            }
            if (type_index == UINT32_MAX) throw std::runtime_error("vulkan video pipeline: no suitable memory type");
            VkMemoryAllocateInfo allocate_info { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
            allocate_info.allocationSize = requirements.size;
            allocate_info.memoryTypeIndex = type_index;
            check(_fn.vkAllocateMemory(_device, &allocate_info, nullptr, &result.memory), "vkAllocateMemory");
            check(_fn.vkBindBufferMemory(_device, result.buffer, result.memory, 0), "vkBindBufferMemory");
            const VkMemoryPropertyFlags properties = _memory_properties.memoryTypes[type_index].propertyFlags;
            result.coherent = (properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
            if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            {
                check(_fn.vkMapMemory(_device, result.memory, 0, VK_WHOLE_SIZE, 0, &result.mapped), "vkMapMemory");
            }
        }
        catch (...)
        {
            _destroy_buffer(result);
            throw;
        }
        return result;
    }

    void _destroy_buffer(Buffer & buffer)
    {
        if (buffer.mapped) _fn.vkUnmapMemory(_device, buffer.memory);
        if (buffer.buffer) _fn.vkDestroyBuffer(_device, buffer.buffer, nullptr);
        if (buffer.memory) _fn.vkFreeMemory(_device, buffer.memory, nullptr);
        buffer = Buffer();
    }

    void _release()
    {
        // our Vulkan objects first, then the frames and decoders, the device last
        if (_device && _fn.vkDeviceWaitIdle)
        {
            _fn.vkDeviceWaitIdle(_device);
            for (Slot & slot : _slots)
            {
                for (SlotVideo & video : slot.videos) _destroy_buffer(video.row_counts);
                if (slot.fence) _fn.vkDestroyFence(_device, slot.fence, nullptr);
                slot.fence = VK_NULL_HANDLE;
            }
            for (VideoState & video : _videos)
            {
                _destroy_buffer(video.luma[0]);
                _destroy_buffer(video.luma[1]);
            }
            if (_command_pool) _fn.vkDestroyCommandPool(_device, _command_pool, nullptr);
            if (_descriptor_pool) _fn.vkDestroyDescriptorPool(_device, _descriptor_pool, nullptr);
            if (_pipeline) _fn.vkDestroyPipeline(_device, _pipeline, nullptr);
            if (_pipeline_layout) _fn.vkDestroyPipelineLayout(_device, _pipeline_layout, nullptr);
            if (_set_layout) _fn.vkDestroyDescriptorSetLayout(_device, _set_layout, nullptr);
            _command_pool = VK_NULL_HANDLE;
            _descriptor_pool = VK_NULL_HANDLE;
            _pipeline = VK_NULL_HANDLE;
            _pipeline_layout = VK_NULL_HANDLE;
            _set_layout = VK_NULL_HANDLE;
        }
        for (Slot & slot : _slots)
        {
            for (SlotVideo & video : slot.videos) av_frame_free(&video.frame);
        }
        for (VideoState & video : _videos)
        {
            av_frame_free(&video.first_frame);
            video.decoder.reset();
        }
        av_buffer_unref(&_hw_device);
        _device = VK_NULL_HANDLE;
    }

private:
    AVBufferRef * _hw_device = nullptr;
    AVVulkanDeviceContext * _vk = nullptr;
    VkDevice _device = VK_NULL_HANDLE;
    VkQueue _queue = VK_NULL_HANDLE;
    uint32_t _queue_family = 0;
    int _device_index = 0;
    std::string _device_name;
    uint32_t _max_workgroups = 65535;
    VkPhysicalDeviceMemoryProperties _memory_properties {};
    struct {
#define TRDROP_VKV_DECLARE(name) PFN_##name name = nullptr;
        TRDROP_VKV_DEVICE_FUNCTIONS(TRDROP_VKV_DECLARE)
#undef TRDROP_VKV_DECLARE
    } _fn;

    VkDescriptorSetLayout _set_layout = VK_NULL_HANDLE;
    VkPipelineLayout _pipeline_layout = VK_NULL_HANDLE;
    VkPipeline _pipeline = VK_NULL_HANDLE;
    VkDescriptorPool _descriptor_pool = VK_NULL_HANDLE;
    VkCommandPool _command_pool = VK_NULL_HANDLE;

    std::vector<VideoState> _videos;
    std::array<Slot, kSlots> _slots;
    size_t _submitted = 0;
    size_t _collected = 0;
    long long _frames_read = 0;
};

} // namespace

std::unique_ptr<FramePipeline> make_vulkan_video_pipeline(const std::vector<std::string> & paths, int device_index)
{
    return std::make_unique<VulkanVideoPipeline>(paths, device_index);
}

} // namespace trdrop
