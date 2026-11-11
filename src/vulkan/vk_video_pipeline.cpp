// Vulkan Video pipeline: FFmpeg decodes on the GPU (Vulkan hwaccel), the Y, U and V planes of every
// decoded image are copied into buffers on the GPU and compared with the previous ones by
// yuv_diff.comp. Only the per row counts are read back, the frames never leave the GPU - unless the motion
// video is written and can't be drawn on the GPU, then the planes are also copied into host memory.
//
// Motion video on the GPU: overlay_base.comp scales every frame down into a picture that stays on the GPU
// until the analysis has its values (it looks ahead), overlay_draw.comp then draws the scene on it, the
// result is copied into a frame of FFmpeg's Vulkan H.264 encoder. Only the encoded packets come back.
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
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "ffmpeg/ffmpeg_decoder.h"
#include "ffmpeg/ffmpeg_encoder.h"
#include "vulkan/vk_motion_shared.h"
#include "ffmpeg/yuv_picture.h"

#if LIBAVUTIL_VERSION_MAJOR < 60
#error "the Vulkan Video pipeline needs FFmpeg 8.0 or newer"
#endif

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

const uint32_t kMotionSelectSpirv[] = {
#include "motion_select.spv.inc"
};

const uint32_t kMotionTextureSpirv[] = {
#include "motion_texture.spv.inc"
};

const uint32_t kOverlayBaseSpirv[] = {
#include "overlay_base.spv.inc"
};

const uint32_t kOverlayDrawSpirv[] = {
#include "overlay_draw.spv.inc"
};

//! motion video frames being drawn at the same time
constexpr size_t kDrawSlots = 3;

//! sets of frames in flight
constexpr size_t kSlots = 3;

constexpr uint32_t kBindings = 5;

uint32_t align4(uint32_t value) { return (value + 3u) & ~3u; }

//! the VkImages of a frame: one multiplane image or one image per plane
int image_count(const AVVkFrame * vkf)
{
    int count = 0;
    while (count < AV_NUM_DATA_POINTERS && vkf->img[count] != VK_NULL_HANDLE) ++count;
    return count;
}

#define TRDROP_VKV_DEVICE_FUNCTIONS(X) \
    X(vkGetDeviceQueue) X(vkGetDeviceQueue2) \
    X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkResetCommandBuffer) \
    X(vkCmdPipelineBarrier2) X(vkCmdCopyImageToBuffer) X(vkCmdCopyBufferToImage) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) \
    X(vkCmdPushConstants) X(vkCmdDispatch) X(vkCmdDispatchIndirect) X(vkQueueSubmit2) \
    X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences) \
    X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) X(vkMapMemory) X(vkUnmapMemory) \
    X(vkInvalidateMappedMemoryRanges) X(vkFlushMappedMemoryRanges) \
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
            _pyramid = _create_compute(kMotionPyramidSpirv, sizeof(kMotionPyramidSpirv), 2, sizeof(PyramidPush));
            _lk = _create_compute(kMotionLkSpirv, sizeof(kMotionLkSpirv), 4, sizeof(LkPush));
            _select = _create_compute(kMotionSelectSpirv, sizeof(kMotionSelectSpirv), 3, sizeof(SelectPush));
            _texture = _create_compute(kMotionTextureSpirv, sizeof(kMotionTextureSpirv), 4, sizeof(TexturePush));
            _overlay_base = _create_compute(kOverlayBaseSpirv, sizeof(kOverlayBaseSpirv), 3, sizeof(OverlayBasePush));
            _overlay_draw = _create_compute(kOverlayDrawSpirv, sizeof(kOverlayDrawSpirv), 5, sizeof(OverlayDrawPush));
            _gpu_videos.resize(paths.size());
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
        return "vulkan video decode + analysis (" + _device_name + ", device " + std::to_string(_device_index) + "), yuv"
             + (_motion ? ", motion on the GPU" : "");
    }

    void enable_motion() override
    {
        if (_max_group_size < kMotionGroupSize || _max_group_invocations < kMotionGroupSize)
        {
            throw std::runtime_error("the Vulkan device '" + _device_name + "' can't run the motion analysis (workgroups of "
                                   + std::to_string(kMotionGroupSize) + " needed), use --decode ffmpeg");
        }
        _motion = true;
    }

    void enable_motion_detail() override { _motion_detail = true; }

    std::unique_ptr<MotionVideoTarget> start_motion_video(size_t index, const std::string & path, double fps, std::string & reason) override
    {
        GpuVideo & gpu = _gpu_videos.at(index);
        try
        {
            const AVFrame * frame = _videos.at(index).first_frame;
            if (!frame) throw std::logic_error("the motion video has to be started before the first frame");            const cv::Size size = motion_picture_size(frame->width, frame->height);
            // frames for the encoder on the device the frames are decoded on
            gpu.frames = av_hwframe_ctx_alloc(_hw_device);
            if (!gpu.frames) throw std::runtime_error("out of memory");
            AVHWFramesContext * frames = reinterpret_cast<AVHWFramesContext *>(gpu.frames->data);
            frames->format = AV_PIX_FMT_VULKAN;
            frames->sw_format = AV_PIX_FMT_NV12;
            frames->width = size.width;
            frames->height = size.height;
            const int result = av_hwframe_ctx_init(gpu.frames);
            if (result < 0) throw std::runtime_error("no Vulkan frames for the video encoder: " + ffmpeg::error_string(result));
            gpu.encoder = std::make_unique<ffmpeg::EncodedVideo>(path, size.width, size.height, fps, "h264_vulkan", gpu.frames);
            gpu.width = static_cast<uint32_t>(size.width);
            gpu.height = static_cast<uint32_t>(size.height);
            gpu.row_words = (gpu.width + 3) / 4;
            gpu.picture_bytes = static_cast<VkDeviceSize>(gpu.row_words) * 4 * (gpu.height + gpu.height / 2);
            gpu.active = true;
            return std::make_unique<GpuTarget>(*this, index);
        }
        catch (const std::exception & e)
        {
            // e.g. the GPU can't encode H.264 with Vulkan Video
            gpu.encoder.reset();
            av_buffer_unref(&gpu.frames);
            reason = e.what();
            return nullptr;
        }
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
        bool reallocate = false;
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            const AVFrame * frame = slot.videos[i].frame;
            const Plan plan = _plan(frame);
            reallocate |= _videos[i].luma[0].size < plan.luma_bytes || _videos[i].chroma[0].size < plan.chroma_bytes;
            if (_motion)
            {
                const MotionLayout layout = motion_layout(frame->width, frame->height);
                reallocate |= _videos[i].pyramid[0].size < layout.bytes || _videos[i].products.size < motion_product_bytes(layout);
            }
        }
        // the buffers may still be used by submissions in flight
        if (reallocate) check(_fn.vkDeviceWaitIdle(_device), "vkDeviceWaitIdle");

        for (size_t i = 0; i < _videos.size(); ++i)
        {
            VideoState & video = _videos[i];
            SlotVideo & slot_video = slot.videos[i];
            const AVFrame * frame = slot_video.frame;
            const Plan plan = _plan(frame);
            if (static_cast<uint32_t>(frame->height) > _max_workgroups) throw std::runtime_error("frame height exceeds the device limit");

            const bool compare = video.has_previous && video.width == frame->width && video.height == frame->height
                              && video.format == plan.format;
            for (int b = 0; b < 2; ++b)
            {
                if (video.luma[b].size < plan.luma_bytes)
                {
                    _destroy_buffer(video.luma[b]);
                    video.luma[b] = _create_buffer(plan.luma_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                   { VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0 });
                }
                if (video.chroma[b].size < plan.chroma_bytes)
                {
                    _destroy_buffer(video.chroma[b]);
                    video.chroma[b] = _create_buffer(plan.chroma_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                     { VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0 });
                }
            }
            const VkDeviceSize row_bytes = static_cast<VkDeviceSize>(frame->height) * sizeof(uint32_t);
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
            GpuVideo & gpu = _gpu_videos[i];
            slot_video.picture = -1;
            if (gpu.active)
            {
                if (motion_picture_size(frame->width, frame->height) != cv::Size(static_cast<int>(gpu.width), static_cast<int>(gpu.height)))
                {
                    throw std::runtime_error("the size of the frames changed, the motion video can't be drawn on the GPU");
                }
                // a picture on the GPU for the frame, until it is drawn
                if (gpu.free_pictures.empty())
                {
                    gpu.pictures.push_back(_create_buffer(gpu.picture_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, { VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0 }));
                    gpu.free_pictures.push_back(static_cast<int>(gpu.pictures.size()) - 1);
                }
                slot_video.picture = gpu.free_pictures.back();
                gpu.free_pictures.pop_back();
                const VkDescriptorBufferInfo base_buffers[3] = {
                    { video.luma[video.current].buffer, 0, VK_WHOLE_SIZE },
                    { video.chroma[video.current].buffer, 0, VK_WHOLE_SIZE },
                    { gpu.pictures[static_cast<size_t>(slot_video.picture)].buffer, 0, VK_WHOLE_SIZE },
                };
                std::array<VkWriteDescriptorSet, 3> base_writes {};
                for (uint32_t b = 0; b < 3; ++b)
                {
                    base_writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    base_writes[b].dstSet = slot_video.base_set;
                    base_writes[b].dstBinding = b;
                    base_writes[b].descriptorCount = 1;
                    base_writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    base_writes[b].pBufferInfo = &base_buffers[b];
                }
                _fn.vkUpdateDescriptorSets(_device, static_cast<uint32_t>(base_writes.size()), base_writes.data(), 0, nullptr);
            }
            else if (_motion_detail)
            {
                // the planes once more, in host memory: nothing else uses the buffers of a slot that isn't in flight
                const std::initializer_list<VkMemoryPropertyFlags> readable = {
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                };
                if (slot_video.picture_luma.size < plan.luma_bytes)
                {
                    _destroy_buffer(slot_video.picture_luma);
                    slot_video.picture_luma = _create_buffer(plan.luma_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, readable);
                }
                if (slot_video.picture_chroma.size < plan.chroma_bytes)
                {
                    _destroy_buffer(slot_video.picture_chroma);
                    slot_video.picture_chroma = _create_buffer(plan.chroma_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, readable);
                }
            }
            video.width = frame->width;
            video.height = frame->height;
            video.format = plan.format;
            slot_video.plan = plan;
            slot_video.luma = &video.luma[video.current];
            slot_video.chroma = &video.chroma[video.current];

            const std::array<VkDescriptorBufferInfo, kBindings> buffers = {{
                { video.luma[video.current].buffer, 0, VK_WHOLE_SIZE },
                { video.luma[1 - video.current].buffer, 0, VK_WHOLE_SIZE },
                { video.chroma[video.current].buffer, 0, VK_WHOLE_SIZE },
                { video.chroma[1 - video.current].buffer, 0, VK_WHOLE_SIZE },
                { slot_video.row_counts.buffer, 0, VK_WHOLE_SIZE },
            }};
            std::array<VkWriteDescriptorSet, kBindings> writes {};
            for (uint32_t b = 0; b < kBindings; ++b)
            {
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = slot_video.descriptor_set;
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1;
                writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[b].pBufferInfo = &buffers[b];
            }
            _fn.vkUpdateDescriptorSets(_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

            parameters[i] = { static_cast<uint32_t>(frame->width), plan.luma_row_words,
                              static_cast<uint32_t>(std::max(0, video_options[i].pixel_difference)),
                              static_cast<uint32_t>(plan.layout.chroma), static_cast<uint32_t>(plan.layout.shift_x),
                              static_cast<uint32_t>(plan.layout.shift_y), plan.chroma_row_bytes, plan.v_offset };
            slot_video.compared = compare;
            slot_video.height = frame->height;
            slot_video.width = frame->width;

            if (_motion)
            {
                const MotionLayout layout = motion_layout(frame->width, frame->height);
                for (Buffer & pyramid : video.pyramid)
                {
                    if (pyramid.size >= layout.bytes) continue;
                    _destroy_buffer(pyramid);
                    pyramid = _create_buffer(layout.bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, { VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0 });
                }
                // sparse motion: the gradient products of the previous frame, read in the same submission
                const VkDeviceSize product_bytes = motion_product_bytes(layout);
                if (video.products.size < product_bytes)
                {
                    _destroy_buffer(video.products);
                    video.products = _create_buffer(product_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, { VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0 });
                }
                // ... and what the last frame with motion was like, starting with "not sparse"
                if (!video.sparse_latch.buffer)
                {
                    video.sparse_latch = _create_buffer(sizeof(SparseLatch), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, {
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                    });
                    std::memset(video.sparse_latch.mapped, 0, sizeof(SparseLatch));
                    if (!video.sparse_latch.coherent)
                    {
                        VkMappedMemoryRange range { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
                        range.memory = video.sparse_latch.memory;
                        range.size = VK_WHOLE_SIZE;
                        check(_fn.vkFlushMappedMemoryRanges(_device, 1, &range), "vkFlushMappedMemoryRanges");
                    }
                }
                // read back: the tracked points, and for sparse motion what was found and the points added
                const std::pair<Buffer *, std::pair<VkDeviceSize, VkBufferUsageFlags>> read_back[3] = {
                    { &slot_video.motion_points, { kMotionPointBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT } },
                    { &slot_video.candidates, { kMotionCandidateBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT } },
                    { &slot_video.sparse_state, { sizeof(SparseState), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT } },
                };
                for (const auto & [buffer, wanted] : read_back)
                {
                    if (buffer->size >= wanted.first) continue;
                    _destroy_buffer(*buffer);
                    *buffer = _create_buffer(wanted.first, wanted.second, {
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                    });
                }
                slot_video.motion = layout;
                // the pyramid of the previous frame has the same layout if the frames can be compared
                slot_video.motion_estimated = compare;
                slot_video.sparse_possible = compare && sparse_motion_possible(layout);

                const VkDescriptorBufferInfo pyramid_buffers[2] = {
                    { video.luma[video.current].buffer, 0, VK_WHOLE_SIZE },
                    { video.pyramid[video.current].buffer, 0, VK_WHOLE_SIZE },
                };
                const VkDescriptorBufferInfo lk_buffers[4] = {
                    { video.pyramid[1 - video.current].buffer, 0, VK_WHOLE_SIZE },
                    { video.pyramid[video.current].buffer, 0, VK_WHOLE_SIZE },
                    { slot_video.motion_points.buffer, 0, VK_WHOLE_SIZE },
                    { slot_video.candidates.buffer, 0, VK_WHOLE_SIZE },
                };
                const VkDescriptorBufferInfo select_buffers[3] = {
                    { slot_video.motion_points.buffer, 0, VK_WHOLE_SIZE },
                    { slot_video.sparse_state.buffer, 0, VK_WHOLE_SIZE },
                    { video.sparse_latch.buffer, 0, VK_WHOLE_SIZE },
                };
                const VkDescriptorBufferInfo texture_buffers[4] = {
                    { video.pyramid[1 - video.current].buffer, 0, VK_WHOLE_SIZE },
                    { video.products.buffer, 0, VK_WHOLE_SIZE },
                    { slot_video.sparse_state.buffer, 0, VK_WHOLE_SIZE },
                    { slot_video.candidates.buffer, 0, VK_WHOLE_SIZE },
                };
                std::vector<VkWriteDescriptorSet> motion_writes;
                auto write = [&](VkDescriptorSet set, const VkDescriptorBufferInfo * buffers, uint32_t count) {
                    for (uint32_t b = 0; b < count; ++b)
                    {
                        VkWriteDescriptorSet entry { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                        entry.dstSet = set;
                        entry.dstBinding = b;
                        entry.descriptorCount = 1;
                        entry.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        entry.pBufferInfo = &buffers[b];
                        motion_writes.push_back(entry);
                    }
                };
                write(slot_video.pyramid_set, pyramid_buffers, 2);
                write(slot_video.lk_set, lk_buffers, 4);
                write(slot_video.select_set, select_buffers, 3);
                write(slot_video.texture_set, texture_buffers, 4);
                _fn.vkUpdateDescriptorSets(_device, static_cast<uint32_t>(motion_writes.size()), motion_writes.data(), 0, nullptr);
            }
        }

        // lock the decoded images while their state is read and updated
        for (SlotVideo & slot_video : slot.videos) _lock(slot_video.frame, true);

        // one barrier / semaphore wait + signal per VkImage of every frame
        std::vector<VkImageMemoryBarrier2> image_barriers;
        std::vector<VkSemaphoreSubmitInfo> waits;
        std::vector<VkSemaphoreSubmitInfo> signals;
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            const AVVkFrame * vkf = reinterpret_cast<const AVVkFrame *>(slot.videos[i].frame->data[0]);
            for (int j = 0; j < image_count(vkf); ++j)
            {
                VkImageMemoryBarrier2 barrier { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                barrier.srcAccessMask = vkf->access[j];
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                barrier.oldLayout = vkf->layout[j];
                barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                barrier.srcQueueFamilyIndex = vkf->queue_family[j];
                barrier.dstQueueFamilyIndex = vkf->queue_family[j] == VK_QUEUE_FAMILY_IGNORED ? VK_QUEUE_FAMILY_IGNORED : _queue_family;
                barrier.image = vkf->img[j];
                barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                image_barriers.push_back(barrier);

                VkSemaphoreSubmitInfo wait { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
                wait.semaphore = vkf->sem[j];
                wait.value = vkf->sem_value[j];
                wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                waits.push_back(wait);
                wait.value = vkf->sem_value[j] + 1;
                signals.push_back(wait);
            }
        }

        check(_fn.vkResetCommandBuffer(slot.command_buffer, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo begin { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(_fn.vkBeginCommandBuffer(slot.command_buffer, &begin), "vkBeginCommandBuffer");

        // earlier submissions read / wrote the plane buffers and pyramids, the decoded images become copy sources
        VkMemoryBarrier2 before { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        before.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
        before.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        before.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        before.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo dependency { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &before;
        dependency.imageMemoryBarrierCount = static_cast<uint32_t>(image_barriers.size());
        dependency.pImageMemoryBarriers = image_barriers.data();
        _fn.vkCmdPipelineBarrier2(slot.command_buffer, &dependency);

        for (size_t i = 0; i < _videos.size(); ++i)
        {
            const SlotVideo & slot_video = slot.videos[i];
            _record_copies(slot.command_buffer, slot_video, slot_video.luma->buffer, slot_video.chroma->buffer);
            if (_motion_detail && !_gpu_videos[i].active)
            {
                _record_copies(slot.command_buffer, slot_video, slot_video.picture_luma.buffer, slot_video.picture_chroma.buffer);
            }
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

        // motion video on the GPU: the pictures of the frames
        bool base_bound = false;
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            const SlotVideo & slot_video = slot.videos[i];
            if (slot_video.picture < 0) continue;
            const GpuVideo & gpu = _gpu_videos[i];
            const Plan & plan = slot_video.plan;
            OverlayBasePush push {};
            push.width = gpu.width;
            push.height = gpu.height;
            push.row_words = gpu.row_words;
            push.scale = static_cast<uint32_t>(motion_scale(slot_video.width, slot_video.height));
            push.luma_row_words = plan.luma_row_words;
            push.chroma_mode = static_cast<uint32_t>(plan.layout.chroma);
            push.chroma_row_bytes = plan.chroma_row_bytes;
            push.v_offset = plan.v_offset;
            // a chroma sample of the picture covers 2 x 2 of its pixels
            push.chroma_step_x = std::max(1u, (2 * push.scale) >> plan.layout.shift_x);
            push.chroma_step_y = std::max(1u, (2 * push.scale) >> plan.layout.shift_y);
            if (!base_bound) _fn.vkCmdBindPipeline(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _overlay_base.pipeline);
            base_bound = true;
            _fn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _overlay_base.layout, 0, 1, &slot_video.base_set, 0, nullptr);
            _fn.vkCmdPushConstants(slot.command_buffer, _overlay_base.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            _fn.vkCmdDispatch(slot.command_buffer, (gpu.row_words + 7) / 8, (gpu.height / 2 + 7) / 8, 1);
        }

        if (_motion) _record_motion(slot);

        VkMemoryBarrier2 host { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        host.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | (_motion_detail ? VK_PIPELINE_STAGE_2_COPY_BIT : 0);
        host.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | (_motion_detail ? VK_ACCESS_2_TRANSFER_WRITE_BIT : 0);
        host.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        host.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        VkDependencyInfo host_dependency { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        host_dependency.memoryBarrierCount = 1;
        host_dependency.pMemoryBarriers = &host;
        _fn.vkCmdPipelineBarrier2(slot.command_buffer, &host_dependency);
        check(_fn.vkEndCommandBuffer(slot.command_buffer), "vkEndCommandBuffer");

        VkCommandBufferSubmitInfo command_info { VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        command_info.commandBuffer = slot.command_buffer;
        VkSubmitInfo2 submit { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submit.waitSemaphoreInfoCount = static_cast<uint32_t>(waits.size());
        submit.pWaitSemaphoreInfos = waits.data();
        submit.commandBufferInfoCount = 1;
        submit.pCommandBufferInfos = &command_info;
        submit.signalSemaphoreInfoCount = static_cast<uint32_t>(signals.size());
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
            size_t barrier_index = 0;
            for (size_t i = 0; i < _videos.size(); ++i)
            {
                AVVkFrame * vkf = reinterpret_cast<AVVkFrame *>(slot.videos[i].frame->data[0]);
                for (int j = 0; j < image_count(vkf); ++j)
                {
                    vkf->sem_value[j] += 1;
                    vkf->layout[j] = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                    vkf->access[j] = VK_ACCESS_2_TRANSFER_READ_BIT;
                    vkf->queue_family[j] = image_barriers[barrier_index++].dstQueueFamilyIndex;
                }
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
            result.coding = ffmpeg::frame_coding(slot_video.frame);
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
            result.detail.reset();
            if (_motion_detail)
            {
                result.detail = std::make_shared<MotionDetail>();
                // the picture stays on the GPU if the motion video is drawn there
                if (_gpu_videos[i].active) result.detail->gpu_picture = slot_video.picture;
                else _read_picture(slot_video, result.detail->picture);
            }
            if (_motion) _read_motion(slot_video, result.motion, _motion_detail ? &result.detail->points : nullptr);
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
    std::string codec_name(size_t index) const override { return _videos.at(index).decoder->codec_name(); }
    bool lossy_codec(size_t index) const override { return _videos.at(index).decoder->lossy(); }

private:
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        void * mapped = nullptr;
        bool coherent = true;
    };

    //! how the planes of a frame are laid out in the buffers
    struct Plan {
        AVPixelFormat format = AV_PIX_FMT_NONE;
        ffmpeg::YuvLayout layout;
        uint32_t luma_row_words = 0;
        VkDeviceSize luma_bytes = 0;
        uint32_t chroma_width = 0;
        uint32_t chroma_height = 0;
        uint32_t chroma_row_bytes = 0;
        uint32_t v_offset = 0;
        VkDeviceSize chroma_bytes = 0;
    };

    struct VideoState {
        std::unique_ptr<ffmpeg::VideoDecoder> decoder;
        //! decoded while probing, the first submission
        AVFrame * first_frame = nullptr;
        //! ping pong buffers of the planes
        Buffer luma[2];
        Buffer chroma[2];
        //! ... and of the luma pyramids for the motion
        Buffer pyramid[2];
        //! sparse motion: gradient products of the previous frame at level 1 (motion_texture.comp)
        Buffer products;
        //! ... and what motion_select.comp keeps of the last frame in which a point moved (SparseLatch)
        Buffer sparse_latch;
        AVPixelFormat format = AV_PIX_FMT_NONE;
        int current = 0;
        int width = 0;
        int height = 0;
        bool has_previous = false;
    };

    struct SlotVideo {
        AVFrame * frame = nullptr;
        Buffer row_counts;
        //! buffers the planes of the frame are copied to
        Plan plan;
        const Buffer * luma = nullptr;
        const Buffer * chroma = nullptr;
        //! motion video: the planes in host memory, with the same layout
        Buffer picture_luma;
        Buffer picture_chroma;
        //! motion video on the GPU: the picture of the frame (index into GpuVideo::pictures, -1: none)
        int picture = -1;
        VkDescriptorSet base_set = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        // motion: tracked points (vec4 per point) are read back
        Buffer motion_points;
        //! sparse motion: the points added (as motion_texture.comp chose them) and what motion_select.comp found
        Buffer candidates;
        Buffer sparse_state;
        VkDescriptorSet pyramid_set = VK_NULL_HANDLE;
        VkDescriptorSet lk_set = VK_NULL_HANDLE;
        VkDescriptorSet select_set = VK_NULL_HANDLE;
        VkDescriptorSet texture_set = VK_NULL_HANDLE;
        MotionLayout motion;
        bool motion_estimated = false;
        //! the dispatches for sparse motion were recorded
        bool sparse_possible = false;
        bool compared = false;
        int width = 0;
        int height = 0;
    };

    //! motion video of a video, drawn and encoded on the GPU
    struct GpuVideo {
        bool active = false;
        uint32_t width = 0;
        uint32_t height = 0;
        //! uints per row of a picture, and its size: NV12
        uint32_t row_words = 0;
        VkDeviceSize picture_bytes = 0;
        std::unique_ptr<ffmpeg::EncodedVideo> encoder;
        //! the frames the encoder is fed with
        AVBufferRef * frames = nullptr;
        //! pictures of the frames which wait for their values, and the ones not in use
        std::vector<Buffer> pictures;
        std::vector<int> free_pictures;
    };

    //! a motion video frame being drawn
    struct DrawSlot {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool busy = false;
        VkDescriptorSet set = VK_NULL_HANDLE;
        // the scene, uploaded
        Buffer segments;
        Buffer tiles;
        Buffer text;
        //! the drawn picture, copied into the frame of the encoder
        Buffer output;
        AVFrame * frame = nullptr;
    };

    struct Slot {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool busy = false;
        std::vector<SlotVideo> videos;
    };

    //! buffer layout of the planes of a decoded frame
    Plan _plan(const AVFrame * frame) const
    {
        const AVHWFramesContext * frames = reinterpret_cast<const AVHWFramesContext *>(frame->hw_frames_ctx->data);
        Plan plan;
        plan.format = frames->sw_format;
        plan.layout = ffmpeg::yuv_layout(frames->sw_format);
        plan.luma_row_words = static_cast<uint32_t>((frame->width + 3) / 4);
        plan.luma_bytes = static_cast<VkDeviceSize>(plan.luma_row_words) * 4 * static_cast<VkDeviceSize>(frame->height);
        plan.chroma_width = static_cast<uint32_t>(AV_CEIL_RSHIFT(frame->width, plan.layout.shift_x));
        plan.chroma_height = static_cast<uint32_t>(AV_CEIL_RSHIFT(frame->height, plan.layout.shift_y));
        uint32_t bytes = 4; // placeholder buffer without chroma
        if (plan.layout.chroma == ffmpeg::YuvLayout::Interleaved)
        {
            plan.chroma_row_bytes = 2 * plan.chroma_width;
            bytes = plan.chroma_row_bytes * plan.chroma_height;
        }
        else if (plan.layout.chroma == ffmpeg::YuvLayout::Planar)
        {
            plan.chroma_row_bytes = plan.chroma_width;
            plan.v_offset = align4(plan.chroma_width * plan.chroma_height);
            bytes = plan.v_offset + plan.chroma_width * plan.chroma_height;
        }
        plan.chroma_bytes = align4(bytes);
        return plan;
    }

    //! copies the planes of the decoded image of the slot into a luma and a chroma buffer
    void _record_copies(VkCommandBuffer command_buffer, const SlotVideo & slot_video, VkBuffer luma, VkBuffer chroma)
    {
        const AVFrame * frame = slot_video.frame;
        const AVVkFrame * vkf = reinterpret_cast<const AVVkFrame *>(frame->data[0]);
        const Plan & plan = slot_video.plan;
        const int planes = plan.layout.chroma == ffmpeg::YuvLayout::None ? 1 : (plan.layout.chroma == ffmpeg::YuvLayout::Interleaved ? 2 : 3);
        const int images = image_count(vkf);
        // one multiplane image or one image per plane
        const bool multiplane = images == 1 && planes > 1;
        if (!multiplane && images != planes) throw std::runtime_error("unsupported Vulkan frame layout");
        const VkImageAspectFlagBits plane_aspects[3] = { VK_IMAGE_ASPECT_PLANE_0_BIT, VK_IMAGE_ASPECT_PLANE_1_BIT, VK_IMAGE_ASPECT_PLANE_2_BIT };

        auto copy = [&](int plane, VkBuffer buffer, VkDeviceSize offset, uint32_t row_texels, uint32_t width, uint32_t height,
                        int shift_x, int shift_y) {
            VkBufferImageCopy region {};
            region.bufferOffset = offset;
            region.bufferRowLength = row_texels;
            region.bufferImageHeight = height;
            region.imageSubresource = { static_cast<VkImageAspectFlags>(multiplane ? plane_aspects[plane] : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1 };
            // hardware frames keep the top / left cropping in the frame properties
            region.imageOffset = { static_cast<int32_t>(frame->crop_left >> shift_x), static_cast<int32_t>(frame->crop_top >> shift_y), 0 };
            region.imageExtent = { width, height, 1 };
            _fn.vkCmdCopyImageToBuffer(command_buffer, vkf->img[multiplane ? 0 : plane], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
        };
        copy(0, luma, 0, plan.luma_row_words * 4, static_cast<uint32_t>(frame->width), static_cast<uint32_t>(frame->height), 0, 0);
        const int sx = plan.layout.shift_x;
        const int sy = plan.layout.shift_y;
        if (plan.layout.chroma == ffmpeg::YuvLayout::Interleaved)
        {
            // UV pairs, row length in (2 byte) texels
            copy(1, chroma, 0, plan.chroma_width, plan.chroma_width, plan.chroma_height, sx, sy);
        }
        else if (plan.layout.chroma == ffmpeg::YuvLayout::Planar)
        {
            copy(1, chroma, 0, plan.chroma_width, plan.chroma_width, plan.chroma_height, sx, sy);
            copy(2, chroma, plan.v_offset, plan.chroma_width, plan.chroma_width, plan.chroma_height, sx, sy);
        }
    }

    //! builds the luma pyramids of the frames of the slot and tracks the points from the previous frames
    void _record_motion(Slot & slot)
    {
        VkMemoryBarrier2 written { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        written.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        written.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        written.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        written.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo written_dependency { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        written_dependency.memoryBarrierCount = 1;
        written_dependency.pMemoryBarriers = &written;

        _fn.vkCmdBindPipeline(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pyramid.pipeline);
        for (uint32_t level = 0; level < kMaxMotionLevels; ++level)
        {
            bool dispatched = false;
            for (const SlotVideo & slot_video : slot.videos)
            {
                const MotionLayout & layout = slot_video.motion;
                if (level >= layout.levels) continue;
                PyramidPush push {};
                push.dst_width = layout.width[level];
                push.dst_height = layout.height[level];
                push.dst_offset = layout.offset[level];
                if (level == 0)
                {
                    push.mode = 0;
                    push.factor = layout.scale;
                    push.src_width = static_cast<uint32_t>(slot_video.width);
                    push.src_height = static_cast<uint32_t>(slot_video.height);
                    push.src_base = slot_video.plan.luma_row_words;
                }
                else
                {
                    push.mode = 1;
                    push.factor = 2;
                    push.src_width = layout.width[level - 1];
                    push.src_height = layout.height[level - 1];
                    push.src_base = layout.offset[level - 1];
                }
                _fn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pyramid.layout, 0, 1, &slot_video.pyramid_set, 0, nullptr);
                _fn.vkCmdPushConstants(slot.command_buffer, _pyramid.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                _fn.vkCmdDispatch(slot.command_buffer, (push.dst_width + 15) / 16, (push.dst_height + 15) / 16, 1);
                dispatched = true;
            }
            // the next level / the tracking reads this level
            if (dispatched) _fn.vkCmdPipelineBarrier2(slot.command_buffer, &written_dependency);
        }

        auto lk_push = [](const MotionLayout & layout) {
            LkPush push {};
            for (uint32_t level = 0; level < layout.levels; ++level)
            {
                push.level_width[level] = layout.width[level];
                push.level_height[level] = layout.height[level];
                push.level_offset[level] = layout.offset[level];
            }
            push.levels = layout.levels;
            push.grid_x = static_cast<uint32_t>(layout.grid.points_x());
            push.grid_y = static_cast<uint32_t>(layout.grid.points_y());
            push.iterations = kMotionIterations;
            push.max_residual = kMaxResidual;
            return push;
        };
        _fn.vkCmdBindPipeline(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _lk.pipeline);
        bool sparse_possible = false;
        for (const SlotVideo & slot_video : slot.videos)
        {
            if (!slot_video.motion_estimated) continue;
            const LkPush push = lk_push(slot_video.motion);
            _fn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _lk.layout, 0, 1, &slot_video.lk_set, 0, nullptr);
            _fn.vkCmdPushConstants(slot.command_buffer, _lk.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            _fn.vkCmdDispatch(slot.command_buffer, kMotionPoints, 1, 1);
            sparse_possible |= slot_video.sparse_possible;
        }
        if (!sparse_possible) return;

        // Sparse motion (motion.h, kMotionSparseCells): what the regular grid found, then - dispatched indirectly, with
        // no workgroups unless the frame has sparse motion - the texture of the previous frame, the points with the
        // most of it and their tracking
        VkMemoryBarrier2 decided = written;
        decided.dstStageMask |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        decided.dstAccessMask |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        VkDependencyInfo decided_dependency { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        decided_dependency.memoryBarrierCount = 1;
        decided_dependency.pMemoryBarriers = &decided;
        // stage 0: motion_select.comp, 1 and 2: motion_texture.comp, 3: motion_lk.comp with the list
        for (int stage = 0; stage < 4; ++stage)
        {
            const Compute & compute = stage == 0 ? _select : stage == 3 ? _lk : _texture;
            _fn.vkCmdPipelineBarrier2(slot.command_buffer, stage == 1 ? &decided_dependency : &written_dependency);
            _fn.vkCmdBindPipeline(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline);
            for (const SlotVideo & slot_video : slot.videos)
            {
                if (!slot_video.sparse_possible) continue;
                const SparseMotionPushes pushes = sparse_motion_pushes(slot_video.motion, lk_push(slot_video.motion));
                const VkDescriptorSet set = stage == 0 ? slot_video.select_set : stage == 3 ? slot_video.lk_set : slot_video.texture_set;
                _fn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.layout, 0, 1, &set, 0, nullptr);
                if (stage == 0)
                {
                    _fn.vkCmdPushConstants(slot.command_buffer, compute.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushes.select), &pushes.select);
                    _fn.vkCmdDispatch(slot.command_buffer, 1, 1, 1);
                    continue;
                }
                if (stage == 3) _fn.vkCmdPushConstants(slot.command_buffer, compute.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushes.points), &pushes.points);
                else _fn.vkCmdPushConstants(slot.command_buffer, compute.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TexturePush), stage == 1 ? &pushes.products : &pushes.parts);
                _fn.vkCmdDispatchIndirect(slot.command_buffer, slot_video.sparse_state.buffer, static_cast<VkDeviceSize>(stage - 1) * 3 * sizeof(uint32_t));
            }
        }
    }

    //! the frame of a slot as the picture of the motion video, from the planes in host memory
    void _read_picture(const SlotVideo & slot_video, cv::Mat & picture)
    {
        for (const Buffer * buffer : { &slot_video.picture_luma, &slot_video.picture_chroma })
        {
            if (buffer->coherent) continue;
            VkMappedMemoryRange range { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
            range.memory = buffer->memory;
            range.size = VK_WHOLE_SIZE;
            check(_fn.vkInvalidateMappedMemoryRanges(_device, 1, &range), "vkInvalidateMappedMemoryRanges");
        }
        const Plan & plan = slot_video.plan;
        ffmpeg::YuvPlanes planes;
        planes.width = slot_video.width;
        planes.height = slot_video.height;
        planes.layout = plan.layout;
        planes.v_first = plan.format == AV_PIX_FMT_NV21 || plan.format == AV_PIX_FMT_NV42;
        planes.data[0] = static_cast<const uint8_t *>(slot_video.picture_luma.mapped);
        planes.stride[0] = static_cast<size_t>(plan.luma_row_words) * 4;
        planes.data[1] = static_cast<const uint8_t *>(slot_video.picture_chroma.mapped);
        planes.stride[1] = plan.chroma_row_bytes;
        planes.data[2] = planes.data[1] + plan.v_offset;
        planes.stride[2] = plan.chroma_row_bytes;
        ffmpeg::motion_picture(planes, picture, _scratch);
    }

    //! the tracked points of a frame as the median motion per grid cell, and one by one if `detail` is given
    void _read_motion(const SlotVideo & slot_video, MotionField & field, std::vector<MotionPoint> * detail)
    {
        field = MotionField();
        field.pixel_scale = static_cast<float>(slot_video.motion.scale);
        if (detail) detail->clear();
        if (!slot_video.motion_estimated) return;
        field.estimated = true;
        if (!slot_video.motion_points.coherent)
        {
            VkMappedMemoryRange range { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
            range.memory = slot_video.motion_points.memory;
            range.size = VK_WHOLE_SIZE;
            check(_fn.vkInvalidateMappedMemoryRanges(_device, 1, &range), "vkInvalidateMappedMemoryRanges");
        }
        const MotionGrid & grid = slot_video.motion.grid;
        const float points_x = static_cast<float>(grid.points_x());
        const float points_y = static_cast<float>(grid.points_y());
        _binner.clear(grid);
        const float * points = static_cast<const float *>(slot_video.motion_points.mapped);
        for (int y = 0; y < grid.points_y(); ++y)
        {
            for (int x = 0; x < grid.points_x(); ++x)
            {
                const float * point = points + 4 * (y * grid.points_x() + x);
                // (dx, dy, texture, distance by which the way back misses the point)
                const bool texture = point[2] >= kMotionMinTexture && std::isfinite(point[0]) && std::isfinite(point[1]);
                const bool returns = texture && point[3] <= kMotionMaxBackError;
                if (detail)
                {
                    // the same grid as motion_lk.comp, in pixels of the first level of the pyramid
                    MotionPoint tracked;
                    tracked.x = (static_cast<float>(x) + 0.5f) * static_cast<float>(slot_video.motion.width[0]) / points_x - 0.5f;
                    tracked.y = (static_cast<float>(y) + 0.5f) * static_cast<float>(slot_video.motion.height[0]) / points_y - 0.5f;
                    tracked.state = returns ? MotionPoint::Used : texture ? MotionPoint::Rejected : MotionPoint::NoTexture;
                    if (texture)
                    {
                        tracked.dx = point[0];
                        tracked.dy = point[1];
                    }
                    detail->push_back(tracked);
                }
                if (!returns) continue;
                _binner.add((static_cast<float>(x) + 0.5f) / points_x, (static_cast<float>(y) + 0.5f) / points_y, point[0], point[1]);
            }
        }
        // sparse motion: the points added where the previous frame has texture
        if (slot_video.sparse_possible)
        {
            for (const Buffer * buffer : { &slot_video.sparse_state, &slot_video.candidates })
            {
                if (buffer->coherent) continue;
                VkMappedMemoryRange range { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
                range.memory = buffer->memory;
                range.size = VK_WHOLE_SIZE;
                check(_fn.vkInvalidateMappedMemoryRanges(_device, 1, &range), "vkInvalidateMappedMemoryRanges");
            }
            const float width = static_cast<float>(slot_video.motion.width[0]), height = static_cast<float>(slot_video.motion.height[0]);
            read_sparse_motion(*static_cast<const SparseState *>(slot_video.sparse_state.mapped), static_cast<const float *>(slot_video.candidates.mapped), points,
                               [&](float x, float y, float dx, float dy, bool texture, bool returns) {
                if (detail)
                {
                    MotionPoint tracked;
                    tracked.x = x;
                    tracked.y = y;
                    tracked.state = returns ? MotionPoint::Used : texture ? MotionPoint::Rejected : MotionPoint::NoTexture;
                    if (texture)
                    {
                        tracked.dx = dx;
                        tracked.dy = dy;
                    }
                    detail->push_back(tracked);
                }
                if (returns) _binner.add((x + 0.5f) / width, (y + 0.5f) / height, dx, dy);
            });
        }
        _binner.finish(field, kMotionMinCellPoints, field.pixel_scale, slot_video.sparse_possible && static_cast<const SparseState *>(slot_video.sparse_state.mapped)->sparse != 0);
    }

    //! a buffer of at least `bytes`, its contents are lost if it has to grow
    void _ensure_buffer(Buffer & buffer, VkDeviceSize bytes, VkBufferUsageFlags usage, std::initializer_list<VkMemoryPropertyFlags> preferred)
    {
        if (buffer.size >= bytes) return;
        _destroy_buffer(buffer);
        buffer = _create_buffer(std::max(bytes, buffer.size * 2), usage, preferred);
    }

    void _upload(Buffer & buffer, const void * data, size_t bytes)
    {
        _ensure_buffer(buffer, std::max<VkDeviceSize>(bytes, 32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, {
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        });
        if (bytes > 0) std::memcpy(buffer.mapped, data, bytes);
        if (!buffer.coherent)
        {
            VkMappedMemoryRange range { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
            range.memory = buffer.memory;
            range.size = VK_WHOLE_SIZE;
            check(_fn.vkFlushMappedMemoryRanges(_device, 1, &range), "vkFlushMappedMemoryRanges");
        }
    }

    //! Motion video on the GPU: draws the scene on the picture a frame left on the GPU, into a frame of the
    //! encoder, and hands it to the encoder. Nothing but the scene is uploaded, nothing is read back.
    void _draw(size_t index, int picture, const MotionVideoScene & scene)
    {
        GpuVideo & gpu = _gpu_videos.at(index);
        if (picture < 0 || picture >= static_cast<int>(gpu.pictures.size())) throw std::logic_error("motion video: the frame has no picture on the GPU");
        DrawSlot & slot = _draw_slots[_draws % kDrawSlots];
        if (slot.busy)
        {
            check(_fn.vkWaitForFences(_device, 1, &slot.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
            slot.busy = false;
        }

        // the scene as the buffers and the constants of overlay_draw.comp
        build_overlay_upload(scene, gpu.width, gpu.height, gpu.row_words, false, _overlay_upload);
        const OverlayDrawPush & push = _overlay_upload.push;
        _upload(slot.segments, _overlay_upload.segments.data(), _overlay_upload.segments.size() * sizeof(float));
        _upload(slot.tiles, _overlay_upload.tiles.data(), _overlay_upload.tiles.size() * sizeof(uint32_t));
        _upload(slot.text, _overlay_upload.text.data(), _overlay_upload.text.size());
        _ensure_buffer(slot.output, gpu.picture_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       { VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0 });

        const VkDescriptorBufferInfo buffers[5] = {
            { gpu.pictures[static_cast<size_t>(picture)].buffer, 0, VK_WHOLE_SIZE },
            { slot.segments.buffer, 0, VK_WHOLE_SIZE },
            { slot.tiles.buffer, 0, VK_WHOLE_SIZE },
            { slot.text.buffer, 0, VK_WHOLE_SIZE },
            { slot.output.buffer, 0, VK_WHOLE_SIZE },
        };
        std::array<VkWriteDescriptorSet, 5> writes {};
        for (uint32_t b = 0; b < 5; ++b)
        {
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = slot.set;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &buffers[b];
        }
        _fn.vkUpdateDescriptorSets(_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        // a frame of the encoder: NV12, one image with two planes or an image per plane
        av_frame_unref(slot.frame);
        const int allocated = av_hwframe_get_buffer(gpu.frames, slot.frame, 0);
        if (allocated < 0) throw std::runtime_error("no frame for the video encoder (" + ffmpeg::error_string(allocated) + ")");
        AVVkFrame * vkf = reinterpret_cast<AVVkFrame *>(slot.frame->data[0]);
        const int images = image_count(vkf);
        const bool multiplane = images == 1;
        if (!multiplane && images != 2) throw std::runtime_error("unsupported Vulkan frame layout of the video encoder");

        _lock(slot.frame, true);
        std::vector<VkImageMemoryBarrier2> image_barriers;
        std::vector<VkSemaphoreSubmitInfo> waits;
        std::vector<VkSemaphoreSubmitInfo> signals;
        for (int j = 0; j < images; ++j)
        {
            VkImageMemoryBarrier2 barrier { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            barrier.srcAccessMask = vkf->access[j];
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            // the whole image is written
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = vkf->queue_family[j];
            barrier.dstQueueFamilyIndex = vkf->queue_family[j] == VK_QUEUE_FAMILY_IGNORED ? VK_QUEUE_FAMILY_IGNORED : _queue_family;
            barrier.image = vkf->img[j];
            barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            image_barriers.push_back(barrier);

            VkSemaphoreSubmitInfo wait { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
            wait.semaphore = vkf->sem[j];
            wait.value = vkf->sem_value[j];
            wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            waits.push_back(wait);
            wait.value = vkf->sem_value[j] + 1;
            signals.push_back(wait);
        }

        check(_fn.vkResetCommandBuffer(slot.command_buffer, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo begin { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(_fn.vkBeginCommandBuffer(slot.command_buffer, &begin), "vkBeginCommandBuffer");

        // the picture was written by the submission of its frame, the result buffer read by an earlier copy
        VkMemoryBarrier2 before { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        before.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
        before.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
        before.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        before.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo dependency { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &before;
        _fn.vkCmdPipelineBarrier2(slot.command_buffer, &dependency);

        _fn.vkCmdBindPipeline(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _overlay_draw.pipeline);
        _fn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _overlay_draw.layout, 0, 1, &slot.set, 0, nullptr);
        _fn.vkCmdPushConstants(slot.command_buffer, _overlay_draw.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        _fn.vkCmdDispatch(slot.command_buffer, (gpu.row_words + 7) / 8, (gpu.height / 2 + 7) / 8, 1);

        VkMemoryBarrier2 drawn { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        drawn.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        drawn.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        drawn.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        drawn.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        VkDependencyInfo drawn_dependency { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        drawn_dependency.memoryBarrierCount = 1;
        drawn_dependency.pMemoryBarriers = &drawn;
        drawn_dependency.imageMemoryBarrierCount = static_cast<uint32_t>(image_barriers.size());
        drawn_dependency.pImageMemoryBarriers = image_barriers.data();
        _fn.vkCmdPipelineBarrier2(slot.command_buffer, &drawn_dependency);

        const VkDeviceSize row_bytes = static_cast<VkDeviceSize>(gpu.row_words) * 4;
        for (int plane = 0; plane < 2; ++plane)
        {
            VkBufferImageCopy region {};
            region.bufferOffset = plane == 0 ? 0 : row_bytes * gpu.height;
            // in texels: bytes of luma, UV pairs of chroma
            region.bufferRowLength = plane == 0 ? gpu.row_words * 4 : gpu.row_words * 2;
            region.bufferImageHeight = plane == 0 ? gpu.height : gpu.height / 2;
            region.imageSubresource = { static_cast<VkImageAspectFlags>(multiplane ? (plane == 0 ? VK_IMAGE_ASPECT_PLANE_0_BIT : VK_IMAGE_ASPECT_PLANE_1_BIT)
                                                                                    : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1 };
            region.imageExtent = { plane == 0 ? gpu.width : gpu.width / 2, plane == 0 ? gpu.height : gpu.height / 2, 1 };
            _fn.vkCmdCopyBufferToImage(slot.command_buffer, slot.output.buffer, vkf->img[multiplane ? 0 : plane],
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        }
        check(_fn.vkEndCommandBuffer(slot.command_buffer), "vkEndCommandBuffer");

        VkCommandBufferSubmitInfo command_info { VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        command_info.commandBuffer = slot.command_buffer;
        VkSubmitInfo2 submit { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submit.waitSemaphoreInfoCount = static_cast<uint32_t>(waits.size());
        submit.pWaitSemaphoreInfos = waits.data();
        submit.commandBufferInfoCount = 1;
        submit.pCommandBufferInfos = &command_info;
        submit.signalSemaphoreInfoCount = static_cast<uint32_t>(signals.size());
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
            for (int j = 0; j < images; ++j)
            {
                vkf->sem_value[j] += 1;
                vkf->layout[j] = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                vkf->access[j] = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                vkf->queue_family[j] = image_barriers[static_cast<size_t>(j)].dstQueueFamilyIndex;
            }
        }
        _lock(slot.frame, false);
        check(result, "vkQueueSubmit2");
        slot.busy = true;
        ++_draws;

        // the encoder waits for the frame itself
        gpu.encoder->encode(slot.frame);
        av_frame_unref(slot.frame);
        gpu.free_pictures.push_back(picture);
    }

    void _finish_video(size_t index)
    {
        GpuVideo & gpu = _gpu_videos.at(index);
        if (gpu.encoder) gpu.encoder->finish();
    }

    //! the motion video of one of the videos, drawn and encoded by this pipeline
    class GpuTarget final : public MotionVideoTarget {
    public:
        GpuTarget(VulkanVideoPipeline & pipeline, size_t index) : _pipeline(pipeline), _index(index) { }
        std::string description() const override { return _pipeline._gpu_videos[_index].encoder->encoder_name() + ", drawn on the GPU"; }
        void write(const MotionDetail & detail, const MotionVideoScene & scene) override { _pipeline._draw(_index, detail.gpu_picture, scene); }
        void finish() override { _pipeline._finish_video(_index); }

    private:
        VulkanVideoPipeline & _pipeline;
        size_t _index;
    };

    static void _check_frame(const AVFrame * frame, const ffmpeg::VideoDecoder & decoder)
    {
        if (frame->format != AV_PIX_FMT_VULKAN || !frame->hw_frames_ctx) throw std::runtime_error("no Vulkan decoding for " + decoder.codec_name());
        const AVHWFramesContext * frames = reinterpret_cast<const AVHWFramesContext *>(frame->hw_frames_ctx->data);
        if (!ffmpeg::is_luma_format(frames->sw_format))
        {
            const char * name = av_get_pix_fmt_name(frames->sw_format);
            throw std::runtime_error(std::string("pixel format ") + (name ? name : "unknown") + " is no 8-bit YUV format");
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
        _max_group_size = properties.limits.maxComputeWorkGroupSize[0];
        _max_group_invocations = properties.limits.maxComputeWorkGroupInvocations;
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
        std::array<VkDescriptorSetLayoutBinding, kBindings> bindings {};
        for (uint32_t b = 0; b < kBindings; ++b)
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
        module_info.codeSize = sizeof(kYuvDiffSpirv);
        module_info.pCode = kYuvDiffSpirv;
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

    struct Compute {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
    };

    //! a compute pipeline with `bindings` storage buffers in one set and push constants
    Compute _create_compute(const uint32_t * code, size_t code_bytes, uint32_t bindings, uint32_t push_bytes)
    {
        Compute compute;
        try
        {
            std::vector<VkDescriptorSetLayoutBinding> layout_bindings(bindings);
            for (uint32_t b = 0; b < bindings; ++b)
            {
                layout_bindings[b].binding = b;
                layout_bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                layout_bindings[b].descriptorCount = 1;
                layout_bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            VkDescriptorSetLayoutCreateInfo set_layout_info { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            set_layout_info.bindingCount = bindings;
            set_layout_info.pBindings = layout_bindings.data();
            check(_fn.vkCreateDescriptorSetLayout(_device, &set_layout_info, nullptr, &compute.set_layout), "vkCreateDescriptorSetLayout");

            VkPushConstantRange push_range { VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes };
            VkPipelineLayoutCreateInfo layout_info { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            layout_info.setLayoutCount = 1;
            layout_info.pSetLayouts = &compute.set_layout;
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &push_range;
            check(_fn.vkCreatePipelineLayout(_device, &layout_info, nullptr, &compute.layout), "vkCreatePipelineLayout");

            VkShaderModuleCreateInfo module_info { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
            module_info.codeSize = code_bytes;
            module_info.pCode = code;
            VkShaderModule module = VK_NULL_HANDLE;
            check(_fn.vkCreateShaderModule(_device, &module_info, nullptr, &module), "vkCreateShaderModule");
            VkComputePipelineCreateInfo pipeline_info { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipeline_info.stage.module = module;
            pipeline_info.stage.pName = "main";
            pipeline_info.layout = compute.layout;
            const VkResult result = _fn.vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &compute.pipeline);
            _fn.vkDestroyShaderModule(_device, module, nullptr);
            check(result, "vkCreateComputePipelines");
        }
        catch (...)
        {
            _destroy_compute(compute);
            throw;
        }
        return compute;
    }

    void _destroy_compute(Compute & compute)
    {
        if (compute.pipeline) _fn.vkDestroyPipeline(_device, compute.pipeline, nullptr);
        if (compute.layout) _fn.vkDestroyPipelineLayout(_device, compute.layout, nullptr);
        if (compute.set_layout) _fn.vkDestroyDescriptorSetLayout(_device, compute.set_layout, nullptr);
        compute = Compute();
    }

    void _create_slots()
    {
        // per video of a slot: the comparison, the pyramid and the tracking set
        const uint32_t set_count = static_cast<uint32_t>(kSlots * _videos.size());
        // ... and the picture of the motion video; a set per motion video frame being drawn
        // ... and for sparse motion what the grid found and the texture
        VkDescriptorPoolSize pool_size { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (kBindings + 2 + 4 + 3 + 3 + 4) * set_count + 5 * static_cast<uint32_t>(kDrawSlots) };
        VkDescriptorPoolCreateInfo pool_info { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        pool_info.maxSets = 6 * set_count + static_cast<uint32_t>(kDrawSlots);
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
                set_info.pSetLayouts = &_pyramid.set_layout;
                check(_fn.vkAllocateDescriptorSets(_device, &set_info, &video.pyramid_set), "vkAllocateDescriptorSets");
                set_info.pSetLayouts = &_lk.set_layout;
                check(_fn.vkAllocateDescriptorSets(_device, &set_info, &video.lk_set), "vkAllocateDescriptorSets");
                set_info.pSetLayouts = &_select.set_layout;
                check(_fn.vkAllocateDescriptorSets(_device, &set_info, &video.select_set), "vkAllocateDescriptorSets");
                set_info.pSetLayouts = &_texture.set_layout;
                check(_fn.vkAllocateDescriptorSets(_device, &set_info, &video.texture_set), "vkAllocateDescriptorSets");
                set_info.pSetLayouts = &_overlay_base.set_layout;
                check(_fn.vkAllocateDescriptorSets(_device, &set_info, &video.base_set), "vkAllocateDescriptorSets");
            }
        }
        for (DrawSlot & slot : _draw_slots)
        {
            VkCommandBufferAllocateInfo command_info { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            command_info.commandPool = _command_pool;
            command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            command_info.commandBufferCount = 1;
            check(_fn.vkAllocateCommandBuffers(_device, &command_info, &slot.command_buffer), "vkAllocateCommandBuffers");
            VkFenceCreateInfo fence_info { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
            check(_fn.vkCreateFence(_device, &fence_info, nullptr, &slot.fence), "vkCreateFence");
            VkDescriptorSetAllocateInfo set_info { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            set_info.descriptorPool = _descriptor_pool;
            set_info.descriptorSetCount = 1;
            set_info.pSetLayouts = &_overlay_draw.set_layout;
            check(_fn.vkAllocateDescriptorSets(_device, &set_info, &slot.set), "vkAllocateDescriptorSets");
            slot.frame = av_frame_alloc();
            if (!slot.frame) throw std::runtime_error("out of memory");
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
        // the encoders of the motion videos and their frames use the device as well
        for (DrawSlot & slot : _draw_slots) av_frame_free(&slot.frame);
        for (GpuVideo & gpu : _gpu_videos)
        {
            gpu.encoder.reset();
            av_buffer_unref(&gpu.frames);
        }
        // our Vulkan objects first, then the frames and decoders, the device last
        if (_device && _fn.vkDeviceWaitIdle)
        {
            _fn.vkDeviceWaitIdle(_device);
            for (DrawSlot & slot : _draw_slots)
            {
                for (Buffer * buffer : { &slot.segments, &slot.tiles, &slot.text, &slot.output }) _destroy_buffer(*buffer);
                if (slot.fence) _fn.vkDestroyFence(_device, slot.fence, nullptr);
                slot.fence = VK_NULL_HANDLE;
            }
            for (GpuVideo & gpu : _gpu_videos)
            {
                for (Buffer & buffer : gpu.pictures) _destroy_buffer(buffer);
                gpu.pictures.clear();
            }
            _destroy_compute(_overlay_base);
            _destroy_compute(_overlay_draw);
            for (Slot & slot : _slots)
            {
                for (SlotVideo & video : slot.videos)
                {
                    _destroy_buffer(video.row_counts);
                    _destroy_buffer(video.motion_points);
                    _destroy_buffer(video.candidates);
                    _destroy_buffer(video.sparse_state);
                    _destroy_buffer(video.picture_luma);
                    _destroy_buffer(video.picture_chroma);
                }
                if (slot.fence) _fn.vkDestroyFence(_device, slot.fence, nullptr);
                slot.fence = VK_NULL_HANDLE;
            }
            for (VideoState & video : _videos)
            {
                for (Buffer & buffer : video.luma) _destroy_buffer(buffer);
                for (Buffer & buffer : video.chroma) _destroy_buffer(buffer);
                for (Buffer & buffer : video.pyramid) _destroy_buffer(buffer);
                _destroy_buffer(video.products);
                _destroy_buffer(video.sparse_latch);
            }
            _destroy_compute(_pyramid);
            _destroy_compute(_lk);
            _destroy_compute(_select);
            _destroy_compute(_texture);
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
    uint32_t _max_group_size = 0;
    uint32_t _max_group_invocations = 0;
    VkPhysicalDeviceMemoryProperties _memory_properties {};
    struct {
#define TRDROP_VKV_DECLARE(name) PFN_##name name = nullptr;
        TRDROP_VKV_DEVICE_FUNCTIONS(TRDROP_VKV_DECLARE)
#undef TRDROP_VKV_DECLARE
    } _fn;

    // motion: pyramid and tracking pipelines
    bool _motion = false;
    //! motion video: the picture and the points of every frame are kept
    bool _motion_detail = false;
    cv::Mat _scratch;
    // ... or drawn on the GPU
    std::vector<GpuVideo> _gpu_videos;
    std::array<DrawSlot, kDrawSlots> _draw_slots;
    size_t _draws = 0;
    Compute _overlay_base;
    Compute _overlay_draw;
    OverlayUpload _overlay_upload;
    Compute _pyramid;
    Compute _lk;
    Compute _select;
    Compute _texture;
    MotionCellBinner _binner;

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
