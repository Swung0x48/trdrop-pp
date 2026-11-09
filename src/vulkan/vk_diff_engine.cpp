// Vulkan implementation of the DiffEngine: grayscale conversion, frame difference, threshold and
// the per row sum run in one compute shader (row_diff.comp), the rest of the analysis stays on the CPU.
//
// - the Vulkan loader is loaded at runtime, so a Vulkan build still runs (with the CPU backend)
//   on systems without a Vulkan driver
// - frames are decoded directly into host visible buffers (no extra copy) and up to kSlots
//   sets of frames are in flight, so decoding the next frame overlaps with the GPU work
// - the grayscale image of the previous frame stays on the GPU (ping pong buffers per video)

#include "diff_engine.h"

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <initializer_list>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace trdrop {

namespace {

const uint32_t kRowDiffSpirv[] = {
#include "row_diff.spv.inc"
};

//! sets of frames in flight
constexpr size_t kSlots = 2;
//! must match local_size_x of row_diff.comp (only used for documentation / sanity)
constexpr uint32_t kWorkgroupSize = 128;

struct PushConstants {
    uint32_t width;
    uint32_t height;
    uint32_t src_row_words;
    uint32_t threshold;
    uint32_t compare;
};

// ---------------------------------------------------------------------------------------------
// loader

#define TRDROP_VK_INSTANCE_FUNCTIONS(X) \
    X(vkDestroyInstance) \
    X(vkEnumeratePhysicalDevices) \
    X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkCreateDevice) \
    X(vkGetDeviceProcAddr)

#define TRDROP_VK_DEVICE_FUNCTIONS(X) \
    X(vkDestroyDevice) \
    X(vkGetDeviceQueue) \
    X(vkCreateBuffer) \
    X(vkDestroyBuffer) \
    X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory) \
    X(vkFreeMemory) \
    X(vkBindBufferMemory) \
    X(vkMapMemory) \
    X(vkUnmapMemory) \
    X(vkInvalidateMappedMemoryRanges) \
    X(vkCreateShaderModule) \
    X(vkDestroyShaderModule) \
    X(vkCreateDescriptorSetLayout) \
    X(vkDestroyDescriptorSetLayout) \
    X(vkCreatePipelineLayout) \
    X(vkDestroyPipelineLayout) \
    X(vkCreateComputePipelines) \
    X(vkDestroyPipeline) \
    X(vkCreateDescriptorPool) \
    X(vkDestroyDescriptorPool) \
    X(vkAllocateDescriptorSets) \
    X(vkUpdateDescriptorSets) \
    X(vkCreateCommandPool) \
    X(vkDestroyCommandPool) \
    X(vkAllocateCommandBuffers) \
    X(vkBeginCommandBuffer) \
    X(vkEndCommandBuffer) \
    X(vkResetCommandBuffer) \
    X(vkCmdBindPipeline) \
    X(vkCmdBindDescriptorSets) \
    X(vkCmdPushConstants) \
    X(vkCmdDispatch) \
    X(vkCmdPipelineBarrier) \
    X(vkQueueSubmit) \
    X(vkCreateFence) \
    X(vkDestroyFence) \
    X(vkWaitForFences) \
    X(vkResetFences) \
    X(vkDeviceWaitIdle)

#define TRDROP_VK_DECLARE(name) PFN_##name name = nullptr;

struct GlobalFunctions {
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    PFN_vkCreateInstance vkCreateInstance = nullptr;
    PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties = nullptr;
};

struct InstanceFunctions { TRDROP_VK_INSTANCE_FUNCTIONS(TRDROP_VK_DECLARE) };
struct DeviceFunctions { TRDROP_VK_DEVICE_FUNCTIONS(TRDROP_VK_DECLARE) };

//! loads the Vulkan loader library once, throws if it is not installed
const GlobalFunctions & global_functions()
{
    static GlobalFunctions functions;
    static std::string error;
    static std::once_flag once;
    std::call_once(once, [] {
        PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr;
#ifdef _WIN32
        HMODULE library = LoadLibraryA("vulkan-1.dll");
        if (library) get_instance_proc_addr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
            reinterpret_cast<void *>(GetProcAddress(library, "vkGetInstanceProcAddr")));
#else
        const char * names[] = {
#ifdef __APPLE__
            "libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib",
#else
            "libvulkan.so.1", "libvulkan.so",
#endif
        };
        for (const char * name : names)
        {
            void * library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
            if (!library) continue;
            get_instance_proc_addr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
            if (get_instance_proc_addr) break;
        }
#endif
        if (!get_instance_proc_addr)
        {
            error = "the Vulkan loader library could not be loaded";
            return;
        }
        functions.vkGetInstanceProcAddr = get_instance_proc_addr;
        functions.vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
            get_instance_proc_addr(VK_NULL_HANDLE, "vkCreateInstance"));
        functions.vkEnumerateInstanceLayerProperties = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
            get_instance_proc_addr(VK_NULL_HANDLE, "vkEnumerateInstanceLayerProperties"));
        if (!functions.vkCreateInstance) error = "the Vulkan loader library is invalid";
    });
    if (!error.empty()) throw std::runtime_error(error);
    return functions;
}

void check(VkResult result, const char * what)
{
    if (result != VK_SUCCESS)
    {
        throw std::runtime_error(std::string(what) + " failed (VkResult " + std::to_string(static_cast<int>(result)) + ")");
    }
}

const char * device_type_name(VkPhysicalDeviceType type)
{
    switch (type)
    {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete GPU";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU";
    default: return "other";
    }
}

bool env_flag(const char * name)
{
    const char * value = std::getenv(name);
    return value && *value && std::strcmp(value, "0") != 0;
}

//! VkInstance + the instance level functions
class Instance {
public:
    Instance()
    {
        const GlobalFunctions & global = global_functions();

        VkApplicationInfo application {};
        application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        application.pApplicationName = "trdrop-pp";
        application.applicationVersion = 1;
        application.pEngineName = "trdrop-pp";
        application.apiVersion = VK_API_VERSION_1_0;

        // TRDROP_VK_VALIDATION=1 enables the Khronos validation layer if it is installed (debugging)
        std::vector<const char *> layers;
        if (env_flag("TRDROP_VK_VALIDATION") && global.vkEnumerateInstanceLayerProperties)
        {
            uint32_t count = 0;
            global.vkEnumerateInstanceLayerProperties(&count, nullptr);
            std::vector<VkLayerProperties> available(count);
            global.vkEnumerateInstanceLayerProperties(&count, available.data());
            for (const VkLayerProperties & layer : available)
            {
                if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) layers.push_back("VK_LAYER_KHRONOS_validation");
            }
        }

        VkInstanceCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        info.pApplicationInfo = &application;
        info.enabledLayerCount = static_cast<uint32_t>(layers.size());
        info.ppEnabledLayerNames = layers.data();
        check(global.vkCreateInstance(&info, nullptr, &handle), "vkCreateInstance");

#define TRDROP_VK_LOAD_INSTANCE(name) \
        fn.name = reinterpret_cast<PFN_##name>(global.vkGetInstanceProcAddr(handle, #name)); \
        if (!fn.name) throw std::runtime_error("missing Vulkan function " #name);
        TRDROP_VK_INSTANCE_FUNCTIONS(TRDROP_VK_LOAD_INSTANCE)
#undef TRDROP_VK_LOAD_INSTANCE
    }

    ~Instance()
    {
        if (handle && fn.vkDestroyInstance) fn.vkDestroyInstance(handle, nullptr);
    }

    Instance(const Instance &) = delete;
    Instance & operator=(const Instance &) = delete;

    std::vector<VkPhysicalDevice> physical_devices() const
    {
        uint32_t count = 0;
        check(fn.vkEnumeratePhysicalDevices(handle, &count, nullptr), "vkEnumeratePhysicalDevices");
        std::vector<VkPhysicalDevice> devices(count);
        if (count > 0) check(fn.vkEnumeratePhysicalDevices(handle, &count, devices.data()), "vkEnumeratePhysicalDevices");
        devices.resize(count);
        return devices;
    }

    VkInstance handle = VK_NULL_HANDLE;
    InstanceFunctions fn;
};

//! makes sure the shader computes exactly the same grayscale values as cv::cvtColor of this OpenCV build
void verify_gray_formula()
{
    static bool verified = false;
    if (verified) return;
    cv::Mat bgr(256, 256, CV_8UC3);
    cv::Mat gray;
    for (int r = 0; r < 256; ++r)
    {
        for (int g = 0; g < 256; ++g)
        {
            uint8_t * pixel = bgr.ptr<uint8_t>(g);
            for (int b = 0; b < 256; ++b)
            {
                pixel[3 * b + 0] = static_cast<uint8_t>(b);
                pixel[3 * b + 1] = static_cast<uint8_t>(g);
                pixel[3 * b + 2] = static_cast<uint8_t>(r);
            }
        }
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        for (int g = 0; g < 256; ++g)
        {
            const uint8_t * row = gray.ptr<uint8_t>(g);
            for (int b = 0; b < 256; ++b)
            {
                const uint32_t expected = (static_cast<uint32_t>(b) * 3735u + static_cast<uint32_t>(g) * 19235u
                                         + static_cast<uint32_t>(r) * 9798u + 16384u) >> 15;
                if (row[b] != expected)
                {
                    throw std::runtime_error("the grayscale conversion of this OpenCV build differs from the Vulkan shader");
                }
            }
        }
    }
    verified = true;
}

size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

// ---------------------------------------------------------------------------------------------
// engine

class VulkanDiffEngine final : public DiffEngine {
public:
    VulkanDiffEngine(int device_index, size_t video_count)
        : _instance(std::make_unique<Instance>()), _videos(video_count), _targets(video_count)
    {
        try
        {
            verify_gray_formula();
            _select_device(device_index);
            _create_device();
            _create_pipeline();
            _create_slots();
        }
        catch (...)
        {
            _release();
            throw;
        }
    }

    ~VulkanDiffEngine() override { _release(); }

    VulkanDiffEngine(const VulkanDiffEngine &) = delete;
    VulkanDiffEngine & operator=(const VulkanDiffEngine &) = delete;

    std::string description() const override
    {
        return "vulkan (" + _device_name + ", device " + std::to_string(_device_index) + ")";
    }

    std::vector<cv::Mat> & next_frames() override
    {
        Slot & slot = _slots[_submitted % kSlots];
        if (slot.busy) throw std::logic_error("vulkan engine: next_frames() called while the slot is in flight");
        for (size_t i = 0; i < _targets.size(); ++i)
        {
            const SlotVideo & video = slot.videos[i];
            // hand out a header to the mapped buffer of the last frame layout, the decoder
            // writes into it if the next frame has the same size (the common case)
            if (video.source.mapped && video.rows > 0)
            {
                _targets[i] = cv::Mat(video.rows, video.cols, CV_8UC3, video.source.mapped, video.step);
            }
            else
            {
                _targets[i].release();
            }
        }
        return _targets;
    }

    void submit(const std::vector<VideoOptions> & video_options) override
    {
        Slot & slot = _slots[_submitted % kSlots];
        if (slot.busy) throw std::logic_error("vulkan engine: submit() called while the slot is in flight");

        std::array<PushConstants, kMaxVideoCount> parameters {};
        for (size_t i = 0; i < _targets.size(); ++i)
        {
            const cv::Mat & frame = _targets[i];
            SlotVideo & video = slot.videos[i];
            VideoState & state = _videos[i];
            if (frame.empty() || frame.type() != CV_8UC3) throw std::runtime_error("vulkan engine: expected 8-bit BGR frames");
            const int rows = frame.rows;
            const int cols = frame.cols;
            if (static_cast<uint32_t>(rows) > _max_workgroups)
            {
                throw std::runtime_error("vulkan engine: frame height " + std::to_string(rows) + " exceeds the device limit");
            }

            // source: every row starts at a 4 byte boundary and holds whole groups of 4 pixels (12 bytes)
            const size_t step = align_up(3 * align_up(static_cast<size_t>(cols), 4), 16);
            const bool in_place = frame.data == video.source.mapped && video.rows == rows && video.cols == cols
                               && frame.step[0] == step;
            if (!in_place)
            {
                const VkDeviceSize needed = static_cast<VkDeviceSize>(step) * static_cast<VkDeviceSize>(rows);
                if (video.source.size < needed)
                {
                    _destroy_buffer(video.source);
                    video.source = _create_buffer(needed, {
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    });
                }
                video.rows = rows;
                video.cols = cols;
                video.step = step;
                cv::Mat destination(rows, cols, CV_8UC3, video.source.mapped, step);
                frame.copyTo(destination);
            }

            const VkDeviceSize row_bytes = static_cast<VkDeviceSize>(rows) * sizeof(uint32_t);
            if (video.row_counts.size < row_bytes)
            {
                _destroy_buffer(video.row_counts);
                video.row_counts = _create_buffer(row_bytes, {
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                });
            }

            // grayscale ping pong buffers, stay on the device
            const bool compare = state.has_previous && state.rows == rows && state.cols == cols;
            const VkDeviceSize gray_bytes = static_cast<VkDeviceSize>(align_up(static_cast<size_t>(cols), 4))
                                          * static_cast<VkDeviceSize>(rows);
            if (state.gray[0].size < gray_bytes)
            {
                // the buffers may still be used by the submission in flight
                check(_dfn.vkDeviceWaitIdle(_device), "vkDeviceWaitIdle");
                for (Buffer & gray : state.gray)
                {
                    _destroy_buffer(gray);
                    gray = _create_buffer(gray_bytes, { VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0 });
                }
            }
            state.rows = rows;
            state.cols = cols;

            const Buffer & previous_gray = state.gray[1 - state.current];
            const Buffer & current_gray = state.gray[state.current];
            const std::array<VkDescriptorBufferInfo, 4> buffers = {{
                { video.source.buffer, 0, VK_WHOLE_SIZE },
                { previous_gray.buffer, 0, VK_WHOLE_SIZE },
                { current_gray.buffer, 0, VK_WHOLE_SIZE },
                { video.row_counts.buffer, 0, VK_WHOLE_SIZE },
            }};
            std::array<VkWriteDescriptorSet, 4> writes {};
            for (uint32_t b = 0; b < 4; ++b)
            {
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = video.descriptor_set;
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1;
                writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[b].pBufferInfo = &buffers[b];
            }
            _dfn.vkUpdateDescriptorSets(_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

            parameters[i].width = static_cast<uint32_t>(cols);
            parameters[i].height = static_cast<uint32_t>(rows);
            parameters[i].src_row_words = static_cast<uint32_t>(step / 4);
            parameters[i].threshold = static_cast<uint32_t>(std::max(0, video_options[i].pixel_difference));
            parameters[i].compare = compare ? 1u : 0u;

            video.compared = compare;
            video.result_rows = rows;
            video.result_cols = cols;
        }

        // record
        check(_dfn.vkResetCommandBuffer(slot.command_buffer, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo begin {};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(_dfn.vkBeginCommandBuffer(slot.command_buffer, &begin), "vkBeginCommandBuffer");

        // the previous submission wrote the grayscale image we read now (and reads the one we write)
        VkMemoryBarrier before {};
        before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        before.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        _dfn.vkCmdPipelineBarrier(slot.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                , 0, 1, &before, 0, nullptr, 0, nullptr);

        _dfn.vkCmdBindPipeline(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);
        for (size_t i = 0; i < _targets.size(); ++i)
        {
            _dfn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline_layout
                                       , 0, 1, &slot.videos[i].descriptor_set, 0, nullptr);
            _dfn.vkCmdPushConstants(slot.command_buffer, _pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT
                                  , 0, sizeof(PushConstants), &parameters[i]);
            _dfn.vkCmdDispatch(slot.command_buffer, parameters[i].height, 1, 1);
        }

        VkMemoryBarrier after {};
        after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        _dfn.vkCmdPipelineBarrier(slot.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT
                                , 0, 1, &after, 0, nullptr, 0, nullptr);
        check(_dfn.vkEndCommandBuffer(slot.command_buffer), "vkEndCommandBuffer");

        check(_dfn.vkResetFences(_device, 1, &slot.fence), "vkResetFences");
        VkSubmitInfo submit_info {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &slot.command_buffer;
        check(_dfn.vkQueueSubmit(_queue, 1, &submit_info, slot.fence), "vkQueueSubmit");

        slot.busy = true;
        ++_submitted;
        for (VideoState & state : _videos)
        {
            state.has_previous = true;
            state.current = 1 - state.current;
        }
    }

    size_t pending() const override { return _submitted - _collected; }
    size_t max_pending() const override { return kSlots; }

    void collect(std::vector<RowCounts> & out) override
    {
        Slot & slot = _slots[_collected % kSlots];
        if (!slot.busy) throw std::logic_error("vulkan engine: nothing to collect");
        check(_dfn.vkWaitForFences(_device, 1, &slot.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");

        out.resize(_videos.size());
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            const SlotVideo & video = slot.videos[i];
            RowCounts & result = out[i];
            result.compared = video.compared;
            result.cols = video.result_cols;
            result.counts.clear();
            if (!video.compared) continue;
            if (!video.row_counts.coherent)
            {
                VkMappedMemoryRange range {};
                range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
                range.memory = video.row_counts.memory;
                range.offset = 0;
                range.size = VK_WHOLE_SIZE;
                check(_dfn.vkInvalidateMappedMemoryRanges(_device, 1, &range), "vkInvalidateMappedMemoryRanges");
            }
            const uint32_t * counts = static_cast<const uint32_t *>(video.row_counts.mapped);
            result.counts.assign(counts, counts + video.result_rows);
        }
        slot.busy = false;
        ++_collected;
    }

private:
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        void * mapped = nullptr;
        bool coherent = true;
    };

    //! per video state which lives across submissions
    struct VideoState {
        Buffer gray[2];
        int current = 0;
        int rows = 0;
        int cols = 0;
        bool has_previous = false;
    };

    //! per video resources of one submission slot
    struct SlotVideo {
        Buffer source;
        Buffer row_counts;
        //! layout of the frame in `source`
        int rows = 0;
        int cols = 0;
        size_t step = 0;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        //! what the submission computed
        bool compared = false;
        int result_rows = 0;
        int result_cols = 0;
    };

    struct Slot {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool busy = false;
        std::vector<SlotVideo> videos;
    };

    void _release()
    {
        // (a device whose functions could not all be loaded is left to the driver)
        if (!_device || !_dfn.vkDeviceWaitIdle) return;
        _dfn.vkDeviceWaitIdle(_device);
        for (Slot & slot : _slots)
        {
            for (SlotVideo & video : slot.videos)
            {
                _destroy_buffer(video.source);
                _destroy_buffer(video.row_counts);
            }
            if (slot.fence) _dfn.vkDestroyFence(_device, slot.fence, nullptr);
        }
        for (VideoState & video : _videos)
        {
            _destroy_buffer(video.gray[0]);
            _destroy_buffer(video.gray[1]);
        }
        if (_command_pool) _dfn.vkDestroyCommandPool(_device, _command_pool, nullptr);
        if (_descriptor_pool) _dfn.vkDestroyDescriptorPool(_device, _descriptor_pool, nullptr);
        if (_pipeline) _dfn.vkDestroyPipeline(_device, _pipeline, nullptr);
        if (_pipeline_layout) _dfn.vkDestroyPipelineLayout(_device, _pipeline_layout, nullptr);
        if (_set_layout) _dfn.vkDestroyDescriptorSetLayout(_device, _set_layout, nullptr);
        _dfn.vkDestroyDevice(_device, nullptr);
        _device = VK_NULL_HANDLE;
    }

    void _select_device(int device_index)
    {
        const std::vector<VkPhysicalDevice> devices = _instance->physical_devices();
        if (devices.empty()) throw std::runtime_error("no Vulkan device found");
        if (device_index >= 0)
        {
            if (static_cast<size_t>(device_index) >= devices.size())
            {
                throw std::runtime_error("Vulkan device " + std::to_string(device_index) + " does not exist ("
                                       + std::to_string(devices.size()) + " available, see --list-vkdevices)");
            }
            _device_index = device_index;
        }
        else
        {
            // automatic: the first discrete GPU, otherwise the first device
            _device_index = 0;
            for (size_t i = 0; i < devices.size(); ++i)
            {
                VkPhysicalDeviceProperties properties;
                _instance->fn.vkGetPhysicalDeviceProperties(devices[i], &properties);
                if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                {
                    _device_index = static_cast<int>(i);
                    break;
                }
            }
        }
        _physical_device = devices[static_cast<size_t>(_device_index)];
        VkPhysicalDeviceProperties properties;
        _instance->fn.vkGetPhysicalDeviceProperties(_physical_device, &properties);
        _device_name = properties.deviceName;
        _max_workgroups = properties.limits.maxComputeWorkGroupCount[0];
        if (properties.limits.maxComputeWorkGroupSize[0] < kWorkgroupSize)
        {
            throw std::runtime_error("Vulkan device '" + _device_name + "' does not support the required workgroup size");
        }
        _instance->fn.vkGetPhysicalDeviceMemoryProperties(_physical_device, &_memory_properties);
    }

    void _create_device()
    {
        uint32_t family_count = 0;
        _instance->fn.vkGetPhysicalDeviceQueueFamilyProperties(_physical_device, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        _instance->fn.vkGetPhysicalDeviceQueueFamilyProperties(_physical_device, &family_count, families.data());
        bool found = false;
        for (uint32_t i = 0; i < family_count; ++i)
        {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                _queue_family = i;
                found = true;
                break;
            }
        }
        if (!found) throw std::runtime_error("Vulkan device '" + _device_name + "' has no compute queue");

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info {};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = _queue_family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;

        VkDeviceCreateInfo device_info {};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        check(_instance->fn.vkCreateDevice(_physical_device, &device_info, nullptr, &_device), "vkCreateDevice");

#define TRDROP_VK_LOAD_DEVICE(name) \
        _dfn.name = reinterpret_cast<PFN_##name>(_instance->fn.vkGetDeviceProcAddr(_device, #name)); \
        if (!_dfn.name) throw std::runtime_error("missing Vulkan function " #name);
        TRDROP_VK_DEVICE_FUNCTIONS(TRDROP_VK_LOAD_DEVICE)
#undef TRDROP_VK_LOAD_DEVICE

        _dfn.vkGetDeviceQueue(_device, _queue_family, 0, &_queue);
    }

    void _create_pipeline()
    {
        std::array<VkDescriptorSetLayoutBinding, 4> bindings {};
        for (uint32_t b = 0; b < 4; ++b)
        {
            bindings[b].binding = b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo set_layout_info {};
        set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        set_layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        set_layout_info.pBindings = bindings.data();
        check(_dfn.vkCreateDescriptorSetLayout(_device, &set_layout_info, nullptr, &_set_layout), "vkCreateDescriptorSetLayout");

        VkPushConstantRange push_range {};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = sizeof(PushConstants);
        VkPipelineLayoutCreateInfo layout_info {};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &_set_layout;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push_range;
        check(_dfn.vkCreatePipelineLayout(_device, &layout_info, nullptr, &_pipeline_layout), "vkCreatePipelineLayout");

        VkShaderModuleCreateInfo module_info {};
        module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        module_info.codeSize = sizeof(kRowDiffSpirv);
        module_info.pCode = kRowDiffSpirv;
        VkShaderModule module = VK_NULL_HANDLE;
        check(_dfn.vkCreateShaderModule(_device, &module_info, nullptr, &module), "vkCreateShaderModule");

        VkComputePipelineCreateInfo pipeline_info {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = _pipeline_layout;
        const VkResult result = _dfn.vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &_pipeline);
        _dfn.vkDestroyShaderModule(_device, module, nullptr);
        check(result, "vkCreateComputePipelines");
    }

    void _create_slots()
    {
        const uint32_t set_count = static_cast<uint32_t>(kSlots * _videos.size());
        VkDescriptorPoolSize pool_size {};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = 4 * set_count;
        VkDescriptorPoolCreateInfo pool_info {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = set_count;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        check(_dfn.vkCreateDescriptorPool(_device, &pool_info, nullptr, &_descriptor_pool), "vkCreateDescriptorPool");

        VkCommandPoolCreateInfo command_pool_info {};
        command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        command_pool_info.queueFamilyIndex = _queue_family;
        check(_dfn.vkCreateCommandPool(_device, &command_pool_info, nullptr, &_command_pool), "vkCreateCommandPool");

        for (Slot & slot : _slots)
        {
            slot.videos.resize(_videos.size());

            VkCommandBufferAllocateInfo command_info {};
            command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            command_info.commandPool = _command_pool;
            command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            command_info.commandBufferCount = 1;
            check(_dfn.vkAllocateCommandBuffers(_device, &command_info, &slot.command_buffer), "vkAllocateCommandBuffers");

            VkFenceCreateInfo fence_info {};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            check(_dfn.vkCreateFence(_device, &fence_info, nullptr, &slot.fence), "vkCreateFence");

            for (SlotVideo & video : slot.videos)
            {
                VkDescriptorSetAllocateInfo set_info {};
                set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                set_info.descriptorPool = _descriptor_pool;
                set_info.descriptorSetCount = 1;
                set_info.pSetLayouts = &_set_layout;
                check(_dfn.vkAllocateDescriptorSets(_device, &set_info, &video.descriptor_set), "vkAllocateDescriptorSets");
            }
        }
    }

    //! creates a storage buffer in the first memory type matching one of the property sets (in order)
    Buffer _create_buffer(VkDeviceSize size, std::initializer_list<VkMemoryPropertyFlags> preferred)
    {
        Buffer result;
        result.size = size;
        VkBufferCreateInfo buffer_info {};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = size;
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(_dfn.vkCreateBuffer(_device, &buffer_info, nullptr, &result.buffer), "vkCreateBuffer");

        VkMemoryRequirements requirements;
        _dfn.vkGetBufferMemoryRequirements(_device, result.buffer, &requirements);

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
        if (type_index == UINT32_MAX)
        {
            _dfn.vkDestroyBuffer(_device, result.buffer, nullptr);
            throw std::runtime_error("vulkan engine: no suitable memory type");
        }

        VkMemoryAllocateInfo allocate_info {};
        allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate_info.allocationSize = requirements.size;
        allocate_info.memoryTypeIndex = type_index;
        const VkResult allocated = _dfn.vkAllocateMemory(_device, &allocate_info, nullptr, &result.memory);
        if (allocated != VK_SUCCESS)
        {
            _dfn.vkDestroyBuffer(_device, result.buffer, nullptr);
            check(allocated, "vkAllocateMemory");
        }
        try
        {
            check(_dfn.vkBindBufferMemory(_device, result.buffer, result.memory, 0), "vkBindBufferMemory");
            const VkMemoryPropertyFlags properties = _memory_properties.memoryTypes[type_index].propertyFlags;
            result.coherent = (properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
            if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            {
                check(_dfn.vkMapMemory(_device, result.memory, 0, VK_WHOLE_SIZE, 0, &result.mapped), "vkMapMemory");
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
        if (buffer.mapped) _dfn.vkUnmapMemory(_device, buffer.memory);
        if (buffer.buffer) _dfn.vkDestroyBuffer(_device, buffer.buffer, nullptr);
        if (buffer.memory) _dfn.vkFreeMemory(_device, buffer.memory, nullptr);
        buffer = Buffer();
    }

private:
    std::unique_ptr<Instance> _instance;
    VkPhysicalDevice _physical_device = VK_NULL_HANDLE;
    int _device_index = 0;
    std::string _device_name;
    uint32_t _max_workgroups = 65535;
    VkPhysicalDeviceMemoryProperties _memory_properties {};
    uint32_t _queue_family = 0;
    VkDevice _device = VK_NULL_HANDLE;
    VkQueue _queue = VK_NULL_HANDLE;
    DeviceFunctions _dfn;

    VkDescriptorSetLayout _set_layout = VK_NULL_HANDLE;
    VkPipelineLayout _pipeline_layout = VK_NULL_HANDLE;
    VkPipeline _pipeline = VK_NULL_HANDLE;
    VkDescriptorPool _descriptor_pool = VK_NULL_HANDLE;
    VkCommandPool _command_pool = VK_NULL_HANDLE;

    std::vector<VideoState> _videos;
    std::array<Slot, kSlots> _slots;
    std::vector<cv::Mat> _targets;
    size_t _submitted = 0;
    size_t _collected = 0;
};

} // namespace

std::vector<VulkanDeviceInfo> list_vulkan_devices()
{
    Instance instance;
    std::vector<VulkanDeviceInfo> result;
    for (VkPhysicalDevice device : instance.physical_devices())
    {
        VkPhysicalDeviceProperties properties;
        instance.fn.vkGetPhysicalDeviceProperties(device, &properties);
        result.push_back({ properties.deviceName, device_type_name(properties.deviceType) });
    }
    return result;
}

std::unique_ptr<DiffEngine> make_vulkan_engine(int device_index, size_t video_count)
{
    return std::make_unique<VulkanDiffEngine>(device_index, video_count);
}

} // namespace trdrop
