#pragma once

// What the pipelines that analyze on the GPU share: the constants the compute shaders are called with and
// how a motion video scene is handed to overlay_draw.comp. The Vulkan Video pipeline (FFmpeg, desktop) and
// the MediaCodec pipeline (Android) run the same shaders on the same buffer layouts.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "motion.h"
#include "motion_video_target.h"

namespace trdrop {

//! must match yuv_diff.comp
struct PushConstants {
    uint32_t width;
    uint32_t luma_row_words;
    uint32_t threshold;
    uint32_t chroma_mode;
    uint32_t shift_x;
    uint32_t shift_y;
    uint32_t chroma_row_bytes;
    uint32_t v_offset;
};

//! must match motion_pyramid.comp
struct PyramidPush {
    uint32_t mode;
    uint32_t factor;
    uint32_t src_width;
    uint32_t src_height;
    uint32_t src_base;
    uint32_t dst_width;
    uint32_t dst_height;
    uint32_t dst_offset;
};

constexpr uint32_t kMaxMotionLevels = 4;

//! must match motion_lk.comp
struct LkPush {
    uint32_t level_width[kMaxMotionLevels];
    uint32_t level_height[kMaxMotionLevels];
    uint32_t level_offset[kMaxMotionLevels];
    uint32_t levels;
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t iterations;
    float max_residual;
};

//! the motion is measured at the grid of points of motion.h (kMotionPoints of them), at its resolution
constexpr uint32_t kMotionIterations = 20; // at most, per level
constexpr uint32_t kMotionGroupSize = 256; // local_size_x of motion_lk.comp
//! the frames have to match at a point after the tracking (mean absolute difference)
constexpr float kMaxResidual = 12.0f;

//! pyramid of the luma plane a frame's motion is measured on
struct MotionLayout {
    uint32_t scale = 1;
    MotionGrid grid;
    uint32_t levels = 0;
    uint32_t width[kMaxMotionLevels] = {};
    uint32_t height[kMaxMotionLevels] = {};
    uint32_t offset[kMaxMotionLevels] = {};
    uint64_t bytes = 0;
};

inline MotionLayout motion_layout(int width, int height)
{
    MotionLayout layout;
    layout.scale = static_cast<uint32_t>(motion_scale(width, height));
    layout.grid = motion_grid(width, height);
    uint32_t w = static_cast<uint32_t>(width) / layout.scale;
    uint32_t h = static_cast<uint32_t>(height) / layout.scale;
    uint32_t offset = 0;
    while (layout.levels < kMaxMotionLevels && (layout.levels == 0 || std::min(w, h) >= 32))
    {
        layout.width[layout.levels] = w;
        layout.height[layout.levels] = h;
        layout.offset[layout.levels] = offset;
        offset += w * h;
        w = (w + 1) / 2;
        h = (h + 1) / 2;
        ++layout.levels;
    }
    layout.bytes = static_cast<uint64_t>(offset) * sizeof(float);
    return layout;
}

//! must match overlay_base.comp
struct OverlayBasePush {
    uint32_t width;
    uint32_t height;
    uint32_t row_words;
    uint32_t scale;
    uint32_t luma_row_words;
    uint32_t chroma_mode;
    uint32_t chroma_row_bytes;
    uint32_t v_offset;
    uint32_t chroma_step_x;
    uint32_t chroma_step_y;
};

//! must match overlay_draw.comp
struct OverlayDrawPush {
    uint32_t width;
    uint32_t height;
    uint32_t row_words;
    uint32_t tile_size;
    uint32_t tiles_x;
    uint32_t tiles_y;
    uint32_t box_x;
    uint32_t box_y;
    uint32_t box_width;
    uint32_t box_height;
    uint32_t text_row_words;
    float dim;
    float box_dim;
};

//! the segments of a motion video frame are sorted into tiles of this size (a multiple of 4)
constexpr uint32_t kOverlayTile = 32;

//! a motion video scene as the buffers and the constants of overlay_draw.comp
struct OverlayUpload {
    //! 8 floats per segment: x0, y0, x1, y1, radius, Y, U, V
    std::vector<float> segments;
    //! per tile where its segments start in the index list (one more entry than tiles), then the indices
    std::vector<uint32_t> tiles;
    //! coverage of the text, rows padded to whole uints
    std::vector<uint8_t> text;
    OverlayDrawPush push {};
};

//! width, height, row_words: of the picture (see overlay_base.comp).
//! bt709: the colors of the segments become YUV with the BT.709 matrix instead of BT.601 (limited range)
inline void build_overlay_upload(const MotionVideoScene & scene, uint32_t width, uint32_t height, uint32_t row_words, bool bt709, OverlayUpload & upload)
{
    const int tiles_x = static_cast<int>((width + kOverlayTile - 1) / kOverlayTile);
    const int tiles_y = static_cast<int>((height + kOverlayTile - 1) / kOverlayTile);
    const size_t tile_count = static_cast<size_t>(tiles_x) * static_cast<size_t>(tiles_y);
    // the tiles a segment reaches into; overlay_draw.comp fades a segment out a pixel beyond its radius
    auto tiles_of = [&](const MotionVideoSegment & segment, int & x0, int & x1, int & y0, int & y1) {
        const float reach = segment.radius + 1.5f;
        const float tile = static_cast<float>(kOverlayTile);
        x0 = static_cast<int>(std::floor((std::min(segment.x0, segment.x1) - reach) / tile));
        x1 = static_cast<int>(std::floor((std::max(segment.x0, segment.x1) + reach) / tile));
        y0 = static_cast<int>(std::floor((std::min(segment.y0, segment.y1) - reach) / tile));
        y1 = static_cast<int>(std::floor((std::max(segment.y0, segment.y1) + reach) / tile));
        if (x1 < 0 || y1 < 0 || x0 >= tiles_x || y0 >= tiles_y) return false;
        x0 = std::max(x0, 0);
        y0 = std::max(y0, 0);
        x1 = std::min(x1, tiles_x - 1);
        y1 = std::min(y1, tiles_y - 1);
        return true;
    };
    std::vector<uint32_t> & tiles = upload.tiles;
    tiles.assign(tile_count + 1, 0);
    int x0, x1, y0, y1;
    for (const MotionVideoSegment & segment : scene.segments)
    {
        if (!tiles_of(segment, x0, x1, y0, y1)) continue;
        for (int y = y0; y <= y1; ++y)
        {
            for (int x = x0; x <= x1; ++x) tiles[static_cast<size_t>(y * tiles_x + x) + 1] += 1;
        }
    }
    for (size_t t = 0; t < tile_count; ++t) tiles[t + 1] += tiles[t];
    // the offsets stay in front, a copy of them counts up while the indices are filled in
    std::vector<uint32_t> next(tiles.begin(), tiles.begin() + static_cast<std::ptrdiff_t>(tile_count));
    tiles.resize(tile_count + 1 + tiles[tile_count], 0);
    upload.segments.resize(scene.segments.size() * 8);
    for (size_t s = 0; s < scene.segments.size(); ++s)
    {
        const MotionVideoSegment & segment = scene.segments[s];
        const float r = segment.red, g = segment.green, b = segment.blue;
        float * data = upload.segments.data() + 8 * s;
        data[0] = segment.x0;
        data[1] = segment.y0;
        data[2] = segment.x1;
        data[3] = segment.y1;
        data[4] = segment.radius;
        if (bt709)
        {
            data[5] = 16.0f + (46.559f * r + 156.629f * g + 15.812f * b) / 255.0f;
            data[6] = 128.0f + (-25.664f * r - 86.336f * g + 112.0f * b) / 255.0f;
            data[7] = 128.0f + (112.0f * r - 101.730f * g - 10.270f * b) / 255.0f;
        }
        else
        {
            // as OpenCV converts BGR to YUV
            data[5] = 16.0f + (65.481f * r + 128.553f * g + 24.966f * b) / 255.0f;
            data[6] = 128.0f + (-37.797f * r - 74.203f * g + 112.0f * b) / 255.0f;
            data[7] = 128.0f + (112.0f * r - 93.786f * g - 18.214f * b) / 255.0f;
        }
        if (!tiles_of(segment, x0, x1, y0, y1)) continue;
        for (int y = y0; y <= y1; ++y)
        {
            for (int x = x0; x <= x1; ++x)
            {
                const size_t tile = static_cast<size_t>(y * tiles_x + x);
                tiles[tile_count + 1 + next[tile]++] = static_cast<uint32_t>(s);
            }
        }
    }

    OverlayDrawPush & push = upload.push;
    push = OverlayDrawPush();
    push.width = width;
    push.height = height;
    push.row_words = row_words;
    push.tile_size = kOverlayTile;
    push.tiles_x = static_cast<uint32_t>(tiles_x);
    push.tiles_y = static_cast<uint32_t>(tiles_y);
    push.dim = scene.dim;
    push.box_dim = scene.box_dim;
    upload.text.clear();
    const cv::Rect picture(0, 0, static_cast<int>(width), static_cast<int>(height));
    if (!scene.box.empty() && (scene.box & picture) == scene.box && scene.text.size() == scene.box.size() && scene.text.type() == CV_8UC1)
    {
        push.box_x = static_cast<uint32_t>(scene.box.x);
        push.box_y = static_cast<uint32_t>(scene.box.y);
        push.box_width = static_cast<uint32_t>(scene.box.width);
        push.box_height = static_cast<uint32_t>(scene.box.height);
        push.text_row_words = (push.box_width + 3) / 4;
        upload.text.assign(static_cast<size_t>(push.text_row_words) * 4 * push.box_height, 0);
        for (int y = 0; y < scene.box.height; ++y)
        {
            std::memcpy(upload.text.data() + static_cast<size_t>(y) * push.text_row_words * 4, scene.text.ptr(y), static_cast<size_t>(scene.box.width));
        }
    }
}

} // namespace trdrop
