# trdrop++

Analyzes captured game video. It estimates the *real* framerate and frametime of a capture, detects frame
tears and, from the motion between the frames, measures how evenly the animation advances
([perceived smoothness](#motion-analysis---motion)). The per frame data is written as CSV.
With `--resolution` it also estimates the resolution an upscaled picture was
[rendered at](#render-resolution---resolution).
The program and the repository are called `trdrop-pp`, `pp` being the `++` that file names and URLs don't like.

The only required dependency is OpenCV (`core`, `imgproc`, `videoio`, `video`).
Optionally the per pixel work runs on the GPU via Vulkan ([build with `-DUSE_VULKAN=ON`](#vulkan-gpu-acceleration)),
and HEVC / H.264 captures (e.g. OBS / NVENC recordings) are decoded on the GPU as well ([`-DUSE_FFMPEG=ON`](#ffmpeg--vulkan-video-gpu-decoding)).

## Usage

```
trdrop-pp [options] <video> [<video> [<video>]]
```

| option | description |
| --- | --- |
| `-o, --output <file>` | per frame CSV (default / `-`: stdout) |
| `--tears <file>` | additional CSV with one row per detected tear |
| `--pixel-diff <0-255>` | grayscale difference a pixel may have without counting as different (default `0`, lossy codecs `4`) |
| `--tolerance-pct <%>` | for [lossy codecs](#lossy-recordings): a frame only counts as new if more than this percentage of its pixels is different (default `0`, lossy codecs: automatic) |
| `--dismiss-tear-pct <%>` | tears smaller than this percentage of the frame height are dismissed (default `10`) |
| `--source-fps <fps>` | override the recorded framerate of the container |
| `--max-frames <n>` | stop after n frames |
| `--motion` | [motion analysis](#motion-analysis---motion): animation step, animation error and perceived smoothness |
| `--motion-window <ms>` | fixed length of the window the motion of a frame is compared to (default: 400, longer for low framerates) |
| `--motion-video <file>` | `--motion`, and write a [copy of the video with the motion drawn on it](#motion-video---motion-video) (`.mp4`, `.mkv`, `.mov`; one file per input video, comma separated) |
| `--arrow-scale <x>` | draw the arrows of the motion video x times longer (default `1`) |
| `--async-capture` | `--motion` for a [capture that didn't run in step with the game's output](#captures-that-are-not-in-step---async-capture): pictures recorded an interval late don't count as animation error, pictures the capture lost are counted in |
| `--resolution` | `--motion`, and estimate the [render resolution](#render-resolution---resolution) of an upscaled picture |
| `--no-summary` | don't print the summary to stderr |
| `-q, --quiet` | no progress / input info on stderr |
| `--vkdevice <n>` | Vulkan device to use (default: the first discrete GPU, otherwise device `0`). Vulkan builds only |
| `--no-vk` | don't use Vulkan, analyze with OpenCV on the CPU (accepted and ignored in builds without Vulkan) |
| `--list-vkdevices` | list the Vulkan devices with their `--vkdevice` index and exit. Vulkan builds only |
| `--decode <mode>` | `auto` (default: the fastest), `vulkan`, `ffmpeg` or `opencv`, see [GPU decoding](#ffmpeg--vulkan-video-gpu-decoding). FFmpeg builds only |

`--pixel-diff`, `--tolerance-pct`, `--dismiss-tear-pct` and `--source-fps` take a single value for all videos or
comma separated values per video, e.g. `--pixel-diff 0,5,0`.

Only the CSV is written to stdout, progress / input info / summary go to stderr, so the output
can be piped directly into the [plot scripts](#plotting).

Up to 3 videos are read in lockstep; the analysis stops when the shortest video ends.
Lossless captures give exact results. Lossy recordings (H.264, HEVC, ...) are detected and analyzed
with a tolerance, see [lossy recordings](#lossy-recordings).

### Example

```
trdrop-pp fixtures/stutter.avi -o stutter.csv --tears stutter_tears.csv
```

```
v1: fixtures/stutter.avi (320x180, FFV1, 60 fps, 240 frames reported, pixel-diff 0, tolerance-pct 0, dismiss-tear-pct 10)
backend: opencv decode, vulkan (NVIDIA GeForce RTX 4070 SUPER, device 0)
frame 240 (100.0%)
analyzed 240 frames in 0.35 s (681.2 fps) | analysis 13.1 ms (0.055 ms/frame) | opencv decode, vulkan (NVIDIA GeForce RTX 4070 SUPER, device 0)
v1: effective 29.87 fps | new 119, duplicate 120, torn 0 (0 tears) | fps avg 29.99 min 29.00 max 31.00 | frametime avg 33.52 ms max 50.00 ms
```

### Per frame CSV

One row per frame, one column group per input video (`v1_`, `v2_`, `v3_`):

```
frame,v1_diff,v1_fps,v1_frametime_ms,v1_tears
0,,0,0,0
1,0,0,0,0
2,0.6,0.6,0,1
3,0.4,1,0,1
```

| column | meaning |
| --- | --- |
| `frame` | frame index, starting at 0 |
| `vN_diff` | difference to the previous frame: `0` duplicate, `1` new frame, `(0,1)` torn frame (`1 - torn part`). Empty for frame 0 |
| `vN_fps` | sum of `vN_diff` over the last second (`ceil(recorded fps)` frames) - the "real" framerate |
| `vN_frametime_ms` | how long the last completed frame was shown |
| `vN_tears` | amount of tears detected in this frame |

### Tear CSV

```
frame,video,row_a,row_b,video_height
2,1,71,72,180
```

A tear lies between the pixel rows `row_a` and `row_b` of video number `video`.

### Summary (stderr)

Per video: effective framerate over the whole clip (`sum(diff) / compared frames * recorded fps`),
counts of new / duplicate / torn frames and the min / avg / max of the rolling framerate and frametime
after the first full one second window.

### Lossy recordings

The basic rule counts a frame as new as soon as a single pixel is different. That is right for
lossless captures, but a lossy encoder changes pixels of identical source frames (noise, and blocks
predicted from the next frame), so nearly every frame would count as new - e.g. a 4K H.264 recording
whose picture only changed every 4th frame was reported with 58 instead of 14 fps.

`--tolerance-pct <%>`: a frame only counts as new if more than this percentage of its pixels is
different (by more than `--pixel-diff`); for the tear detection a row with at most this percentage of
different pixels counts as unchanged. With `0` every different pixel counts.

Without the option the codec decides (printed in the `vN:` line):

* lossless codecs (FFV1, HuffYUV, Ut Video, MagicYUV, raw, ...): `--tolerance-pct 0`
* lossy codecs: `--pixel-diff 4` (unless given) and an automatic tolerance: at least 0.5%, and 5% of
  what a new frame recently changed (the 90th percentile of the changed pixels per frame over the last
  2 seconds). The relative part is needed for fast motion, where a few percent of the next frame leak
  into duplicate frames, while a fixed high value would hide the small changes of a static scene.
  Rows always use 0.5%.

  Checked against a lossless capture with 271 duplicates in 899 frames (41.9 fps): its 100 Mbit/s HEVC
  version gives 281 duplicates (41.1 fps) - and 0 duplicates (60 fps) without the tolerance.

  0.5% is what the duplicates of a poor recording need, but it hides small real changes: a static camera
  with a few distant cars, a menu with a blinking cursor. Where the duplicates of the recording change
  less, the minimum is therefore lower - 45 times what the calmer third of the recent duplicates changed,
  between 0.05% and 0.5% - and the relative part looks at the last second only. That needs:

  * duplicates to measure, i.e. a recording at more than the framerate
    ([twice the framerate](#recording-at-twice-the-framerate) is enough): 30 of them, until then it is 0.5%
  * a decoder that tells which frames are keyframes. FFmpeg decoding does (`--decode vulkan` and `ffmpeg`),
    `--decode opencv` doesn't and stays at 0.5% - that is also what a video without an 8-bit YUV format
    (10-bit, HDR) is decoded with; the Android app was not checked. Keyframes are coded anew, a duplicate
    changes 0.1 - 10% of the picture there, so they keep 0.5% (as does a recording that consists of keyframes
    only, e.g. MJPEG): a small change that falls on a keyframe is missed, and a duplicate on one may count
    as a new frame (as before)

  Measured on a 1080p240 capture of a replay with 9211 new frames in 22849 - among them 58 from 0.6 seconds
  of a static camera with distant cars, which change 0.2 - 0.4% of the picture - and on re-encodes of it.
  Which frames are new was told from the capture itself (the 0.5% rule, plus every frame in which more than
  0.005% of the pixels change by more than 12: every picture is then recorded 2 to 5 times):

  | recording | 0.5% throughout: missed / wrongly new | lower minimum: missed / wrongly new |
  | --- | --- | --- |
  | as captured: HEVC, constant quality, 37 Mbit/s | 58 / 0 | 0 / 0 |
  | H.264 with 2 B-frames, constant quality, 23 Mbit/s | 60 / 31 | 0 / 31 |
  | HEVC with 3 B-frames, constant quality, 19 Mbit/s | 63 / 28 | 0 / 28 |
  | HEVC, 6 Mbit/s | 69 / 51 | 24 / 55 |
  | HEVC, 2 Mbit/s | 86 / 107 | 69 / 116 |
  | H.264 with 2 B-frames, 3 Mbit/s | 211 / 4066 | 211 / 4052 |

  In the constant quality re-encodes all wrongly new frames are duplicates on keyframes. `motion_ball_small`
  (`tools/make_ball_videos.py`: 20 fps in 60, the ball changes 0.2% of the picture) reads 20 fps after its
  first half second, and 0 fps with 0.5% throughout.

H.264 / HEVC / AV1 are always treated as lossy, also if they were encoded in a lossless mode
(pass `--tolerance-pct 0 --pixel-diff 0` then). The automatic values are heuristics: after fast motion
small real changes may count as duplicates for up to 2 seconds (1 second where the minimum is lower), a
picture without any duplicates around it (recorded at the framerate) keeps the 0.5%, and tears can't be
detected reliably in lossy recordings.

## Motion analysis (`--motion`)

The framerate only counts new frames. `--motion` also measures how evenly the animation advances,
i.e. whether the video looks smooth. It needs motion it can track - a moving camera, scrolling, driving,
or moving objects in at least 2% of the picture (12 of its 640 cells); the rest may be static.

```
trdrop-pp capture.mkv --motion -o capture.csv
python scripts/smoothness.py capture.csv --fps 60 -o smoothness.png
```

How it works:

1. Motion vectors between consecutive frames, sub pixel accurate (pyramidal Lucas-Kanade on the luma at up
   to 1920x1080) at a regular grid of 64x40 points. A point is used if its surroundings have texture and if
   tracking it back from where it arrived ends within 1 pixel of the point - wrong matches (repeating texture,
   motion too large) don't come back. The points are reduced to the median vector per cell of a 32x20 grid
   (4 points per cell, at least 2 of them). A video that is higher than wide - a recording of a phone's
   screen - is measured at up to 1080x1920 with the grid turned (40x64 points, 20x32 cells).
   * GPU (with Vulkan Video decoding): compute shaders (`src/vulkan/motion_pyramid.comp`, `motion_lk.comp`),
     the frames stay on the GPU and only the tracked points are read back.
   * CPU (`--decode ffmpeg` / `opencv`): OpenCV.

   Sparse motion: a camera that follows a car over plain asphalt leaves a kerb and a few markings to track,
   and the grid misses most of what lies between its points. If the grid finds motion - a cell with 2 points
   that moved by 0.5 pixels - in fewer than 12 cells, points are added where the previous frame has texture:
   every cell is cut into 8x8 parts, and of each part the point with the most texture is tracked as well
   (texture: the smaller eigenvalue of the gradient matrix over 7x7 pixels at half the resolution, at least
   0.5 per pixel). Cells whose points of the grid were tracked and didn't move - the car - get none. A frame
   without motion, a duplicate, is treated like the last frame with motion, so the cells that measure the
   motion have their points in it too. In such a frame the points of a cell have to agree: the cell counts if
   most of its points, and at least 2, moved like their median (within 20% of it or half a pixel), and those
   points are the cell - wrong matches rarely agree, and many more points bring many more of them. Frames with
   motion in 12 cells or more are measured by the grid alone, as before. (GPU: `motion_select.comp` decides,
   `motion_texture.comp` picks the points; both only run for such frames.)
2. Animation step of a frame: the common scale of its motion field relative to the average motion field
   in a window around it (triangular weights): `m_cell = step * v_cell`. Every cell with motion (at least
   0.5 pixels per frame on average) has one vote, `m.v / |v|^2`, and the weighted median of the votes is the
   step. So pans, forward motion (the cells move in different directions), static overlays and a picture of
   which only a part moves all work: a frame needs 12 voting cells (2% of the picture). `1` means the frame
   advanced as much as a display interval does on average, `0` a duplicate, `2` twice as much. The steps are
   scaled so that the animation advances by one interval per displayed interval on average over the window.
   The cells never agree exactly in real footage (the depth of what a cell shows changes, objects move on
   their own). No frame is dropped for that; instead the spread of the votes (their scaled median absolute
   deviation) is written next to the step.
   The window is 400 ms and grows up to 1 second until it holds 10 new frames: with a low framerate a
   short window only holds a few steps, their average follows them and hides their differences.
   `--motion-window <ms>` sets a fixed length.
3. Animation error of a new frame: its animation step (summed since the previous new frame) minus the
   time that frame was displayed, in ms. A 30 fps animation in a 60 fps video has no error (step 2 after
   2 intervals); steps of 0.5 / 1.5 intervals have -8.3 / +8.3 ms.
4. Perceived smoothness: the framerate of evenly paced motion that looks as smooth - the new frames per
   second divided by `1 + cv^2`, `cv` being the coefficient of variation of the steps
   (`fps * sum(step)^2 / (displayed intervals * sum(step^2))`). It equals the framerate of the new frames if
   they are evenly paced and drops with uneven steps (30 new frames per second with steps 1, 3, 1, 3: 24 fps).

Additional CSV columns (empty where not available):

| column | meaning |
| --- | --- |
| `vN_anim_step` | animation step of the frame in display intervals; empty without usable motion |
| `vN_anim_step_spread` | how much the moving parts of the picture disagree about the step, in display intervals (`0`: rigid motion). A difference between two steps which is smaller than this says little |
| `vN_anim_error_ms` | new frames: animation error |
| `vN_perceived_fps` | perceived smoothness over the last second (needs 6 steps and usable motion in half of it) |
| `vN_lost` | only with `--async-capture`, new frames: pictures the capture lost between the previous new frame and this one |

and a summary line per video:

```
v1 motion: measured 239 of 239 new frames (100.0%) | 60.00 new frames/s | perceived smoothness 48.02 fps (10% low 47.98, whole video 48.01) | animation error median 8.32 ms avg 8.32 ms p95 8.61 ms max 8.95 ms (spread of the votes on a step: 0.12 ms)
```

The perceived smoothness is the median of the values over one second each (and their 10th percentile),
which a single long step hardly changes; "whole video" is the same formula over all steps. The spread
is the median of `vN_anim_step_spread` over the new frames, as time.

Accuracy with known answers (H.264 versions of the [fixtures](#tests); GPU and CPU agree within 0.1 fps / 0.1 ms):

| video | expected | measured |
| --- | --- | --- |
| even pan / forward motion | 60 fps, error 0 | 60.00 fps, avg 0.1 ms |
| new frame every 2nd frame | 30 fps, error 0 | 30.00 fps, avg 0.2 ms |
| steps 0.5 / 1.5 | 48 fps, error 8.33 ms | 48.0 fps, avg 8.3 ms |
| the same, but only a window of 5% of the picture moves | 48 fps, error 8.33 ms | 47.9 fps, avg 8.4 ms |
| sparse motion: ten balls of 8 pixels radius pass a large one that stands still, a new frame every 2nd frame which advances by 1 / 3 (`motion_ball_sparse`, see `tools/make_ball_videos.py`) | 24 fps, errors of 16.7 ms | after its first second (in which the duplicates are measured, see [lossy recordings](#lossy-recordings)): 24.3 fps (CPU) / 24.1 fps (GPU), median 16.4 / 16.5 ms. The grid alone measures 22% (CPU) / none (GPU) of the new frames |
| freeze of 5 frames, time lost | one error of -83.3 ms | -75.8 ms |
| new frame every 4th frame, steps of 3 / 4 / 5 intervals, fast pan / forward motion | 14.55 fps, errors -16.7 / 0 / +16.7 ms | 14.66 fps, the errors are off by 3.2 ms on average |
| real footage: a 4K 60 fps game recording re-timed to a new frame every 4th frame which advances by 3 / 4 / 5 source frames (truth: the steps the 60 fps source itself shows) | 13.3 fps, errors of 17.5 ms on average | 13.4 fps (CPU) / 13.8 fps (GPU), the errors are off by 11 ms on average (median 8 ms) and 3 ms too positive |

Speed with the motion analysis (decoding + comparison + motion, RTX 4070 SUPER / i7-14700KF):

| video | CPU (`--decode ffmpeg`) | GPU (default) |
| --- | --- | --- |
| 4K H.264 recordings | 127 - 174 fps | 236 - 244 fps |
| 1440p HEVC capture | 236 fps | 807 fps |

Frames with sparse motion take longer, on the CPU mostly for the texture of the frame: a 1080p240 capture with
9 seconds of them in 95 runs at 840 instead of 940 fps on the GPU and at 220 instead of 260 fps on the CPU; a
recording of a phone's screen at 550 instead of 730 fps (GPU); a 4K recording that consists of such frames (a
static camera, a character, a fire) at 78 instead of 170 fps on the CPU, on the GPU as before (235 fps).

Limits:

* A frame is compared to the average motion around it, which assumes that the real motion changes slowly
  within the window: a camera that speeds up can't be told from frames that advance further. Hard stops look
  like errors; around a freeze the neighbouring frames get a few ms of error (the freeze lowers the average).
* Single animation errors of real footage should be read with care: with the known answer above they are off
  by about 10 ms, which is also what the spread of the votes says there (9 - 10 ms; 0.1 ms for the rigid
  fixtures). The spread doesn't contain the first limit. The perceived smoothness averages over many steps
  and doesn't depend on their scale; it was within 1 - 3% of the truth.
* On a 4K driving recording (15 - 17 fps content in a 60 fps video, with a menu at 60 fps) 88% (GPU) / 90%
  (CPU) of the new frames are measured, both agree on the perceived smoothness within 1% (14.5 / 14.4 fps) and
  on the steps of the frames both measure (correlation 0.91).
* Only what can be tracked counts as motion: the camera, characters, vehicles. Flames, particles, water and
  the like can't. In a recording with a static camera, a walking character and a fire, 17% (GPU) / 20% (CPU)
  of the new frames are measured - those in which the character covers enough of the picture. There GPU and
  CPU agree on the perceived smoothness (57.5 / 57.0 fps), less on the single steps (correlation 0.66).
* Sparse motion: on a 1080p240 capture of a replay, 5 seconds follow a car head on over plain asphalt, with a
  kerb and a few markings passing by. Of the 592 new frames of that shot the grid alone measures 0 (GPU) / 5
  (CPU), with the points added where the picture has texture 577 (GPU) / 411 (CPU): perceived smoothness 119
  fps at 120 fps, animation errors of 0.4 ms (median). What is added are corners: a straight line has no
  texture along itself, and the frames in which only such lines or nothing at all passes by stay unmeasured.
  Nothing helps where nothing moves by half a pixel per frame (a static camera with distant cars).
  The added points change other frames with little motion as well. Against the known answer of the re-timed
  recording above the errors are off by 11.9 instead of 11.7 ms on the GPU and by 10.7 instead of 10.6 ms on
  the CPU. On eight other recordings the perceived smoothness moved by up to 0.3 fps (GPU) / 0.4 fps (CPU).
  Single steps move more next to a cut into a shot that only the added points measure: the frames of the 0.2
  seconds before it are compared with an average motion that now includes that shot (in one recording the
  steps of 20 frames changed by more than 0.2 intervals, some to half their value).
* Motion blurred repeating texture (grass, gravel passing by) gives matches that stay where they are and
  still return when tracked back: in one shot 29% of the used points, with any density of points. Cells with
  such points pull the step of a frame; the more of them a frame has, the less its step is worth - the spread
  of the votes shows it.
* Parts of the picture which update at different rates (e.g. a 30 fps animation in a 60 fps camera pan) are
  not separated, the larger part wins.
* It can't tell whether uneven steps come from the game or from the recording. If the recording is known to be
  out of step with the game's output, `--async-capture` (below) takes that into account;
  [recording at twice the framerate](#recording-at-twice-the-framerate) avoids it.
* The GPU path needs Vulkan Video decoding (`--decode vulkan`, or `auto`, which prefers it with `--motion`);
  other sources (e.g. lossless RGB) are measured on the CPU.
* The rows of the CSV are written with a delay of up to half a second (the look ahead of the window).

### Motion video (`--motion-video`)

```
trdrop-pp capture.mkv --motion-video capture_motion.mp4 -o capture.csv
```

writes a copy of the video that shows what the motion analysis saw, frame by frame:

| | |
| --- | --- |
| arrow | a point of the grid that was tracked: from where it was in the previous frame to where it is now, colored by its length (blue: short, red: 45 pixels and more). On a duplicate frame the arrows of the last new frame stay, dimmed |
| dot | a tracked point that didn't move |
| red x | a point whose match doesn't come back, it is not used |
| nothing | no texture at the point |
| text | the row of the frame: new / duplicate / torn, points used and rejected, animation step and the spread of its votes, animation error, framerate and perceived smoothness. With `--async-capture` also where a picture was lost |

The video has the resolution the motion is measured at (the source divided by a whole number, 1920x1080 at
most: 4K becomes 1080p; 1080x1920 at most for a video that is higher than wide: 1080x2400 becomes 540x1200)
and the framerate of the source; the picture is darkened so the arrows stand out.
`--arrow-scale 8` draws the arrows 8 times longer, for slow motion where they would be a pixel or two.
With several input videos give one file each, comma separated; an empty name skips a video
(`--motion-video ,b.mp4`).

The analysis is the same with and without the option (the CSV is identical), on every decoder: the arrows
are the points this run measured, on the GPU or on the CPU. Where the video is made follows the decoder,
the `vN motion video:` line says which it is:

| decoding | picture, drawing | encoding | 4K 120 fps HEVC recording, frames per second |
| --- | --- | --- | --- |
| Vulkan Video (the default with `--motion`) | on the GPU | on the GPU: H.264 with Vulkan Video (`h264_vulkan`) | 595 (601 without the video) |
| Vulkan Video on a GPU without Vulkan H.264 encoding | the planes of the frame are copied into memory, drawn on the CPU | as below | about 200 |
| `--decode ffmpeg` / `opencv` | on the CPU | H.264 with the first encoder that works on the machine - the GPU's (NVENC, AMF, Quick Sync, VideoToolbox), x264 if the FFmpeg build has it, the one of Windows (Media Foundation), OpenH264 - and MPEG-4 if none does | 95 (156 without the video) |
| builds without FFmpeg | on the CPU | what OpenCV can: MPEG-4 (`.mp4`, `.mkv`, `.mov`) or Motion JPEG (`.avi`) | |

On the GPU a frame never leaves it: a compute shader scales it down into a picture that waits there until
the analysis has the values of the frame (it looks ahead), a second one dims it and draws the arrows and
the text box on it, and the result is copied into a frame of the encoder. Only the arrows, the text (four
lines, rasterized on the CPU) and the encoded video cross between CPU and GPU. Both ways draw the same
scene and look the same apart from the anti-aliasing of the lines.

The pictures that wait take up to 200 MB at 4K 120 fps - of GPU memory, or of main memory when drawn on
the CPU. The container follows from the name.

### Captures that are not in step (`--async-capture`)

A capture card that records a console at a fixed 60 Hz in step with its output sees every picture exactly as
long as it was shown. A capture that has its own clock does not - a variable refresh rate output recorded at
a fixed rate, or a capture program (OBS, ...) that composes its frames at its own pace from what the card
delivers. A picture shown at time t is first recorded at the next tick of the capture, up to one interval
later, and a picture that is shown for less than an interval may not be recorded at all. In the recording
that looks like trouble the game doesn't have:

```
shown by the game   A  B  C  D  E  F  G  H  I     120 fps, evenly paced
recorded            A  B  B  C  D  E  G  H  I     B twice (C was a moment late for its tick), F lost
animation step         1  0  1  1  1  2  1  1
animation error        0    -1  0  0 +1  0  0     in intervals: C after B was recorded twice, G after E once
```

A repeated picture, later a step twice as long, and in between an animation error of -1 and +1 interval;
the framerate and the perceived smoothness drop although every picture of the game was on time. Without the
option the summary points it out if many errors are one whole interval, early and late alike.

With `--async-capture` (it implies `--motion`):

* The time a picture was shown counts as known to one interval only. The drift between animation time and
  recorded time may move freely within a band of one interval - that is the capture - and only what pushes
  the band along is animation error. Above: 0 for every frame.
* A step about twice as long as the steps around it, after a picture that was recorded once, stands for two
  pictures of which the capture lost one (`vN_lost`). They count as two steps in the perceived smoothness and
  in the new frames per second of the motion line, and `smoothness.py` adds them to the framerate. This
  needs pictures that are shown for about one interval (1.2 at most): a picture that is shown longer is
  always recorded.
* If the repeat and the lost picture fall together (`A B B D`), the recording is the same as a game that
  showed one picture twice. While the capture demonstrably loses pictures (at least 3 in the last second)
  it counts as the capture; the summary says how many these are.
* The summary line ends with `asynchronous capture: 1404 pictures re-timed by an interval, 933 lost pictures
  counted in (218 of them after a repeated picture)`.

The first line of the summary and `vN_diff` / `vN_fps` / `vN_frametime_ms` stay what the recording contains.

What it costs: a game that really shows a picture an interval late and makes up for it afterwards (bad frame
pacing: 30 fps shown for 1, 3, 1, 3 intervals) produces exactly the same recording, and animation errors of
less than one interval peak to peak disappear altogether (`motion_ball_uneven`: 0 instead of 8.3 ms; its
perceived smoothness stays 48 fps, the steps are still uneven). So the option is a statement about the
capture which the video can't confirm: use it if you know that the capture is not in step, not to make a
result look better. Better still, fix the capture ([below](#recording-at-twice-the-framerate)) - no analysis
brings back a picture that was not recorded.

Measured (`motion_ball_capture` from `tools/make_ball_videos.py`: an even 60 fps, 10 times per second one
picture recorded twice and another one lost; and a 4K 120 fps recording of a game locked to 120 fps, made
with a capture program that lost and repeated about 20 pictures per second):

| video | as recorded | with `--async-capture` |
| --- | --- | --- |
| `motion_ball_capture` | 50.1 fps, perceived 45.2 fps, animation error avg 5.1 ms | 59.8 fps, perceived 60.0 fps, avg 0.2 ms |
| game recording, 120 fps | 99.3 fps, perceived 90.4 fps, avg 2.9 ms (31% of the errors are +-8.3 ms) | 119.5 fps, perceived 119.5 fps, avg 0.4 ms |

### Recording at twice the framerate

A recording that needs no such assumption: record at twice the highest framerate of the source or more - a
120 Hz output at 240 fps, at a lower resolution if the capture card asks for it. Every picture is then shown
for at least two intervals of the capture and is recorded whatever the two clocks do, also with a variable
refresh rate. Analyze it as it is; `--async-capture` isn't needed (what it changes: below).

For example with FFmpeg on Windows, straight from the card (the first line lists the devices). A capture
program that composes at its own framerate (OBS, ...) would have to run at the doubled rate too; that was not
tried.

```
ffmpeg -list_devices true -f dshow -i dummy
ffmpeg -f dshow -rtbufsize 2G -video_size 1920x1080 -framerate 240 -pixel_format nv12 -i video="<device>" -fps_mode passthrough -enc_time_base 1:1000000 -c:v hevc_nvenc -preset p1 -rc constqp -qp 18 capture.mp4
```

`-fps_mode passthrough` writes every frame the card delivers once; messages like `real-time buffer too full`
or `frame dropped` mean that the encoder doesn't keep up and frames are missing.

What to expect and how to check it:

* Every picture is recorded twice or more: `vN_frametime_ms` is never one interval of the capture. If it is,
  pictures may have been lost as well - the source was faster than assumed or the card delivered fewer
  frames than it claims. Where the source runs at its highest rate the pictures are recorded exactly twice.
* The time a picture was shown is known to one interval of the capture, so the animation error carries up to
  one interval of noise (4.2 ms at 240 fps). The framerate and the perceived smoothness count pictures and
  compare their steps, they don't have that noise.
* `--async-capture` changes only the animation error of such a recording. No picture is recorded once, so
  none counts as lost, and the framerate and the perceived smoothness stay as they are. Its band of one
  interval takes the noise of the capture out of the error - and real errors of less than an interval with
  it. The error without the option is an upper bound, the one with it a lower bound.
* Don't expect more from the timestamps of the file: the card this was tried with numbers its frames on the
  240 fps grid whenever they arrive. trdrop++ doesn't read them.

Twice is not a threshold. No picture is lost as long as an interval of the capture is shorter than the
shortest time a picture is shown, so 144 fps would do for a 120 Hz output. What the factor buys:

* The check above. At 1.2 times the rate a picture that is recorded once is normal (38% of the pictures
  below), a lost one doesn't stand out.
* Time resolution: an interval of 6.9 ms instead of 4.2 ms, the noise in the animation error doubles.
* Margin: the source and the capture don't run at exactly the rates they name.

Measured: a console game with variable refresh rate (about 47 - 120 fps, 102 seconds) recorded at 1080p
240 fps with a USB capture card. Of its 8532 pictures none was recorded once (37% twice, 44% three times,
15% four times); in the five seconds in which the game is fastest (115 fps) 91% of the pictures are recorded
twice. For the second row the recording was resampled to 144 fps (every tick takes the latest picture, as a
capture would; a real capture at 144 fps wasn't tried).

| video | new frames per second | perceived smoothness (10% low) | animation error, average / 95th percentile | the error with `--async-capture` |
| --- | --- | --- | --- | --- |
| recorded at 240 fps | 83.7 | 81.9 fps (57.5) | 1.3 ms / 3.3 ms | 0.15 ms / 0.8 ms |
| resampled to 144 fps | 83.8 | 82.0 fps (57.6) | 2.7 ms / 5.8 ms | 0.20 ms / 1.4 ms |

The other values are the same with `--async-capture`, no picture counts as lost in either video.

## Render resolution (`--resolution`)

Games render fewer pixels than they put out and scale the picture up - with a plain filter, or with a temporal
upscaler (DLSS, FSR 2 and later, XeSS, TSR, PSSR) that builds its detail from many frames. `--resolution`
(it implies `--motion`) looks for two traces of that and prints what it found:

```
trdrop-pp capture.mkv --resolution -o capture.csv
```

```
v1 resolution: pixel grid at 0.3333 cycles/pixel (13.1x / 8.6x the background in the columns / rows): rendered at 33.3% or 66.7% of the output
v1 resolution: picture entering at the edge of a pan, 1401 + 786 frames (moving right + left): contrast against the picture that leaves down to 0.31 (at 0.266 cycles/pixel), one half at 0.165 cycles/pixel; 1.13 for what entered 8 - 15 frames ago: a single frame resolves about 33% of the output, the rest builds up over the frames (temporal upscaling)
v1 resolution: estimate 1280x720 (33.3% of 3840x2160): the scale of the pixel grid that fits the pan, not 2560x1440 (66.7%)
```

The CSV is the same with and without the option.

1. Pixel grid. In a picture that was scaled up, how much a pixel differs from its neighbours depends on where it
   lies between two source pixels, and that repeats with the source sampling rate. Per frame the mean
   `|p[x-1] - 2 p[x] + p[x+1]|` of every column (and the same of every row; without the lowest 12% of the
   picture, where overlays tend to be) is taken, the slow variation removed, and the spectra of these profiles
   are added up over the video. A picture rendered at r times the output resolution has a peak at r cycles per
   pixel, folded into 0 - 0.5: 33% and 67% both give 0.333, 59% and 41% give 0.41. It counts if the columns
   and the rows both have it 3 times above the background around it. The position is sharp - 1 / 3840 of a
   cycle for a 4K picture - so a scale of 67% (0.33) is told from one of two thirds (0.3333).
2. The picture that enters at the edge of a pan. A temporal upscaler has no earlier frames for the picture that
   enters at one edge while the camera pans: it is what a single rendered frame holds, nothing finer than the
   render resolution. The vertical spectrum (the motion blur of a horizontal pan is horizontal) of up to 256
   columns next to that edge is compared with the same columns of the other edge, where the picture leaves
   after it crossed the screen. The columns are grouped by when they entered - the summed motion of the
   frames before reaches them - and the contrast of each group is the root of the ratio of the two spectra,
   with the coarsest detail (0.047 cycles per pixel) as 1. Where the contrast of the newest columns (the outer
   half of what entered with the frame) is down to one half is read as what a single frame resolves: half its
   sampling rate, so twice that frequency is the render scale. Of the two scales the pixel grid allows, the one
   nearer to it is the estimate.
   A frame counts if the picture moves by 4 - 64 pixels (where it enters; a camera that turns moves the edges
   up to twice as fast as the middle) with little vertical motion, and if no frame of the last 20 moved the
   other way and 15 of them moved this way - then what leaves at the other edge has been on the screen for a
   while. 30 such frames are needed.

Where it is computed follows the decoder, like the motion: with Vulkan Video decoding two compute shaders
(`src/vulkan/resolution_lattice.comp`, `resolution_edge.comp`) read the luma on the GPU and 80 kB per frame
come back; otherwise the CPU does the same. The sums of the pixel grid are integers and identical on both
(checked on 1561 frames), the spectra agree to 0.001%.

Measured (4K recordings; RTX 4070 SUPER / i7-14700KF):

| recording | pixel grid | pan | estimate |
| --- | --- | --- | --- |
| DLSS at 33% (ultra performance), 4K60 HEVC with a constant QP of 1, 80 seconds with gamepad pans | 0.3333, 13.1x / 8.6x | 1401 + 786 frames, one half at 0.165 | 1280x720 - right |
| the same, 26 seconds of it: GPU / CPU (`--decode ffmpeg`) | 0.3333, 16.5x / 13.8x on both | 824 + 336 frames, 0.162 / 756 + 336 frames, 0.161 | 1280x720 on both |
| DLSS at 33%, another recording of 46 seconds with a slow pan | 0.3333, 9.6x / 6.6x | 349 frames, one half at 0.165 | 1280x720 - right |
| DLSS at 50% (performance), three H.264 recordings at 40 - 80 Mbit/s | none (2.1x - 2.2x) | 5 - 7 frames: none | none - as expected, see below |
| DLSS at 33%, two H.264 recordings of a still scene at 2 - 5 Mbit/s | none (2.5x at 0.3333; 2.3x elsewhere) | none | none - missed |
| console game, 4K120 recording whose picture was scaled from 1440p (the game, the console or the capture) | 0.3333, 6.4x / 5.6x | no pan in it | 1280x720 or 2560x1440 |
| the same game, a replay with trackside cameras | 0.3333, 4.1x / 5.9x | 75 frames, the contrast stays above 0.81 | 1280x720 or 2560x1440: the pan doesn't tell |
| the same game recorded at 1080p | none (1.1x) | | none: the console scales its picture down |

Speed: the 4K60 recording runs at 111 frames per second on the GPU with and without the option, at 52 instead
of 58 on the CPU.

Recording for it:

* at the resolution the game puts out, without any scaling on the way. A console set to 1080p scales its
  picture down, nothing of the render resolution is left then - record it apart from a recording at
  [twice the framerate](#recording-at-twice-the-framerate).
* with a constant quality, not a constant bitrate. 10 seconds of the first recording, re-encoded with HEVC
  (NVENC) at a constant QP:

  | QP | bitrate | pixel grid, columns / rows | one half at |
  | --- | --- | --- | --- |
  | 1 (as recorded) | 891 Mbit/s | 17.8x / 13.7x | 0.170 |
  | 8 | 417 Mbit/s | 13.3x / 9.8x | 0.167 |
  | 12 | 228 Mbit/s | 10.7x / 7.6x | 0.167 |
  | 16 | 129 Mbit/s | 8.4x / 5.6x | 0.169 |
  | 20 | 66 Mbit/s | 4.4x / 4.1x | 0.178 |
  | 24 | 37 Mbit/s | 3.7x / 3.2x | 0.200 |

  The estimate was 1280x720 for all of them, but at QP 20 and above the peak is close to what counts and its
  position is off by half a bin, and the pan reads higher. QP 12 - 16 is a good choice; QP 1 gives files of
  800 Mbit/s for nothing.
* in the game: stand still and turn the camera sideways at an even speed, a full turn in about 10 seconds
  (20 - 30 pixels per frame at 4K), in both directions; motion blur, depth of field, film grain, sharpening
  and frame generation off; a scene with fine horizontal structure (roof tiles, brickwork, steps).

Limits:

* It was checked against one known answer - DLSS at 33%, two recordings of the same place. Whether the pan
  measurement follows the render scale, and what FSR, XeSS or an engine's own upscaler leave, is not known
  yet: read the lines with the measurements, the estimate is their interpretation.
* A render scale of exactly 50% leaves no pixel grid (both pixel positions between two source pixels are
  alike). 25% and 75% fall on the block grid of video encoders, which every lossy recording has at 0.25: the
  summary says so instead of guessing. Native resolution leaves none either, so "no pixel grid" says little.
* The pixel grid can't tell r from 1 - r. The pan can if the two are far apart, but not above a render scale
  of about 70%: a single frame then resolves detail that the picture hardly has.
* A pan in which the entering picture keeps its contrast doesn't rule out a temporal upscaler. In three
  recordings of DLSS at 50% (H.264 at 40 - 80 Mbit/s) an earlier version of the measurement found no loss at
  any frequency in 34 - 122 pan frames; with the rules above only 5 - 7 of their frames count, so the program
  reports no pan for them. Whether that is the recordings or the method is open.
* A pixel grid says that the picture was scaled, not by whom: the game, the console's output or the capture
  (a capture program that scales its source to a larger canvas) all leave the same one.
* One scale for the whole video: a render scale that changes with the load smears the peak.
* Overlays that are drawn at the output resolution (a HUD) are part of the picture that is measured.
* The pan needs a horizontal pan the motion analysis can follow; ages are counted in recorded frames, so a
  game that runs faster than the recording has more than one of its frames in the newest columns.

## Plotting

`scripts/` contains two matplotlib scripts that read the per frame CSV, either from a file or
piped from trdrop-pp via stdin:

```
pip install -r scripts/requirements.txt

python scripts/framerate.py capture.csv
trdrop-pp capture.avi | python scripts/framerate.py --fps 60
trdrop-pp a.avi b.avi | python scripts/frametime_histogram.py --labels PS5,XSX --percent -o frametimes.png
```

| script | plot |
| --- | --- |
| `framerate.py` | framerate of all videos over time in one chart, legend with avg / min |
| `frametime_histogram.py` | one frametime histogram per video (stacked, shared x axis): how long each new frame stayed on screen, with n / avg / p99 / max |
| `smoothness.py` | for `--motion` CSVs: framerate and perceived smoothness, animation error over time (with `--fps`: the spread of the votes as a band) and as histogram. With `--fps` and an `--async-capture` CSV the framerate includes the pictures the capture lost |

The histogram counts every new frame once (the `vN_frametime_ms` value at rows with `vN_diff > 0`).
Frametimes are multiples of `1000 / recorded fps`, so by default every distinct value gets its own bar;
`--bins N` switches to N equal width bins.

Common options: `--labels a,b,c` legend names per video, `--title`, `-o file.png` saves the
plot instead of opening a window (no display needed), `--dpi`.
`framerate.py`: `--fps` shows the x axis in seconds (and hides the first second, where the framerate
window is still filling up), `--skip N` hides the first N frames, `--max-fps` fixes the y axis.
`frametime_histogram.py`: `--percent` share of frames instead of counts, `--bins N`, `--log`.

## Algorithm

Framerate, frametime and tears:

1. Both frames are converted to grayscale, a pixel is different if `|a - b| > pixel-diff`.
2. For every row the share of different pixels is computed.
3. A tear is the boundary between an unchanged row and a row with >= 10% different pixels.
   The biggest tear defines the torn part of the frame; if it's below `dismiss-tear-pct` the frame
   counts as fully new, otherwise the frame counts as `1 - torn part`. Complementing tears
   therefore don't change the framerate (e.g. `0.6 + 0.4`).
4. The framerate is the sum of these values within the last `ceil(recorded fps)` frames,
   the frametime is the time between the last two frame changes in that window.

Note: every boundary between an unchanged row and a changed row counts as a tear. On real captures
with static black bars or HUD elements this produces tears on most new frames.

## Building

Requirements: CMake >= 3.21, a C++17 compiler and OpenCV 4 (`core`, `imgproc`, `videoio`, `video`).

### Windows (MSVC)

Download the OpenCV Windows pack from the [OpenCV releases](https://github.com/opencv/opencv/releases)
(e.g. `opencv-4.12.0-windows.exe`) and extract it. From a *x64 Native Tools Command Prompt*:

```
cmake -S . -B build -G Ninja -DOpenCV_DIR=<opencv>/build/x64/vc16/lib
cmake --build build
```

Pointing `OpenCV_DIR` directly at `x64/vc16/lib` is needed for newer MSVC versions which the pack's
top level config doesn't recognize (they are ABI compatible). The OpenCV DLLs, including
`opencv_videoio_ffmpeg*.dll` for video decoding, are copied next to `trdrop-pp.exe`.

vcpkg works as well: `vcpkg install opencv4[ffmpeg]` and configure with the vcpkg toolchain file.

The executable uses UTF-8 as its code page (application manifest, Windows 10 1903 or newer), so file
names with non-ASCII characters (e.g. Chinese) work on the command line.

### Linux / macOS

```
sudo apt install libopencv-dev   # or: brew install opencv
cmake -S . -B build
cmake --build build
```

### Installing / packaging

```
cmake --install build --prefix dist
cmake --build build --target package      # the same as dist/, as build/trdrop-pp-<version>-<system>.zip
```

On Windows this gives a folder that can be copied to another PC and used as it is:

| | |
| --- | --- |
| `bin/` | `trdrop-pp.exe` and the DLLs it needs: OpenCV (with its FFmpeg plugin), with `-DUSE_FFMPEG=ON` and `FFMPEG_DIR` the FFmpeg DLLs, and the MSVC runtime. `-DTRDROP_INSTALL_RUNTIME=OFF` installs the program alone |
| `scripts/` | the [plot scripts](#plotting), `make_ball_videos.py` and `requirements.txt` (they need Python, see there) |
| `skills/` | the [agent skill](#agent-skill) |
| `licenses/` | the license of trdrop++ and those of the DLLs in `bin/` |
| `README.md` | this file |

The Vulkan loader is not included, it comes with the graphics driver. This was tested with the OpenCV
Windows pack; a vcpkg OpenCV has further DLLs of its own which these rules don't collect, use vcpkg's
deployment (`X_VCPKG_APPLOCAL_DEPS_INSTALL`) there.

On Linux / macOS the program goes to `<prefix>/bin`, the scripts and skills to `<prefix>/share/trdrop-pp/` and
README / licenses to `<prefix>/share/doc/trdrop-pp`; the libraries are those of the system.

### Vulkan (GPU acceleration)

`-DUSE_VULKAN=ON` adds a Vulkan compute backend (default `OFF`). It needs the Vulkan headers and
`glslc` at build time, e.g. from the [Vulkan SDK](https://vulkan.lunarg.com/) (`VULKAN_SDK` set) or
`apt install libvulkan-dev glslc`:

```
cmake -S . -B build-vk -G Ninja -DUSE_VULKAN=ON -DOpenCV_DIR=<opencv>/build/x64/vc16/lib
cmake --build build-vk
```

The Vulkan loader is loaded at runtime, so the binary also starts on systems without a Vulkan
driver: it prints a warning and uses the CPU (an explicit `--vkdevice` that can't be used is an error).
`--no-vk` forces the CPU path. `TRDROP_VK_VALIDATION=1` enables the Khronos validation layer for debugging.

What runs on the GPU: grayscale conversion, the difference to the previous frame, the
`--pixel-diff` threshold and the per row count of different pixels (`src/vulkan/row_diff.comp`).
The tear / framerate / frametime logic stays on the CPU. The GPU path is bit exact: it uses
OpenCV's 8-bit fixed point grayscale formula (checked against `cv::cvtColor` at startup for all
16.7M colors, otherwise the CPU is used), so the CSVs are identical to `--no-vk`.
Frames are decoded directly into GPU visible memory and the GPU works on one set of frames while
the next one is decoded.

The summary line shows the wall time and the time the main thread spent in the analysis.
Example (i7-14700KF, RTX 4070 SUPER, synthetic FFV1 videos, median of 3 runs):

| input | CPU analysis | Vulkan analysis | CPU total | Vulkan total |
| --- | --- | --- | --- | --- |
| 1366x767, 240 frames | 0.49 ms/frame | 0.10 ms/frame | 0.69 s | 0.63 s |
| 1920x1080, 600 frames | 0.77 ms/frame | 0.11 ms/frame | 2.86 s | 2.63 s |
| 3840x2160, 360 frames | 3.34 ms/frame | 0.15 ms/frame | 6.43 s | 6.43 s |
| 3x 1920x1080, 600 frames | 3.41 ms/frame | 0.17 ms/frame | 7.84 s | 6.74 s |
| real 1440x1080 captures, ~1200-1450 frames | 0.52 - 1.11 ms/frame | 0.11 - 0.26 ms/frame | | 4 - 19% less |

The analysis itself gets 4 - 20x faster, but the total is bound by video decoding (FFmpeg, on the CPU),
so the end to end gain is small: 0 - 20% depending on the resolution and codec.
For HEVC / H.264 captures that limit is removed by decoding on the GPU, see below.

### FFmpeg / Vulkan Video (GPU decoding)

`-DUSE_FFMPEG=ON` (default `OFF`, together with `-DUSE_VULKAN=ON`) decodes with FFmpeg directly. Videos
that FFmpeg can decode with Vulkan Video (HEVC, H.264 from OBS / NVENC / capture cards, ...) are
decoded on the GPU, the Y, U and V planes of every frame are copied on the GPU and compared there
(`src/vulkan/vk_video_pipeline.cpp`, `src/vulkan/yuv_diff.comp`). Only the per row counts are read
back, the frames never leave the GPU. Without `USE_VULKAN` only the `ffmpeg` (CPU) mode is available.

It needs the FFmpeg >= 8.0 development files (`libavformat`, `libavcodec`, `libavutil`), linked
dynamically, so an LGPL build is sufficient and the license of trdrop++ is not affected. On Windows
e.g. the `ffmpeg-n9.0-latest-win64-lgpl-shared-9.0.zip` of [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases),
the DLLs are copied next to `trdrop-pp.exe` (ship them together with FFmpeg's license).
On Linux pkg-config is used when `FFMPEG_DIR` isn't set.

```
cmake -S . -B build-gpu -G Ninja -DUSE_VULKAN=ON -DUSE_FFMPEG=ON -DFFMPEG_DIR=<ffmpeg dev package> -DOpenCV_DIR=<opencv>/build/x64/vc16/lib
cmake --build build-gpu
```

`--decode` selects the path:

| mode | decoding | comparison | |
| --- | --- | --- | --- |
| `auto` (default) | the fastest of the modes below, see the rules | | the choice and its reason are printed |
| `vulkan` | GPU (Vulkan Video) | Y + U + V, GPU | |
| `ffmpeg` | FFmpeg on the CPU (multi threaded) | Y + U + V, CPU (row parallel) | same results as `vulkan` |
| `opencv` | OpenCV | BGR -> grayscale, GPU or CPU | always used with `--no-vk` |

`auto` uses `vulkan` if every video can be decoded with Vulkan Video, except H.264 on CPUs with
16 or more threads: the hardware decoders of e.g. the RTX 4070 (SUPER) decode 4K H.264 slower
(~230 fps) than a 28 thread CPU (~320 fps). YUV formats without Vulkan decoding (MagicYUV,
UtVideo, raw NV12, AV1 via libdav1d, ...) use `ffmpeg`, RGB sources (e.g. FFV1 bgra) `opencv`.

YUV vs grayscale: `vulkan` / `ffmpeg` compare the decoded planes directly instead of converting to
BGR first (which dominated the time for 4K 4:2:0 sources). A pixel (x, y) changed if its luma
changed by more than `--pixel-diff`, or the chroma sample covering it (4:2:0: one per 2x2 pixels)
changed by more than `--pixel-diff` in U or V. So a change of only the color is detected as well
(lossy H.264 recordings contain frames whose luma is bit identical while the chroma changed).
The values are limited range (16 - 235 / 240), so `--pixel-diff` has a slightly different scale
and the CSVs differ slightly from `opencv`.
Supported are 8-bit YUV formats (4:2:0 / 4:2:2 / 4:4:4, NV12, ...), tested with 4:2:0 (NV12 on the
GPU, yuv420p on the CPU). 10-bit (HDR) and RGB sources use the `opencv` path. Rotation metadata is
ignored by `vulkan` / `ffmpeg`.

Speed compared to decoding and comparing with OpenCV on the CPU only (wall time incl. startup,
i7-14700KF (28 threads), RTX 4070 SUPER, no other GPU load):

| input | OpenCV, CPU | default (`auto`) | `auto` chose |
| --- | --- | --- | --- |
| 4K H.264 game recording, 1007 frames | 18.2 s | **3.3 s (5.6x)** | `ffmpeg` |
| 4K H.264 game recording, 1580 frames | 21.3 s | **5.0 s (4.3x)** | `ffmpeg` |
| 4K H.264 game recording, 3859 frames | 50.2 s | **11.9 s (4.2x)** | `ffmpeg` |
| 1440p HEVC capture, 1247 frames | 4.7 s | **1.9 s (2.5x)** | `vulkan` |
| 4K HEVC 100 Mbit/s, 900 frames | 12.1 s | **2.5 s (5.0x)** | `vulkan` |
| 4K MagicYUV (lossless), 900 frames | 13.2 s | **2.9 s (4.5x)** | `ffmpeg` |
| 1080p FFV1 bgra, 600 frames | | about the same | `opencv` |

`--decode vulkan` on the H.264 recordings: 3.0 - 3.6x. Comparing U and V as well costs no measurable
time on the GPU.

## Android app

`android/` is an app that runs the analysis on a phone: choose a video, get the summary, the charts (drawn
by the app, the same as the Python scripts draw) and the motion video. It uses the sources of this program
as a native library. Everything runs on the GPU if the phone allows it - MediaCodec decodes into a hardware
buffer, the compute shaders of the Vulkan Video pipeline analyze and draw, MediaCodec encodes -, otherwise
MediaCodec decodes and the CPU analyzes. See [android/README.md](android/README.md) for what it does, how
fast it is and how to build it.

## Agent skill

`skills/trdrop-pp/` is a skill for AI coding agents (Claude Code and others that read `SKILL.md`): how to
run an analysis, read the summary, plot and report it, which options fit which recording, and what the tool
can't measure. To use it, copy or link the folder into the agent's skills folder (Claude Code:
`~/.claude/skills/` or `<project>/.claude/skills/`) and put `trdrop-pp` on the `PATH`. In an installed
package the skill also finds the program and the scripts next to itself (`bin/`, `scripts/`).

## Tests

`trdrop_make_fixtures <dir>` (built by default, disable with `-DTRDROP_BUILD_TOOLS=OFF`) writes
lossless test videos with a known pattern:

| file | content | expected |
| --- | --- | --- |
| `30in60.avi` | 60 fps container, new frame every 2nd frame | 30 fps, 33.3 ms |
| `stutter.avi` | frames held 1, 2, 3, 2, ... frames | ~30 fps, 16.7 - 50 ms |
| `tear.avi` | 30 fps content, every new frame first appears torn at 40% | 30 fps, diffs 0.6 / 0.4, tear at row 71/72 |
| `60.avi` | every frame new | 60 fps, 16.7 ms |
| `motion_even.avi` | `--motion`: a texture panning evenly | step 1, error 0, perceived 60 fps |
| `motion_30in60.avi` | pan, new frame every 2nd frame | step 2, error 0, perceived 30 fps |
| `motion_uneven.avi` | pan, the frames advance by 0.5 / 1.5 intervals | error -8.3 / +8.3 ms, perceived 48 fps |
| `motion_hitch.avi` | pan, frames 100-104 freeze (time lost), frames 170-173 freeze and 174 catches up | error -83.3 ms at frame 105, step 5 without error at 174 |
| `motion_zoom.avi` | forward motion (zoom), even | step 1, error 0, perceived 60 fps |
| `motion_partial.avi` | a static picture, only a window of 5% of it shows the pan advancing by 0.5 / 1.5 intervals | error -8.3 / +8.3 ms, perceived 48 fps |

```
build/trdrop_make_fixtures test-out/fixtures
build/trdrop-pp test-out/fixtures/tear.avi --tears test-out/tears.csv
build/trdrop-pp test-out/fixtures/motion_uneven.avi --motion -o motion.csv
```

## License

MIT, see [LICENSE](LICENSE). The framerate, frametime and tear analysis is based on
[trdrop](https://github.com/cirquit/trdrop) by Alexander Isenko.
