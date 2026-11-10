# trdrop++ output in detail

Read this when a question needs more than the summary lines: what a column means exactly, how a number is
defined, or a statistic the summary doesn't give.

## Per frame CSV (`-o`)

One row per frame of the video(s). `vN_` is the column group of the N-th input video.

| column | meaning |
| --- | --- |
| `frame` | frame index, starting at 0 |
| `vN_diff` | difference to the previous frame: `0` duplicate, `1` new frame, between 0 and 1 a torn frame (`1 - torn part`). Empty for frame 0 |
| `vN_fps` | sum of `vN_diff` over the last second (`ceil(recorded fps)` frames). During the first second it is still filling up |
| `vN_frametime_ms` | how long the last completed picture was shown; a multiple of `1000 / recorded fps` |
| `vN_tears` | number of tears detected in this frame |

Only with `--motion` (empty where not available):

| column | meaning |
| --- | --- |
| `vN_anim_step` | how far the animation advanced since the previous frame, in display intervals. `1` = as much as it advances per recorded frame on average around this frame, `0` = a duplicate, `2` = twice as much. Empty: no usable motion in this frame |
| `vN_anim_step_spread` | how much the moving parts of the picture disagree about the step, in intervals (`0`: rigid motion). The uncertainty of the step |
| `vN_anim_error_ms` | new frames only: the animation step since the previous new frame minus the time that frame was displayed, in ms. `0` = the picture advanced exactly as long as the previous one was shown |
| `vN_perceived_fps` | perceived smoothness over the last second. Needs 6 steps and usable motion in half of that second |

Only with `--async-capture`:

| column | meaning |
| --- | --- |
| `vN_lost` | new frames: pictures the capture lost between the previous new frame and this one (`0` or `1`). They count as frames in `vN_perceived_fps`; `vN_diff` and `vN_fps` don't know them |

With `--async-capture`, `vN_anim_error_ms` is what is left after allowing every picture to have been shown up
to one interval before it was recorded (the drift between animation and recorded time may move within a band
of one interval, only what pushes the band counts).

The rows of a `--motion` run are written with a delay of up to half a second of video, the content is the same.

## Tear CSV (`--tears`)

`frame, video, row_a, row_b, video_height`: a tear lies between the pixel rows `row_a` and `row_b` of video
number `video`.

## Definitions

- **New frame**: lossless source - any pixel differs. Lossy source - more than the tolerance (automatic: at
  least 0.5% of the pixels, and 5% of what new frames changed recently) differ by more than `--pixel-diff`.
- **Framerate**: sum of `diff` over the last second. A torn frame counts partly, two complementing torn
  frames count as one new frame.
- **Animation step**: every part of the picture that moves compares its motion in this frame with its
  average motion per interval in a window of about 0.4 - 1 s around the frame; the median of these votes is
  the step. So `step = 1` means "one interval's worth of motion", whatever the speed.
- **Animation error** = (sum of the steps since the previous new frame - number of intervals that frame was
  shown) x interval length. Examples at 60 fps: 30 fps content, evenly paced: step 2 after 2 intervals,
  error 0. Steps alternating 0.5 / 1.5 with every frame new: errors -8.3 / +8.3 ms.
- **Perceived smoothness** = new frames per second / (1 + cv^2), cv being the coefficient of variation of
  the steps between new frames. Evenly paced: equals the framerate. 30 new frames per second with steps
  1, 3, 1, 3: 24 fps. It does not depend on the speed of the motion.

Because the step is relative to the average around the frame, a change of the real speed within the window
(the camera accelerates) looks like uneven steps. On real footage a single animation error is good to about
10 ms; the perceived smoothness averages over many steps and was within 1 - 3% in tests with a known answer.

## Statistics from the CSV

The summary covers the whole file. For a part of it, or other percentiles, read the CSV - plain Python is
enough:

```python
import csv, statistics

rows = list(csv.DictReader(open("capture.csv", newline="")))
fps = 60                      # recorded framerate, from the "v1:" line of the log
start, end = 10 * fps, 20 * fps    # seconds 10 - 20

part = rows[start:end]
new = [r for r in part if r["v1_diff"] not in ("", "0")]
errors = sorted(abs(float(r["v1_anim_error_ms"])) for r in part if r["v1_anim_error_ms"])
perceived = [float(r["v1_perceived_fps"]) for r in part if r["v1_perceived_fps"]]

print("new frames per second:", len(new) / (len(part) / fps))
if errors:
    print("animation error median / p95:", statistics.median(errors), errors[int(0.95 * (len(errors) - 1))])
if perceived:
    print("perceived smoothness median:", statistics.median(perceived))
```

Empty cells are "not available", not zero: skip them. `vN_frametime_ms` repeats on every row until the next
picture, so for a frametime distribution take it only at rows with `vN_diff > 0` (that is what
`frametime_histogram.py` does).

## Checking that the tool works

`<scripts>/make_ball_videos.py <folder>` (in a source checkout: `tools/make_ball_videos.py`) writes short videos of a ball moving at constant speed with known
pacing (needs numpy, opencv-python and ffmpeg). Expected: `motion_ball_60` 60 / 60 fps (framerate /
perceived), `_30` 30 / 30, `_15` 15 / 15, `_uneven` 60 / 48, `_stutter` 30 / 24, `_hitch` 54 / 35.3,
`_capture` (an even 60 fps through a capture that repeats and loses pictures) 50 / 45 as recorded and
60 / 60 with `--async-capture`.
