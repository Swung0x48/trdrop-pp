# trdrop-cli

Command line version of [trdrop](https://github.com/cirquit/trdrop), a raw video analysis program.
It estimates the *real* framerate and frametime of an uncompressed capture and detects frame tears,
using the same algorithms as the original GUI, and writes the raw per frame data as CSV.

No Qt dependency. The only dependency is OpenCV (`core`, `imgproc`, `videoio`).
The overlay rendering / image sequence export of the GUI is intentionally not part of the CLI.

## Usage

```
trdrop-pp [options] <video> [<video> [<video>]]
```

| option | description |
| --- | --- |
| `-o, --output <file>` | per frame CSV (default / `-`: stdout) |
| `--tears <file>` | additional CSV with one row per detected tear |
| `--pixel-diff <0-255>` | grayscale difference a pixel may have without counting as different (default `0`) |
| `--dismiss-tear-pct <%>` | tears smaller than this percentage of the frame height are dismissed (default `10`) |
| `--source-fps <fps>` | override the recorded framerate of the container |
| `--max-frames <n>` | stop after n frames |
| `--no-summary` | don't print the summary to stderr |
| `-q, --quiet` | no progress / input info on stderr |

`--pixel-diff`, `--dismiss-tear-pct` and `--source-fps` take a single value for all videos or
comma separated values per video, e.g. `--pixel-diff 0,5,0`.

Only the CSV is written to stdout, progress / input info / summary go to stderr, so the output
can be piped directly into the [plot scripts](#plotting).

Up to 3 videos are read in lockstep; the analysis stops when the shortest video ends.
Use lossless captures: with lossy codecs (h264, mjpeg, ...) compression noise makes nearly every
frame "different" unless `--pixel-diff` is raised.

### Example

```
trdrop-pp fixtures/stutter.avi -o stutter.csv --tears stutter_tears.csv
```

```
v1: fixtures/stutter.avi (320x180, 60 fps, 240 frames reported, pixel-diff 0, dismiss-tear-pct 10)
frame 240 (100.0%)
analyzed 240 frames
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

The histogram counts every new frame once (the `vN_frametime_ms` value at rows with `vN_diff > 0`).
Frametimes are multiples of `1000 / recorded fps`, so by default every distinct value gets its own bar;
`--bins N` switches to N equal width bins.

Common options: `--labels a,b,c` legend names per video, `--title`, `-o file.png` saves the
plot instead of opening a window (no display needed), `--dpi`.
`framerate.py`: `--fps` shows the x axis in seconds (and hides the first second, where the framerate
window is still filling up), `--skip N` hides the first N frames, `--max-fps` fixes the y axis.
`frametime_histogram.py`: `--percent` share of frames instead of counts, `--bins N`, `--log`.

## Algorithm

Identical to trdrop's `FrameProcessing`:

1. Both frames are converted to grayscale, a pixel is different if `|a - b| > pixel-diff`.
2. For every row the share of different pixels is computed.
3. A tear is the boundary between an unchanged row and a row with >= 10% different pixels.
   The biggest tear defines the torn part of the frame; if it's below `dismiss-tear-pct` the frame
   counts as fully new, otherwise the frame counts as `1 - torn part`. Complementing tears
   therefore don't change the framerate (e.g. `0.6 + 0.4`).
4. The framerate is the sum of these values within the last `ceil(recorded fps)` frames,
   the frametime is the time between the last two frame changes in that window.

Note: every boundary between an unchanged row and a changed row counts as a tear. On real captures
with static black bars or HUD elements this produces tears on most new frames (same as the GUI).

Differences to the GUI:

* All frames are analyzed from the first one (the GUI consumed the first frame for its preview).
* The CSV is overwritten instead of appended to, and contains additional columns.
* Fixed: tears of multiple videos were mixed when computing the torn part, and a difference
  in the last row of a frame was ignored.

## Building

Requirements: CMake >= 3.21, a C++17 compiler and OpenCV 4.

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

### Linux / macOS

```
sudo apt install libopencv-dev   # or: brew install opencv
cmake -S . -B build
cmake --build build
```

## Tests

`trdrop_make_fixtures <dir>` (built by default, disable with `-DTRDROP_BUILD_TOOLS=OFF`) writes
lossless test videos with a known pattern:

| file | content | expected |
| --- | --- | --- |
| `30in60.avi` | 60 fps container, new frame every 2nd frame | 30 fps, 33.3 ms |
| `stutter.avi` | frames held 1, 2, 3, 2, ... frames | ~30 fps, 16.7 - 50 ms |
| `tear.avi` | 30 fps content, every new frame first appears torn at 40% | 30 fps, diffs 0.6 / 0.4, tear at row 71/72 |
| `60.avi` | every frame new | 60 fps, 16.7 ms |

```
build/trdrop_make_fixtures test-out/fixtures
build/trdrop-pp test-out/fixtures/tear.avi --tears test-out/tears.csv
```

## License

MIT, see [LICENSE](LICENSE). Based on trdrop by Alexander Isenko.
