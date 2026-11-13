---
name: trdrop-pp
description: Analyze captured game or screen video with trdrop++ (the `trdrop-pp` command) - the real framerate, frametimes, duplicate frames and frame tears of a recording, and with `--motion` how evenly the animation advances (animation error, perceived smoothness). Use this skill whenever the user wants to measure or compare the framerate, frame pacing, stutter, judder, smoothness or tearing of a video file (capture card, OBS, console or PC capture; .mkv / .mp4 / .avi), asks whether a game "really" runs at N fps or whether frame generation (DLSS / FSR / ...) output is smooth, wants framerate / frametime / smoothness charts or a motion vector overlay made from a video, or mentions trdrop, trdrop++ or trdrop-pp - even if they don't name the tool.
---

# trdrop++

trdrop++ looks at a recording of a game (or of any screen) frame by frame and reports what the recording
really shows, independent of what an on-screen counter claims:

| question | what answers it |
| --- | --- |
| How many new pictures per second? | framerate (`vN_fps`, "effective fps") |
| How long does a picture stay on screen? | frametime (`vN_frametime_ms`) |
| Are there torn frames? | tears (`vN_tears`, `--tears`) |
| Does the motion advance evenly, i.e. does it *look* smooth? | `--motion`: animation error, perceived smoothness |
| At which resolution was an upscaled picture rendered? | `--resolution`: the `vN resolution:` lines of the summary |

The last one is the reason to prefer this tool over counting frames: 60 new frames per second whose
animation advances by 0.5, 1.5, 0.5, 1.5 ... intervals has a perfect framerate and frametime graph, but
looks like 48 fps. Only `--motion` sees that, so use it unless the user only wants the framerate.

## Find the program and the scripts

1. `trdrop-pp` (Windows: `trdrop-pp.exe`) on the `PATH`.
2. Otherwise relative to this skill's folder. In an installed package the layout is
   `bin/`, `scripts/`, `skills/trdrop-pp/` - so the program is `../../bin/trdrop-pp` and the Python scripts
   are in `../../scripts/`. (Linux / macOS install: the scripts are in `../../scripts/` too, the program is in
   `<prefix>/bin`.) In a source checkout the scripts are in `../../scripts/` and the program is in the build
   folder.
3. If neither exists, ask the user where trdrop++ is installed instead of searching the disk.

Check with `trdrop-pp --version`. `trdrop-pp --help` lists every option and CSV column and is the
authority if this text and the program disagree.

The plot scripts need Python with matplotlib (`python -m pip install -r <scripts>/requirements.txt`).

## Run an analysis

```
trdrop-pp <video> --motion -o <name>.csv 2> <name>.log
```

- The per frame data goes to the CSV. Without `-o` it goes to stdout, which floods the terminal.
- Everything meant for people - which settings were chosen, the summary - goes to stderr. Keep it (`2> file`,
  or read it from the tool output): the summary is what you report, and the first lines say how the video
  was treated (codec, lossy or not, which decoder, GPU or CPU).
- `-q` drops the progress and the `v1: <file> (...)` line that names the codec, the recorded framerate and the
  chosen tolerance; the summary stays. Leave it off when writing the log to a file - that line is needed for
  the plots (`--fps`) and for judging the result.
- Paths with spaces or non-ASCII characters work, quote them.
- Speed: roughly 150 - 250 analyzed frames per second for 4K video with `--motion`, far more for smaller
  video. A one minute 4K60 recording takes 15 - 30 seconds. For a first look at a long recording add
  `--max-frames 1800`.

Up to three videos can be given at once. They are read in lockstep (frame 0 with frame 0, ...) until the
shortest ends, each gets its own columns (`v1_`, `v2_`, `v3_`) and summary lines. Use this to compare captures
that start at the same moment; analyze them separately otherwise.

## Read the summary

```
v1: effective 29.87 fps | new 119, duplicate 120, torn 0 (0 tears) | fps avg 30.00 min 30.00 max 30.00 | frametime avg 33.33 ms max 33.33 ms
v1 motion: measured 119 of 119 new frames (100.0%) | 30.00 new frames/s | perceived smoothness 24.13 fps (10% low 24.12, whole video 24.13) | animation error median 16.43 ms avg 16.33 ms p95 16.69 ms max 18.28 ms (spread of the votes on a step: 0.85 ms)
```

First line, from comparing pixels:

- **effective fps**: new frames over the whole clip. `fps avg / min / max` is the same over a rolling
  second and the better number to quote (it leaves out the first second).
- **new / duplicate / torn**: a recording at 60 fps of a game running at 30 fps has one duplicate after
  every new frame. That is normal, not a defect of the recording.
- **frametime**: how long a picture stayed. `max` far above `avg` is a hitch.

Second line, from the motion between frames (only with `--motion`):

- **measured N of M new frames**: the share of new frames whose motion could be measured. Everything else on
  the line describes only those. Below about 50% say so and treat the rest as a sample, not the whole video.
- **perceived smoothness**: the framerate that evenly paced motion would need to look as smooth. Equal to
  the framerate if the pacing is even, lower if not. It is the median over one second windows; `10% low` is
  the bad stretches; `whole video` is one number over all steps, which a single hitch pulls down a lot.
- **animation error**: per new frame, how far the animation advanced minus how long the previous picture
  was shown. 0 is perfect pacing. Read the median / p95 as "typical / bad" unevenness in milliseconds.
- **spread of the votes**: how much the parts of the picture disagreed about a step. This is the
  measurement's own uncertainty: animation errors smaller than this say little.

A game's own fps counter and these numbers often disagree, and that is usually the finding rather than a
mistake: frame generation and repeated frames raise the counter but not what the recording shows.

For what the columns mean exactly, the formulas, and how to compute other statistics from the CSV, read
`references/metrics.md`.

## Plot

```
python <scripts>/smoothness.py <name>.csv --fps 60 -o <name>_smoothness.png          # needs a --motion CSV
python <scripts>/framerate.py <name>.csv --fps 60 -o <name>_framerate.png
python <scripts>/frametime_histogram.py <name>.csv --percent -o <name>_frametimes.png
```

- `--fps` is the recorded framerate of the video (it is in the `v1:` line of the log); it turns the x axis
  into seconds. Always pass `-o`, without it the script tries to open a window.
- `--labels "PS5,Xbox"` names the videos in the legend, `--title` sets the title. Keep both ASCII unless you
  know the default font has the glyphs.
- `smoothness.py` is the one to show first: framerate against perceived smoothness, the animation error of
  every new frame over time, and its histogram.
- Look at the image before describing it.

## Show what was measured

```
trdrop-pp <video> --motion-video <name>_motion.mp4 -o <name>.csv 2> <name>.log
```

`--motion-video` (implies `--motion`) also writes a copy of the video with the motion drawn on it: an arrow
per tracked point, a red x where a match was rejected, and the values of every frame as text (new /
duplicate, animation step, animation error, framerate, perceived smoothness). Use it when the user wants to
see what was measured, or to understand why a stretch could not be measured. `--arrow-scale 8` for slow
motion, where the arrows would only be a pixel or two long.

It needs nothing but trdrop++ itself and doesn't change the numbers. With Vulkan Video decoding (the
default) the video is drawn and encoded on the GPU and costs next to nothing; the `v1 motion video:` line
says `drawn on the GPU` then. If it says `drawn on the CPU` (other decoders, or a GPU that can't encode
H.264 with Vulkan) the run takes two to three times as long - leave the option off there when only the
numbers are wanted. With several input videos give one file per video, comma separated.

## Report

Lead with the answer to what was asked (for example "the recording shows about 15 new frames per second,
not the 60 the counter displays"), then the numbers behind it, then the charts. Give framerate and
perceived smoothness side by side when they differ - the gap is the interesting part. State what could not
be measured (the measured share, stretches without motion) instead of leaving it out. Say which file each
number came from when several were analyzed.

## Choosing options

| situation | what to do |
| --- | --- |
| H.264 / HEVC / AV1 recording (OBS, capture software) | nothing: lossy codecs are detected and compared with a tolerance. The log says `tolerance-pct auto` |
| Lossless capture (FFV1, MagicYUV, Ut Video, raw) | nothing: every different pixel counts |
| A lossy recording reports far more new frames than plausible | compression noise is counted as change: raise `--tolerance-pct` (start with 2) or `--pixel-diff` (start with 8) step by step and watch the duplicate count |
| Small real changes are missed in a mostly static picture (a static camera, a menu) | the automatic tolerance allows for them only where the recording has duplicates that change little (recorded above the framerate, a good bitrate). Otherwise lower it: `--tolerance-pct 0.1`, or `--tolerance-pct 0 --pixel-diff 0` for a lossless source in a lossy container |
| The container's framerate is wrong | `--source-fps <fps>` |
| The capture was not in step with the game's output (variable refresh rate output, or recorded with a capture program like OBS) and the summary shows a `note:` about animation errors of one whole interval | `--async-capture` instead of `--motion`, see below |
| Many tears reported on a picture with black bars or a static HUD | expected: a boundary between unchanged and changed rows counts as a tear. `--dismiss-tear-pct` raises the size a tear must have |
| No GPU, a GPU error, or results must not depend on the GPU | `--no-vk` (CPU only), or `--decode ffmpeg` |
| Several GPUs | `--list-vkdevices`, then `--vkdevice <n>` |
| The user asks for the render resolution / whether the picture is upscaled (DLSS, FSR, a console's dynamic resolution) | `--resolution` (implies `--motion`). It prints what it measured - a pixel grid at r cycles per pixel means a render scale of r or 1 - r, the picture entering at the edge of a camera pan may tell which - and `candidate N:` lines, each a render resolution (or the two of a grid) with a confidence (high / medium / low: levels of the rules, not probabilities). Report all candidates with their confidence, not only the first. With `--resolution-data <name>.res.csv` it also writes the spectra behind them; `python <scripts>/resolution.py <name>.res.csv -o <name>_resolution.png` draws them (add `--around <frequency>` to look at one peak) - do that when there is no candidate or only low ones, and look for a narrow peak at the same frequency in the columns and the rows. Needs a recording at the game's output resolution with a constant quality; the best material is a slow, even sideways pan of the camera. If a HUD touches the left or right edge between 9% and 74% of the height, move the measured part with `--resolution-rows <from>-<to>` (percent of the height) |
| Timings look wrong | another program is using the GPU; the results are unaffected, only the speed |

## Captures that are not in step with the game

A capture with its own clock (OBS and similar programs, a variable refresh rate output recorded at a fixed
rate) records some pictures an interval late and loses others. In the recording that looks like a repeated
picture, later a step twice as long, and animation errors of exactly one interval (8.3 ms at 120 fps,
16.7 ms at 60 fps) in both directions - although the game was on time. trdrop++ prints a `note:` after the
motion line when it sees many of those.

`--async-capture` (implies `--motion`) reads such a recording as what the game showed: display times count
as known to one interval only, and pictures the capture must have lost are counted in. The motion line then
ends with `asynchronous capture: N pictures re-timed by an interval, M lost pictures counted in (K of them
after a repeated picture)`, the CSV has a `vN_lost` column, and `smoothness.py --fps` adds the lost pictures
to the framerate. The first summary line stays what the recording contains.

Use it only when the capture is known to be out of step - ask the user how the video was recorded if the
note appears and it isn't clear. A game with bad frame pacing (pictures really shown an interval late) makes
the same recording, and with the option that is no longer reported; animation errors below one interval
disappear too. When you use it, say so in the report, give the numbers with and without it, and mention how
many pictures were counted in: a capture that loses pictures should be fixed rather than corrected.

## What it can not do

Knowing these keeps the report honest:

- `--motion` needs motion it can track: a moving camera, or moving objects covering at least about 2% of
  the picture. A static scene has no animation to measure - the motion line then reports few measured frames,
  and that is the correct answer. Fire, smoke, particles and water are not trackable.
- Single animation errors of real footage are good to about 10 ms. Trust the distribution (median, p95,
  histogram) and the perceived smoothness, not one frame's value.
- A camera that speeds up or slows down within about half a second looks like uneven pacing.
- If only a rectangular part of the picture changes while the rows above and below stay the same (a video
  playing inside a still page, a game window on a still desktop), the *pixel based* framerate reads low
  because such frames count as torn. The motion line is the better source then.
- It measures the recording. Whether unevenness comes from the game or from the capture can not be told from
  the file alone; `--async-capture` is an assumption about the capture, not a measurement of it.
- `--resolution` was checked against 25 recordings with a known setting. Where it found a pixel grid the
  scale was right. Which of the two scales (r or 1 - r) the pan told for DLSS at 33% and the stair steps of
  the entering picture for XeSS at 33% (the `stair steps` line: it needs straight, nearly level edges in the
  picture and a pan); with an "upward fan" (the summary says so) the larger scale goes first, which was right
  four times out of four but is not understood. It missed TSR at 25% and most recordings at 50%, and it reports what the game rendered, not
  what was set (FSR 3 set to 25% and 33% looked exactly like 50%). Report its measurements and call the
  resolutions candidates. "No candidate" does not mean native resolution: 25% and 75% hide behind the
  encoder's own grid, 50% shows only if its peak is not small against it, and a strongly compressed
  recording loses the trace. A pixel grid also appears when the
  console or the capture program scaled the picture, and a grid in one axis only can be a HUD or a pattern.
