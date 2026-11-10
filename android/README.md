# trdrop++ for Android

An app that analyzes a video on the phone: choose a video, get the summary, the charts and - if wanted - the
motion video. It runs the command line program of trdrop++ as a native library, so the numbers are those of
the desktop version.

| | |
| --- | --- |
| analysis | framerate, frametimes, tears, and the motion analysis (animation step, animation error, perceived smoothness), with the option for captures that are not in step with the game (`--async-capture`) |
| charts | framerate and perceived smoothness over time, frametime distribution, animation error over time (with the spread of the votes) and its distribution. Drawn by the app itself (`charts/ChartView.kt`); touch a chart for its values, zoom with two fingers, double tap to reset. The Python scripts in `scripts/` draw the same from the CSV |
| motion video | H.264 in MP4, played in the app, saved to the gallery (`Movies/trdrop`) or shared |
| files | every analysis keeps its CSV, log and video in `Android/data/io.github.swung0x48.trdrop/files/analyses/<time>/`; the CSV can be shared |

Android 8.0 or newer, arm64.

## Where the analysis runs

| | decoding | analysis | motion video |
| --- | --- | --- | --- |
| GPU | MediaCodec (the hardware decoder) into a hardware buffer, which Vulkan reads without a copy | compute shaders, the ones of the desktop's Vulkan Video pipeline | drawn by compute shaders, encoded by MediaCodec through a swapchain on the encoder's input surface |
| CPU | MediaCodec into memory (through FFmpeg) | OpenCV on the CPU, as `--decode ffmpeg` on the desktop | drawn with OpenCV, encoded by MediaCodec (through FFmpeg) |

On the GPU a frame never leaves it. Vulkan Video, which the desktop version decodes and encodes with, is
not offered by phone GPU drivers; MediaCodec is the way to the same hardware, and
`src/android/vk_mediacodec_pipeline.cpp` connects it to the shaders: `ahb_import.comp` reads the planes of
the decoder's buffer into the layout the other shaders work on, `nv12_to_rgba.comp` converts a drawn
picture for the encoder.

Three frames are in flight, as in the desktop's pipeline: the next frames are decoded and handed to the GPU
while it works on the earlier ones, and a drawn picture is presented to the encoder without waiting for it.
Given one frame at a time, a phone GPU idles between them and is clocked down for it (the Mali ran at half
its frequency and a third of the speed).

"Automatic" uses the GPU if the phone and the video allow it, and the CPU otherwise; both can be chosen by
hand in the app. Measured with a 4K 120 fps HEVC recording (5527 frames), in frames per second:

| phone | GPU | GPU, no motion video | CPU |
| --- | --- | --- | --- |
| Snapdragon (SM8975), Adreno 850 | 200 | 209 | 73 |
| Dimensity 9400 (MT6991), Mali-G925 | 94 | 129 | 44 |

The GPU is what limits both (the Mali at its highest frequency, 99 - 100% busy).

The results of the GPU are those of the desktop's GPU (the frame comparison is identical, the motion values
differ in the fifth digit), those of the CPU are those of the desktop's CPU pipeline (frame comparison
identical, animation steps within 0.03).

The GPU needs a video whose width is a multiple of 4, 8-bit 4:2:0; other videos are analyzed on the CPU.

## Building

Needed: the Android SDK with NDK 28 and CMake (`ANDROID_HOME`), a JDK (17 or newer), and two native
dependencies, by default expected in `../.deps` (paths in `gradle.properties`):

1. The OpenCV Android SDK (4.12.0), unpacked to `.deps/opencv-android/OpenCV-android-sdk`
   (https://github.com/opencv/opencv/releases, `opencv-4.12.0-android-sdk.zip`).
2. FFmpeg (release/9.0) as shared libraries with the MediaCodec codecs, built with `build-ffmpeg.sh`. It
   needs a POSIX shell with make - Linux, macOS, or MSYS2 on Windows:

   ```
   git clone --depth 1 --branch release/9.0 https://github.com/FFmpeg/FFmpeg.git .deps/ffmpeg-android/src
   sh android/build-ffmpeg.sh .deps/ffmpeg-android/src .deps/ffmpeg-android/arm64-v8a
   ```

Then

```
cd android
./gradlew assembleRelease        # Windows: powershell -File build-apk.ps1 (see the comment in it)
adb install -r app/build/outputs/apk/release/app-release.apk
```

The release build is signed with the debug key so that it installs as it is; use your own key to publish.
FFmpeg is built without `--enable-gpl` / `--enable-nonfree` (LGPL), OpenCV is Apache 2.0.

## Testing from a computer

An analysis can be started without touching the phone, its files can be pulled afterwards:

```
adb push capture.mkv /sdcard/Android/data/io.github.swung0x48.trdrop/files/
adb shell am start -n io.github.swung0x48.trdrop/.MainActivity \
    --es video /sdcard/Android/data/io.github.swung0x48.trdrop/files/capture.mkv --ez autostart true
adb pull /sdcard/Android/data/io.github.swung0x48.trdrop/files/analyses
```

More extras: `--ez async true` (capture out of step), `--ez motion_video false`, `--ez arrows true`
(arrows 8 times longer), `--ez gpu true` / `--ez cpu true`, and `--es args "'<more arguments>'"` for
arguments of the command line program (`--max-frames 600`; `--gpu-profile` times every step of the GPU
pipeline and writes them to the log).

## Limits

* One video per analysis.
* The analysis runs in the app's process: it stops if the app is closed, and the screen is kept on while it
  runs.
* If the motion video can't be made on the GPU (the encoder has no input surface), the analysis stops with a
  message; choose CPU then.
