#!/bin/sh
# Cross-compiles FFmpeg as shared libraries for Android (arm64-v8a), with the MediaCodec decoders and
# encoders of the device. The app links against the result (android/app/src/main/cpp/CMakeLists.txt).
#
#   sh android/build-ffmpeg.sh <ffmpeg source dir> <output dir> [ndk dir]
#
# Needs a POSIX shell with make (Linux, macOS, or MSYS2 on Windows) and the Android NDK; the NDK is
# taken from the third argument, ANDROID_NDK_HOME or the newest one in $ANDROID_HOME/ndk.
# LGPL build: no --enable-gpl, no --enable-nonfree.
set -e

SOURCE=${1:?usage: build-ffmpeg.sh <ffmpeg source dir> <output dir> [ndk dir]}
OUTPUT=${2:?usage: build-ffmpeg.sh <ffmpeg source dir> <output dir> [ndk dir]}
NDK=${3:-$ANDROID_NDK_HOME}
API=26

if [ -z "$NDK" ] && [ -n "$ANDROID_HOME" ]; then
    NDK=$(ls -d "$ANDROID_HOME"/ndk/* 2>/dev/null | sort | tail -1)
fi
[ -d "$NDK" ] || { echo "error: Android NDK not found, pass it as the third argument" >&2; exit 1; }

case "$(uname -s)" in
    Linux*) HOST=linux-x86_64; EXE= ;;
    Darwin*) HOST=darwin-x86_64; EXE= ;;
    *) HOST=windows-x86_64; EXE=.exe ;;
esac
TOOLCHAIN=$NDK/toolchains/llvm/prebuilt/$HOST
[ -x "$TOOLCHAIN/bin/clang$EXE" ] || { echo "error: no clang in $TOOLCHAIN/bin" >&2; exit 1; }

SOURCE=$(cd "$SOURCE" && pwd)
mkdir -p "$OUTPUT"
OUTPUT=$(cd "$OUTPUT" && pwd)
BUILD=$OUTPUT/build
mkdir -p "$BUILD"
cd "$BUILD"

"$SOURCE/configure" \
    --prefix="$OUTPUT" \
    --target-os=android --arch=aarch64 --cpu=armv8-a --enable-cross-compile \
    --cc="$TOOLCHAIN/bin/clang$EXE --target=aarch64-linux-android$API" \
    --cxx="$TOOLCHAIN/bin/clang++$EXE --target=aarch64-linux-android$API" \
    --ar="$TOOLCHAIN/bin/llvm-ar$EXE" --nm="$TOOLCHAIN/bin/llvm-nm$EXE" \
    --ranlib="$TOOLCHAIN/bin/llvm-ranlib$EXE" --strip="$TOOLCHAIN/bin/llvm-strip$EXE" \
    --sysroot="$TOOLCHAIN/sysroot" \
    --enable-shared --disable-static --enable-pic \
    --disable-programs --disable-doc --disable-avdevice --disable-avfilter \
    --disable-swscale --disable-swresample \
    --disable-vulkan \
    --enable-jni --enable-mediacodec \
    --extra-ldflags="-Wl,-z,max-page-size=16384"

make -j"$(nproc 2>/dev/null || echo 8)"
make install
echo "FFmpeg for Android installed in $OUTPUT (include/, lib/)"
