#!/usr/bin/env python3
"""Writes the motion_ball_* demo videos for trdrop-pp --motion: a ball moving at constant speed over a
static background, shown with different frame pacing. H.264 in mkv, like a recording.

    python make_ball_videos.py <output dir> [--ffmpeg ffmpeg] [--encoder h264_nvenc]

1280x720, 60 fps, 4 seconds. The animation time of a frame is the position of the ball: at time t (in
display intervals) it is at start + t * velocity.

  motion_ball_60       every frame advances by one interval                    60 fps, perceived 60
  motion_ball_30       a new frame every 2nd frame, advancing by 2             30 fps, perceived 30
  motion_ball_15       a new frame every 4th frame, advancing by 4             15 fps, perceived 15
  motion_ball_uneven   every frame is new, advancing by 0.5 / 1.5              60 fps, perceived 48
  motion_ball_stutter  a new frame every 2nd frame, advancing by 1 / 3         30 fps, perceived 24
  motion_ball_hitch    every second 6 frames freeze, the next one catches up   54 fps, perceived 35.3

Needs numpy, opencv-python and an ffmpeg with the chosen encoder.
"""

import argparse
import os
import subprocess

import cv2
import numpy as np

WIDTH, HEIGHT, FPS, FRAMES = 1280, 720, 60, 240
RADIUS = 120
START = (150.0, 200.0)
VELOCITY = (4.0, 1.2)  # pixels per display interval
BACKGROUND = (52, 44, 40)  # BGR


def animation_times():
    """name -> animation time (in display intervals) of every frame"""
    frames = np.arange(FRAMES)
    hitch = frames.astype(float)
    for second in range(FRAMES // 60):
        # frames 30 - 35 of every second show frame 29, frame 36 is on time again
        hitch[second * 60 + 30:second * 60 + 36] = second * 60 + 29
    return {
        "60": frames.astype(float),
        "30": (frames // 2 * 2).astype(float),
        "15": (frames // 4 * 4).astype(float),
        "uneven": np.where(frames % 2 == 0, frames, frames - 0.5).astype(float),
        # new frames at 0, 2, 4, ...: the animation is at 0, 1, 4, 5, 8, ...
        "stutter": (frames // 4 * 4 + (frames // 2 % 2)).astype(float),
        "hitch": hitch,
    }


def ball_sprite():
    """the ball (BGR, float) and its coverage (0 - 1): patterned, so its motion can be tracked"""
    size = 2 * RADIUS + 8
    center = (size // 2, size // 2)
    rng = np.random.default_rng(3)
    sprite = np.zeros((size, size, 3), np.float32)
    cv2.circle(sprite, center, RADIUS, (40, 150, 240), -1, cv2.LINE_AA)
    for _ in range(46):
        angle, distance = rng.uniform(0, 2 * np.pi), RADIUS * np.sqrt(rng.uniform(0, 1))
        position = (int(center[0] + distance * np.cos(angle)), int(center[1] + distance * np.sin(angle)))
        color = [(30, 60, 150), (120, 220, 255), (250, 250, 250), (20, 30, 60)][rng.integers(0, 4)]
        cv2.circle(sprite, position, int(rng.integers(7, 20)), color, -1, cv2.LINE_AA)
    # a light from the upper left
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    light = 1.15 - 0.45 * np.hypot(x - center[0] + 0.35 * RADIUS, y - center[1] + 0.35 * RADIUS) / (1.35 * RADIUS)
    sprite *= np.clip(light, 0.55, 1.15)[..., None]
    coverage = np.zeros((size, size), np.float32)
    cv2.circle(coverage, center, RADIUS, 1.0, -1, cv2.LINE_AA)
    cv2.circle(sprite, center, RADIUS, (20, 30, 60), 3, cv2.LINE_AA)
    sprite = cv2.GaussianBlur(np.clip(sprite, 0, 255), (5, 5), 1.0)
    coverage = cv2.GaussianBlur(coverage, (5, 5), 1.0)
    return sprite, coverage


def render(sprite, coverage, time):
    """the frame at animation time `time`, the ball placed with sub pixel accuracy"""
    x = START[0] + VELOCITY[0] * time - sprite.shape[1] / 2.0
    y = START[1] + VELOCITY[1] * time - sprite.shape[0] / 2.0
    matrix = np.float64([[1, 0, x], [0, 1, y]])
    ball = cv2.warpAffine(sprite, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_CUBIC)
    alpha = np.clip(cv2.warpAffine(coverage, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_CUBIC), 0, 1)[..., None]
    frame = np.float32(BACKGROUND) * (1 - alpha) + ball * alpha
    return np.clip(frame + 0.5, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Write the motion_ball_* demo videos.")
    parser.add_argument("output", help="output directory")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable (default: ffmpeg in the PATH)")
    parser.add_argument("--encoder", default="h264_nvenc", help="ffmpeg video encoder (default: h264_nvenc)")
    parser.add_argument("--bitrate", default="12M", help="video bitrate (default: 12M)")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    sprite, coverage = ball_sprite()
    for name, times in animation_times().items():
        path = os.path.join(args.output, f"motion_ball_{name}.mkv")
        encoder = subprocess.Popen([args.ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                    "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-c:v", args.encoder, "-b:v", args.bitrate,
                                    "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
        cache = {}
        for time in times:
            if time not in cache:
                cache = {time: render(sprite, coverage, time).tobytes()}  # duplicates are the identical picture
            encoder.stdin.write(cache[time])
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise SystemExit(f"error: {args.ffmpeg} failed for {path}")
        print(f"wrote {path} ({len(times)} frames)")


if __name__ == "__main__":
    main()
