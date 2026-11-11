#!/usr/bin/env python3
"""Writes the motion_ball_* demo videos for trdrop-pp --motion: a ball moving at constant speed over a
static background, shown with different frame pacing. H.264 in mkv, like a recording.

    python make_ball_videos.py <output dir> [--ffmpeg ffmpeg] [--encoder h264_nvenc]

1280x720, 60 fps, 4 seconds. The animation time of a frame is the position of the ball: at time t (in
display intervals) it is at start + t * velocity.

Constant pacing, the framerate and the perceived smoothness are flat lines:

  motion_ball_60       every frame advances by one interval                    60 fps, perceived 60
  motion_ball_30       a new frame every 2nd frame, advancing by 2             30 fps, perceived 30
  motion_ball_15       a new frame every 4th frame, advancing by 4             15 fps, perceived 15
  motion_ball_uneven   every frame is new, advancing by 0.5 / 1.5              60 fps, perceived 48
  motion_ball_stutter  a new frame every 2nd frame, advancing by 1 / 3         30 fps, perceived 24
  motion_ball_hitch    every second 6 frames freeze, the next one catches up   54 fps, perceived 35.3

The pacing changes every 2 seconds (8 seconds, the ball moves half as fast), the values follow within a second:

  motion_ball_change   like motion_ball_60, then _uneven, then _stutter,       60 / 60 / 30 / 60 fps,
                       then _60 again                                          perceived 60 / 48 / 24 / 60

An even 60 fps recorded by a capture that is out of step with it (see --async-capture): 10 times per second
one picture is recorded twice and another one is lost

  motion_ball_capture  as recorded                                             50 fps, perceived 45
                       with --async-capture                                    60 fps, perceived 60

For the framerate alone (too small for --motion): a ball a fifth of the size, which changes 0.2% of the picture.
That is less than the 0.5% a frame of a lossy recording has to change, unless its duplicates show that less is
enough (the automatic tolerance, see "Lossy recordings" in the README). A keyframe every 30 frames, on duplicates

  motion_ball_small    a new frame every 3rd frame, advancing by 3             20 fps after the first half second,
                                                                               in which the duplicates are measured

Sparse motion: the large ball stands still in the middle, ten balls of a radius of 8 pixels move past it. The
regular grid of points catches too few of them (the animation of a fifth of the new frames is measured, and
wrongly); the points added where a picture with sparse motion has texture measure all of them

  motion_ball_sparse   like motion_ball_stutter                                30 fps, perceived 24 after the first
                                                                               second (in which the duplicates are measured)

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


def videos():
    """name -> (animation time of every frame in display intervals, speed of the ball relative to VELOCITY[,
    size of the ball relative to RADIUS, frames whose number modulo 30 are this are keyframes (None: as the
    encoder likes)[, columns and rows of balls that move, a ball of full size stands still in the middle]])"""
    frames = np.arange(FRAMES)
    hitch = frames.astype(float)
    for second in range(FRAMES // 60):
        # frames 30 - 35 of every second show frame 29, frame 36 is on time again
        hitch[second * 60 + 30:second * 60 + 36] = second * 60 + 29
    # how far every frame advances: 2 seconds each of even steps, 0.5 / 1.5, a new frame every 2nd frame
    # advancing by 1 / 3, even steps
    steps = np.concatenate([np.ones(120), np.tile([0.5, 1.5], 60), np.tile([0.0, 1.0, 0.0, 3.0], 30), np.ones(120)])
    steps[0] = 0.0
    # capture out of step: in every 6 frames one picture is recorded an interval late (a repeat) and stays late
    # for `lag` frames, then one picture is lost. The lag changes every half second, 1: lost right after the repeat
    lag = np.array([3, 1, 2, 5, 4, 1, 3, 2])[np.minimum(frames // 30, 7)]
    capture = np.maximum(np.where(frames % 6 < lag, frames - 1, frames), 0)
    return {
        "capture": (capture.astype(float), 1.0),
        "60": (frames.astype(float), 1.0),
        "30": ((frames // 2 * 2).astype(float), 1.0),
        "15": ((frames // 4 * 4).astype(float), 1.0),
        "uneven": (np.where(frames % 2 == 0, frames, frames - 0.5).astype(float), 1.0),
        # new frames at 0, 2, 4, ...: the animation is at 0, 1, 4, 5, 8, ...
        "stutter": ((frames // 4 * 4 + (frames // 2 % 2)).astype(float), 1.0),
        "hitch": (hitch, 1.0),
        "change": (np.cumsum(steps), 0.5),
        # frames 10, 40, 70, ... are duplicates
        "small": ((frames // 3 * 3).astype(float), 1.0, 0.2, 10),
        "sparse": ((frames // 4 * 4 + (frames // 2 % 2)).astype(float), 0.5, 0.07, None, (5, 2)),
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


def place(frame, sprite, coverage, x, y):
    """the frame (float) with the ball centered at (x, y), placed with sub pixel accuracy"""
    matrix = np.float64([[1, 0, x - sprite.shape[1] / 2.0], [0, 1, y - sprite.shape[0] / 2.0]])
    ball = cv2.warpAffine(sprite, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_CUBIC)
    alpha = np.clip(cv2.warpAffine(coverage, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_CUBIC), 0, 1)[..., None]
    return frame * (1 - alpha) + ball * alpha


def render(sprite, coverage, time, speed, starts=(START,), still=None):
    """the frame at animation time `time`: a ball from every start, and the balls `still` (sprite, coverage, x, y)
    in front of them"""
    frame = np.empty((HEIGHT, WIDTH, 3), np.float32)
    frame[:] = np.float32(BACKGROUND)
    for x, y in starts:
        frame = place(frame, sprite, coverage, x + VELOCITY[0] * speed * time, y + VELOCITY[1] * speed * time)
    for ball in still or ():
        frame = place(frame, *ball)
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
    for name, (times, speed, *more) in videos().items():
        path = os.path.join(args.output, f"motion_ball_{name}.mkv")
        ball, ball_coverage, keyframes, starts, still = sprite, coverage, [], (START,), None
        if more:
            size, keyframe, *swarm = more
            ball = cv2.resize(sprite, None, fx=size, fy=size, interpolation=cv2.INTER_AREA)
            ball_coverage = cv2.resize(coverage, None, fx=size, fy=size, interpolation=cv2.INTER_AREA)
            if keyframe is not None: keyframes = ["-g", "600", "-force_key_frames", f"expr:eq(mod(n,30),{keyframe})"]
            if swarm:
                columns, rows = swarm[0]
                # spread over the left of the picture, every second column a little lower; they stay in the picture
                starts = [(60.0 + column * 700.0 / max(1, columns - 1), 50.0 + row * 480.0 / max(1, rows - 1) + column % 2 * 30.0)
                          for row in range(rows) for column in range(columns)]
                still = [(sprite, coverage, WIDTH / 2.0, HEIGHT / 2.0)]
        encoder = subprocess.Popen([args.ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                    "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-c:v", args.encoder, "-b:v", args.bitrate,
                                    *keyframes, "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
        cache = {}
        for time in times:
            if time not in cache:
                cache = {time: render(ball, ball_coverage, time, speed, starts, still).tobytes()}  # duplicates are the identical picture
            encoder.stdin.write(cache[time])
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise SystemExit(f"error: {args.ffmpeg} failed for {path}")
        print(f"wrote {path} ({len(times)} frames)")


if __name__ == "__main__":
    main()
