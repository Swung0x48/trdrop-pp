#!/usr/bin/env python3
"""Writes a copy of a video with the motion vectors trdrop++ --motion works with drawn on top.

    python motion_vectors.py capture.mkv -o vectors.mp4
    python motion_vectors.py capture.mkv --csv capture.csv -o vectors.mp4

The vectors are measured here again, the same way as by the CPU path of trdrop-pp: a regular grid of 64x40
points at up to 1920x1080, pyramidal Lucas-Kanade forward and back, a point is dropped if it doesn't come back
within a pixel. They are not read from trdrop-pp, so single points can differ from what it (or its GPU path)
measured.

  arrow    a point that was tracked: from where it was in the previous frame to where it is now, colored by
           its length (blue: short, red: long). On a duplicate frame the arrows of the last new frame stay, dimmed.
  red x    a point whose match doesn't come back, it is not used
  nothing  no texture at the point

With --csv (written by trdrop-pp --motion for the same video) every frame also shows its row: new or duplicate,
animation step, animation error, perceived smoothness.

Needs numpy, opencv-python and ffmpeg (with ffprobe next to it).
"""

import argparse
import csv
import json
import math
import os
import subprocess

import cv2
import numpy as np

POINTS_X, POINTS_Y = 64, 40
MAX_WIDTH, MAX_HEIGHT = 1920, 1080
MAX_BACK_ERROR = 1.0
LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
          minEigThreshold=1.0 / 1024)


def probe(ffmpeg, path):
    """width, height, frames per second"""
    directory, name = os.path.split(ffmpeg)
    ffprobe = os.path.join(directory, name.replace("ffmpeg", "ffprobe"))
    output = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
                             "-of", "json", path], capture_output=True, check=True).stdout
    stream = json.loads(output)["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/")
    return int(stream["width"]), int(stream["height"]), float(numerator) / float(denominator)


def read_rows(path, video):
    """frame -> row of the trdrop-pp CSV, reduced to the columns of one video"""
    prefix = f"v{video}_"
    rows = {}
    with open(path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows[int(row["frame"])] = {key[len(prefix):]: value for key, value in row.items() if key.startswith(prefix)}
    return rows


def track(previous, current, points):
    """motion of the grid points and their state: 0 no texture, 1 doesn't come back, 2 used"""
    forward, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **LK)
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **LK)
    motion = (forward - points).reshape(-1, 2)
    miss = np.linalg.norm((back - points).reshape(-1, 2), axis=1)
    state = np.zeros(len(motion), np.uint8)
    state[status.ravel() == 1] = 1
    state[(status.ravel() == 1) & (back_status.ravel() == 1) & (miss <= MAX_BACK_ERROR)] = 2
    return motion, state


def draw(frame, points, motion, state, colors, scale, longest, dimmed):
    thickness = 2 if frame.shape[1] >= 1280 else 1
    for (x, y), (dx, dy), s in zip(points, motion, state):
        if s == 1:
            cv2.drawMarker(frame, (int(round(x)), int(round(y))), (40, 40, 230), cv2.MARKER_TILTED_CROSS, 5, 1, cv2.LINE_AA)
        elif s == 2:
            length = math.hypot(dx, dy)
            # the dark end of the color map is left out
            color = colors[min(255, 48 + int(207 * length / longest))]
            if dimmed:
                color = tuple(int(0.45 * c + 60) for c in color)
            start = (int(round(x)), int(round(y)))
            if length * scale < 1.5:
                cv2.circle(frame, start, thickness, color, -1, cv2.LINE_AA)
            else:
                end = (int(round(x + dx * scale)), int(round(y + dy * scale)))
                cv2.arrowedLine(frame, start, end, color, thickness, cv2.LINE_AA, tipLength=min(0.5, 6.0 / (length * scale)))


def label(frame, lines):
    height = 26
    width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0][0] for line in lines) + 20
    frame[8:8 + height * len(lines) + 8, 8:8 + width] = frame[8:8 + height * len(lines) + 8, 8:8 + width] // 4
    for k, line in enumerate(lines):
        cv2.putText(frame, line, (18, 8 + height * (k + 1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Write a copy of a video with its motion vectors drawn on top.")
    parser.add_argument("video")
    parser.add_argument("-o", "--output", required=True, help="output video (mp4, mkv, ...)")
    parser.add_argument("--csv", default=None, help="CSV written by trdrop-pp --motion for this video: shows the values of every frame")
    parser.add_argument("--video-index", type=int, default=1, help="column group of the CSV to show (default: 1)")
    parser.add_argument("--arrow-scale", type=float, default=1.0, help="draw the arrows this many times longer (default: 1, the real motion)")
    parser.add_argument("--longest", type=float, default=45.0, help="motion in pixels (of the measured resolution) that gets the last color (default: 45)")
    parser.add_argument("--dim", type=float, default=0.6, help="brightness of the picture under the arrows, 0 - 1 (default: 0.6)")
    parser.add_argument("--start", type=int, default=0, help="first frame (default: 0)")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after this many frames")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable (default: ffmpeg in the PATH)")
    parser.add_argument("--encoder", default="h264_nvenc", help="ffmpeg video encoder (default: h264_nvenc)")
    parser.add_argument("--bitrate", default="8M", help="video bitrate (default: 8M)")
    args = parser.parse_args()

    source_width, source_height, fps = probe(args.ffmpeg, args.video)
    # the same resolution trdrop-pp measures at
    divisor = max(1, math.ceil(max(source_width / MAX_WIDTH, source_height / MAX_HEIGHT) - 1e-9))
    width, height = source_width // divisor // 2 * 2, source_height // divisor // 2 * 2
    rows = read_rows(args.csv, args.video_index) if args.csv else None

    select = f"select=gte(n\\,{args.start})," if args.start else ""
    decoder = subprocess.Popen([args.ffmpeg, "-nostdin", "-v", "error", "-i", args.video, "-map", "0:v:0", "-fps_mode", "passthrough",
                                "-vf", f"{select}scale={width}:{height}:flags=area", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
                               stdout=subprocess.PIPE, bufsize=width * height * 3)
    encoder = subprocess.Popen([args.ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                                "-r", f"{fps:.6f}", "-i", "-", "-c:v", args.encoder, "-b:v", args.bitrate, "-pix_fmt", "yuv420p", args.output],
                               stdin=subprocess.PIPE)

    points = np.array([[(x + 0.5) * width / POINTS_X - 0.5, (y + 0.5) * height / POINTS_Y - 0.5]
                       for y in range(POINTS_Y) for x in range(POINTS_X)], np.float32).reshape(-1, 1, 2)
    flat_points = points.reshape(-1, 2)
    colors = [tuple(int(c) for c in color) for color in cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)[:, 0]]
    previous = None
    held = None  # the vectors of the last new frame
    index = args.start
    written = 0
    while args.max_frames is None or written < args.max_frames:
        data = decoder.stdout.read(width * height * 3)
        if len(data) < width * height * 3:
            break
        frame = np.frombuffer(data, np.uint8).reshape(height, width, 3)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        picture = (frame * args.dim).astype(np.uint8)
        lines = [f"frame {index}"]
        if previous is not None:
            motion, state = track(previous, gray, points)
            used = state == 2
            row = rows.get(index) if rows else None
            if row is not None and row.get("diff", "") != "":
                duplicate = float(row["diff"]) == 0.0
            else:
                duplicate = not used.any() or np.percentile(np.linalg.norm(motion[used], axis=1), 90) < 0.1
            if duplicate and held is not None:
                draw(picture, flat_points, held[0], held[1], colors, args.arrow_scale, args.longest, True)
            else:
                draw(picture, flat_points, motion, state, colors, args.arrow_scale, args.longest, False)
                held = (motion, state)
            lines[0] += "  duplicate" if duplicate else f"  new   {int(used.sum())} points used, {int((state == 1).sum())} rejected"
            if args.arrow_scale != 1.0:
                lines[0] += f"   (arrows x{args.arrow_scale:g})"
            if row is not None:
                if row.get("anim_step", ""):
                    lines.append(f"animation step {float(row['anim_step']):.2f} intervals (votes spread {float(row['anim_step_spread']):.2f})")
                else:
                    lines.append("animation step: not measurable")
                if row.get("anim_error_ms", ""):
                    lines.append(f"animation error {float(row['anim_error_ms']):+.1f} ms")
                rate = f"framerate {float(row['fps']):.1f} fps" if row.get("fps", "") else ""
                if row.get("perceived_fps", ""):
                    rate += f"   perceived smoothness {float(row['perceived_fps']):.1f} fps"
                if rate:
                    lines.append(rate)
        label(picture, lines)
        encoder.stdin.write(picture.tobytes())
        previous = gray
        index += 1
        written += 1
        if written % 300 == 0:
            print(f"{written} frames", flush=True)
    decoder.kill()
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit(f"error: {args.ffmpeg} failed to write {args.output}")
    print(f"wrote {args.output} ({written} frames, {width}x{height})")


if __name__ == "__main__":
    main()
