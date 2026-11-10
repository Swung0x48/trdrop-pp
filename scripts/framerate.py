#!/usr/bin/env python3
"""Plots the framerate of every video over time from a trdrop++ CSV.

    python framerate.py analysis.csv
    trdrop-pp capture.avi | python framerate.py --fps 60 -o framerate.png
"""

import argparse
import math

import trdrop_csv


def main():
    parser = argparse.ArgumentParser(description="Plot the framerate over time from a trdrop++ CSV.")
    trdrop_csv.add_common_arguments(parser)
    parser.add_argument("--fps", type=trdrop_csv.positive_float, default=None,
                        help="recorded framerate of the capture, shows the x axis in seconds instead of frames")
    parser.add_argument("--skip", type=int, default=None,
                        help="hide the first N frames (default: the first second if --fps is given, "
                             "as the framerate window is still filling up)")
    parser.add_argument("--max-fps", type=float, default=None, help="upper limit of the y axis")
    args = parser.parse_args()

    analysis = trdrop_csv.read_analysis(args.csv, args.labels)
    trdrop_csv.setup_matplotlib(args)
    import matplotlib.pyplot as plt

    skip = args.skip
    if skip is None:
        skip = math.ceil(args.fps) if args.fps else 0
    start = next((i for i, frame in enumerate(analysis.frames) if frame >= skip), len(analysis.frames))
    if start >= len(analysis.frames):
        raise SystemExit(f"error: nothing left to plot after skipping {skip} frames")

    frames = analysis.frames[start:]
    x = [frame / args.fps for frame in frames] if args.fps else frames

    figure, axis = plt.subplots(figsize=(12, 5))
    highest = 0.0
    for video in analysis.videos:
        values = video.fps[start:]
        valid = [v for v in values if not math.isnan(v)]
        average = sum(valid) / len(valid) if valid else 0.0
        low = min(valid) if valid else 0.0
        highest = max(highest, max(valid) if valid else 0.0)
        axis.plot(x, values, linewidth=1.5,
                  label=f"{video.label}  (avg {average:.2f}, min {low:.2f} fps)")

    axis.set_xlabel("time (s)" if args.fps else "frame")
    axis.set_ylabel("framerate (fps)")
    axis.set_xlim(x[0], x[-1])
    axis.set_ylim(0, args.max_fps if args.max_fps else max(1.0, highest * 1.1))
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right")
    axis.set_title(args.title or "Framerate")
    trdrop_csv.finish_plot(figure, args)


if __name__ == "__main__":
    main()
