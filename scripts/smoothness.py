#!/usr/bin/env python3
"""Plots the motion analysis of a trdrop-pp CSV written with --motion.

Three charts, all videos overlaid:
  framerate            pixel based framerate (vN_fps) and perceived smoothness (vN_perceived_fps)
  animation error      per new frame over time (vN_anim_error_ms)
  animation error      histogram

    trdrop-pp capture.mkv --motion -o capture.csv
    python smoothness.py capture.csv --fps 60 -o smoothness.png
"""

import argparse
import math

import trdrop_csv


def percentile(sorted_values, p):
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * p / 100.0
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def main():
    parser = argparse.ArgumentParser(description="Plot the perceived smoothness and animation error from a trdrop-pp --motion CSV.")
    trdrop_csv.add_common_arguments(parser)
    parser.add_argument("--fps", type=trdrop_csv.positive_float, default=None,
                        help="recorded framerate of the capture, shows the x axis in seconds instead of frames")
    parser.add_argument("--max-error", type=trdrop_csv.positive_float, default=None,
                        help="range of the animation error axis in ms (default: the 99th percentile, at least 25)")
    args = parser.parse_args()

    analysis = trdrop_csv.read_analysis(args.csv, args.labels)
    if not any(video.anim_step for video in analysis.videos):
        raise SystemExit("error: the CSV has no motion columns, run trdrop-pp with --motion")
    trdrop_csv.setup_matplotlib(args)
    import matplotlib.pyplot as plt

    x = [frame / args.fps for frame in analysis.frames] if args.fps else analysis.frames
    skip = math.ceil(args.fps) if args.fps else 0
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    figure, (rate_axis, error_axis, histogram_axis) = plt.subplots(3, 1, figsize=(12, 11))

    all_errors = [e for video in analysis.videos for e in video.anim_error_ms if not math.isnan(e)]
    limit = args.max_error or max(25.0, percentile(sorted(abs(e) for e in all_errors), 99) if all_errors else 25.0)
    highest_rate = 1.0

    for number, video in enumerate(analysis.videos):
        color = colors[number % len(colors)]
        perceived = [v for v in video.perceived_fps[skip:] if not math.isnan(v)]
        measured = [v for v in video.fps[skip:] if not math.isnan(v)]
        highest_rate = max([highest_rate] + measured + perceived)
        rate_axis.plot(x[skip:], video.fps[skip:], color=color, linewidth=1.0, alpha=0.45,
                       label=f"{video.label}: framerate (avg {sum(measured) / max(1, len(measured)):.1f} fps)")
        rate_axis.plot(x[skip:], video.perceived_fps[skip:], color=color, linewidth=1.5,
                       label=f"{video.label}: perceived smoothness"
                             + (f" (avg {sum(perceived) / len(perceived):.1f} fps)" if perceived else " (not measurable)"))

        points = [(position, error) for position, error in zip(x, video.anim_error_ms) if not math.isnan(error)]
        errors = [error for _, error in points]
        if errors:
            absolute = sorted(abs(e) for e in errors)
            label = (f"{video.label}: {len(errors)} new frames, avg |e| {sum(absolute) / len(absolute):.1f} ms, "
                     f"p95 {percentile(absolute, 95):.1f} ms, max {absolute[-1]:.0f} ms")
            error_axis.vlines([p for p, _ in points], 0, [max(-limit, min(limit, e)) for e in errors], color=color, linewidth=0.5, alpha=0.6)
            error_axis.plot([p for p, _ in points], [max(-limit, min(limit, e)) for e in errors], ".", markersize=3, color=color, label=label)
            bin_width = limit / 30.0
            bins = [-limit + k * bin_width for k in range(61)]
            histogram_axis.hist([max(-limit, min(limit - 1e-9, e)) for e in errors], bins=bins, color=color,
                                alpha=0.6 if len(analysis.videos) > 1 else 1.0, label=video.label)

    rate_axis.set_ylim(0, highest_rate * 1.1)
    rate_axis.set_xlim(x[skip] if skip < len(x) else x[0], x[-1])
    rate_axis.set_ylabel("fps")
    rate_axis.grid(alpha=0.3)
    rate_axis.legend(fontsize=9)
    rate_axis.set_title(args.title or "perceived smoothness and animation error")

    error_axis.axhline(0, color="0.5", linewidth=0.8)
    error_axis.set_xlim(x[0], x[-1])
    error_axis.set_ylim(-limit * 1.05, limit * 1.05)
    error_axis.set_xlabel("time (s)" if args.fps else "frame")
    error_axis.set_ylabel("animation error (ms)")
    error_axis.grid(alpha=0.3)
    if all_errors:
        error_axis.legend(fontsize=9)

    histogram_axis.set_xlim(-limit, limit)
    histogram_axis.set_xlabel("animation error (ms): animation step of a new frame - time the previous frame was shown"
                              " (values beyond the range are in the outer bars)")
    histogram_axis.set_ylabel("new frames")
    histogram_axis.grid(alpha=0.3, axis="y")
    if len(analysis.videos) > 1 and all_errors:
        histogram_axis.legend(fontsize=9)

    trdrop_csv.finish_plot(figure, args)


if __name__ == "__main__":
    main()
