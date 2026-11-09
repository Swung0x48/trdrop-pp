#!/usr/bin/env python3
"""Plots a histogram of the frametimes of every video from a trdrop-pp CSV.

Every new frame contributes one sample: the time the previous frame was shown
(vN_frametime_ms at the rows where vN_diff > 0).

    python frametime_histogram.py analysis.csv
    trdrop-pp capture.avi | python frametime_histogram.py --percent -o frametimes.png
"""

import argparse
import math

import trdrop_csv

# frametimes are multiples of 1 / recorded fps, so with few distinct values every value gets its own bar
MAX_DISCRETE_VALUES = 40


def frametime_samples(video):
    samples = []
    for diff, frametime in zip(video.diff, video.frametime_ms):
        if not math.isnan(diff) and diff > 0 and not math.isnan(frametime) and frametime > 0:
            samples.append(frametime)
    return samples


def percentile(sorted_values, p):
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * p / 100.0
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def main():
    parser = argparse.ArgumentParser(description="Plot a frametime histogram from a trdrop-pp CSV.")
    trdrop_csv.add_common_arguments(parser)
    parser.add_argument("--bins", type=int, default=None,
                        help="number of bins (default: one bar per distinct frametime if there are few, else 50)")
    parser.add_argument("--percent", action="store_true", help="show the share of frames instead of the count")
    parser.add_argument("--log", action="store_true", help="logarithmic y axis")
    args = parser.parse_args()

    analysis = trdrop_csv.read_analysis(args.csv, args.labels)
    trdrop_csv.setup_matplotlib(args)
    import matplotlib.pyplot as plt

    datasets = [frametime_samples(video) for video in analysis.videos]
    if not any(datasets):
        raise SystemExit("error: no frametimes in the CSV (needs at least two frame changes within one second)")

    # distinct values, rounded to avoid float noise such as 33.3333 vs 33.33330001
    distinct = sorted({round(value, 3) for samples in datasets for value in samples})
    discrete = args.bins is None and len(distinct) <= MAX_DISCRETE_VALUES
    if not discrete:
        # same bins for every video so the histograms are comparable
        bin_count = args.bins or 50
        # raw (not rounded) extremes, otherwise e.g. 16.6667 would fall below a 16.667 edge
        low = min(min(samples) for samples in datasets if samples)
        high = max(max(samples) for samples in datasets if samples)
        if high <= low:
            high = low + 1.0
        edges = [low + (high - low) * i / bin_count for i in range(bin_count + 1)]

    # one histogram per video, stacked vertically with a shared x axis
    count = len(analysis.videos)
    figure, axes = plt.subplots(count, 1, sharex=True, squeeze=False, figsize=(10, 1.5 + 2.8 * count))
    axes = [row[0] for row in axes]
    for index, (video, samples, axis) in enumerate(zip(analysis.videos, datasets, axes)):
        color = f"C{index}"
        if samples:
            ordered = sorted(samples)
            average = sum(samples) / len(samples)
            axis.set_title(f"{video.label}:  {len(samples)} frames, avg {average:.2f} ms, "
                           f"p99 {percentile(ordered, 99):.2f} ms, max {ordered[-1]:.2f} ms", fontsize=10, loc="left")
        else:
            axis.set_title(f"{video.label}:  no frametimes", fontsize=10, loc="left")
        scale = 100.0 / len(samples) if args.percent and samples else 1.0

        if discrete:
            # one bar per distinct frametime, labeled with its value
            gap = min((b - a for a, b in zip(distinct, distinct[1:])), default=max(distinct[0] * 0.5, 1.0))
            counts = {value: 0 for value in distinct}
            for value in samples:
                counts[round(value, 3)] += 1
            heights = [counts[value] * scale for value in distinct]
            bars = axis.bar(distinct, heights, width=gap * 0.7, color=color)
            axis.bar_label(bars, labels=[(f"{h:.1f}%" if args.percent else f"{h:g}") if h > 0 else "" for h in heights],
                           fontsize=8, padding=2)
        elif samples:
            axis.hist(samples, bins=edges, weights=[scale] * len(samples), color=color, rwidth=0.9)

        axis.set_ylabel("frames (%)" if args.percent else "frames")
        if args.log:
            axis.set_yscale("log")
        else:
            # headroom for the bar labels
            axis.set_ylim(0, max(axis.get_ylim()[1] * 1.15, 1.0))
        axis.grid(True, axis="y", alpha=0.3)

    if discrete:
        axes[-1].set_xticks(distinct)
        axes[-1].set_xticklabels([f"{value:g}" for value in distinct], rotation=45 if len(distinct) > 12 else 0)
    axes[-1].set_xlabel("frametime (ms)")
    figure.suptitle(args.title or "Frametime distribution")
    trdrop_csv.finish_plot(figure, args)


if __name__ == "__main__":
    main()
