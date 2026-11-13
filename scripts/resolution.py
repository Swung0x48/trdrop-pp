#!/usr/bin/env python3
"""Plots what the render resolution estimate of trdrop++ is made of, to judge a recording by eye.

    trdrop-pp capture.mkv --resolution-data capture.res.csv -o capture.csv
    python resolution.py capture.res.csv
    python resolution.py capture.res.csv --around 0.42 -o grid.png

Top two charts: the pixel grid. A picture rendered at r times the output resolution has a narrow peak at r or
1 - r cycles per pixel, in the columns and in the rows, at the same place for the whole video. The strongest peaks
off the block grid of the video encoder (dashed lines: multiples of 1/8, dotted: of 1/16) are labelled with both scales and with
the share of the video that has them.

Bottom left: the picture entering at the edge of a pan against the picture leaving at the other edge, by how
long ago it entered. Below 1: the entering picture lacks that detail (DLSS at 33% fell to 0.3 around 0.25, and
crossed 0.5 at half the render sampling rate). Above 1: it has more of it than the settled picture.
Bottom middle: the two spectra the newest and the oldest of those curves are the ratio of.
Bottom right: the stair steps of nearly level edges, by their height, in the entering and in the leaving picture.
An upscaler that shows new pixels as they were rendered has steps of one rendered pixel there: mostly 3 pixels at
a render scale of 33%, 2 at 50%. Steps of 1 pixel say nothing, a smooth edge has them.
"""

import argparse
import csv
import sys

import trdrop_csv

CODEC_GRID = 32.0
GRID_COLOR = "#2f6fb0"
REFERENCE_COLOR = "#8a8a8a"
INK = "#333333"
MIN_PAN_FRAMES = 30


def read_data(path):
    """-> (dict of the '# key: value' lines, dict curve -> {(direction, age): [(frequency, value)]})"""
    meta, curves = {}, {}
    with open(path, newline="", encoding="utf-8") as stream:
        rows = []
        for line in stream:
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                meta[key.strip()] = value.strip()
            elif line.strip():
                rows.append(line)
    for row in csv.DictReader(rows):
        try:
            point = (float(row["frequency"]), float(row["value"]))
        except (KeyError, ValueError):
            raise SystemExit(f"error: {path} is not a file written by trdrop-pp --resolution-data")
        curves.setdefault(row["curve"], {}).setdefault((row["direction"], row["age"]), []).append(point)
    if not curves:
        raise SystemExit(f"error: {path} holds no data (was it written by trdrop-pp --resolution-data?)")
    return meta, curves


def off_block_grid(frequency, gap):
    on_grid = frequency * CODEC_GRID
    return abs(on_grid - round(on_grid)) / CODEC_GRID > gap


def find_peaks(points, count, minimum, apart=0.006, gap=0.0012):
    """the strongest local maxima off the encoder's block grid, between 0.06 and just below 0.5"""
    candidates = []
    for i in range(1, len(points) - 1):
        frequency, value = points[i]
        if frequency < 0.06 or frequency > 0.497 or value < minimum or not off_block_grid(frequency, gap):
            continue
        if value >= points[i - 1][1] and value >= points[i + 1][1]:
            candidates.append((value, frequency))
    peaks = []
    for value, frequency in sorted(candidates, reverse=True):
        if all(abs(frequency - other) >= apart for _, other in peaks):
            peaks.append((value, frequency))
        if len(peaks) == count:
            break
    return peaks


def value_near(points, frequency, tolerance=0.0012):
    values = [value for f, value in points if abs(f - frequency) <= tolerance]
    return max(values) if values else float("nan")


def plot_grid(axis, name, strength, share, args, meta):
    frequencies = [f for f, _ in strength]
    values = [v for _, v in strength]
    low, high = (args.around - args.span, args.around + args.span) if args.around else (0.06, 0.5)
    shown = [v for f, v in strength if low <= f <= high]
    top = max(4.0, 1.25 * max(shown)) if shown else 4.0

    for k in range(1, 16, 2):                               # the encoder's other frequencies: odd multiples of 1/16
        if low <= k / 16.0 <= high:
            axis.axvline(k / 16.0, color=REFERENCE_COLOR, linewidth=0.5, linestyle=(0, (1, 5)), zorder=1)
    for k in range(1, 5):                                   # the encoder's strongest: multiples of 1/8
        frequency = k / 8.0
        if low <= frequency <= high:
            axis.axvline(frequency, color=REFERENCE_COLOR, linewidth=0.8, linestyle=(0, (4, 4)), zorder=1)
            axis.annotate(f"{k}/8" if k % 2 else ("1/4" if k == 2 else "1/2"), (frequency, top), xytext=(3, -3),
                          textcoords="offset points", ha="left", va="top", fontsize=8, color=REFERENCE_COLOR)
    axis.axhline(3.0, color=REFERENCE_COLOR, linewidth=0.8, zorder=1)
    axis.plot(frequencies, values, color=GRID_COLOR, linewidth=0.9, zorder=2)

    width, height = float(meta.get("width", 0) or 0), float(meta.get("height", 0) or 0)
    inside = [(f, v) for f, v in strength if low <= f <= high]
    lossless = meta.get("lossless") == "1"                  # --lossless: no encoder, its grid counts like the rest
    for value, frequency in find_peaks(inside, args.peaks, args.min_strength, gap=-1.0 if lossless else 0.0012):
        label = f"{frequency:.4f}: {value:.1f}x\n{100 * frequency:.1f}% or {100 * (1 - frequency):.1f}%"
        if width and height:
            label += "\n" + " / ".join(f"{2 * round(s * width / 2)}x{2 * round(s * height / 2)}" for s in (frequency, 1 - frequency))
        if share:
            label += f"\nin {100 * value_near(share, frequency, 0.0006):.0f}% of the video"
        axis.plot([frequency], [value], marker="o", markersize=4, color=GRID_COLOR, zorder=3)
        axis.annotate(label, (frequency, value), xytext=(6, 4), textcoords="offset points", ha="left", va="bottom",
                      fontsize=8, color=INK, zorder=4)
    half, quarter = value_near(strength, 0.5), value_near(strength, 0.25)
    encoder = " (recorded lossless)" if lossless else " (encoder)"
    axis.annotate(f"at 0.5: {half:.1f}x   at 0.25{encoder}: {quarter:.1f}x   |   grey line: 3x, a peak both axes have counts from there",
                  (0.01, 0.95), xycoords="axes fraction", ha="left", va="top", fontsize=9, color=INK)
    axis.set_xlim(low, high + (0.004 if not args.around else 0.0))      # room for the peak at 0.5
    axis.set_ylim(0, top * 1.35)
    axis.set_ylabel(f"pixel grid, {name}\n(times the background)")
    axis.grid(True, axis="y", alpha=0.25)


def plot_pan(contrast_axis, spectrum_axis, curves, args, meta):
    import matplotlib.pyplot as plt

    direction = args.direction
    contrast = {age: points for (d, age), points in curves.get("pan_contrast", {}).items() if d == direction}
    counts = {"right": int(meta.get("pan_frames_right", 0) or 0), "left": int(meta.get("pan_frames_left", 0) or 0)}
    frames = counts["right"] + counts["left"] if direction == "both" else counts[direction]
    if not contrast or frames < MIN_PAN_FRAMES:
        # the program needs 30 frames of a pan as well: fewer are noise
        which = "" if direction == "both" else f" with the picture moving {direction}"
        note = f"no pan was measured{which}" if not contrast else f"only {frames} frame{'' if frames == 1 else 's'} of a pan{which}, {MIN_PAN_FRAMES} are needed"
        for axis in (contrast_axis, spectrum_axis):
            axis.text(0.5, 0.5, note, ha="center", va="center", transform=axis.transAxes, color=INK)
            axis.set_axis_off()
        return
    ages = sorted(contrast, key=int)
    shades = plt.cm.Blues
    for index, age in enumerate(ages):                       # the newest columns darkest
        color = shades(0.95 - 0.6 * index / max(1, len(ages) - 1))
        points = contrast[age]
        contrast_axis.plot([f for f, _ in points], [v for _, v in points], color=color, linewidth=2.0 if index == 0 else 1.3,
                           label=meta.get(f"age_{age}", f"age {age}"))
    for level, text in ((1.0, "as much as the leaving picture"), (0.5, "one half")):
        contrast_axis.axhline(level, color=REFERENCE_COLOR, linewidth=0.8, zorder=1)
        contrast_axis.annotate(text, (0.469, level), xytext=(-3, 2), textcoords="offset points", ha="right", va="bottom",
                               fontsize=8, color=REFERENCE_COLOR)
    highest = max(v for points in contrast.values() for _, v in points)
    contrast_axis.set_ylim(0, max(1.3, highest * 1.1))
    contrast_axis.set_xlim(0.047, 0.469)
    contrast_axis.set_xlabel("vertical detail (cycles per pixel)")
    contrast_axis.set_ylabel("contrast, entering / leaving edge")
    contrast_axis.grid(True, alpha=0.25)
    contrast_axis.legend(loc="best", fontsize=8, title="entered", title_fontsize=8)
    frames = f"{meta.get('pan_frames_right', '?')} + {meta.get('pan_frames_left', '?')} frames (moving right + left)"
    contrast_axis.set_title(f"picture entering in a pan, {frames}" + ("" if direction == "both" else f", shown: {direction}"), fontsize=10)
    scale = contrast_axis.secondary_xaxis("top", functions=(lambda f: 200.0 * f, lambda s: s / 200.0))
    scale.set_xlabel("render scale whose single frame ends here (%)", fontsize=8)
    scale.tick_params(labelsize=8)

    entering = {age: points for (d, age), points in curves.get("pan_entering", {}).items() if d == direction}
    leaving = {age: points for (d, age), points in curves.get("pan_leaving", {}).items() if d == direction}
    newest, oldest = ages[0], ages[-1]
    spectrum_axis.plot([f for f, _ in leaving[newest]], [v for _, v in leaving[newest]], color=REFERENCE_COLOR, linewidth=1.6,
                       label="leaving edge (same columns as the newest)")
    spectrum_axis.plot([f for f, _ in entering[newest]], [v for _, v in entering[newest]], color=shades(0.95), linewidth=2.0,
                       label="entering: " + meta.get(f"age_{newest}", f"age {newest}"))
    if oldest != newest:
        spectrum_axis.plot([f for f, _ in entering[oldest]], [v for _, v in entering[oldest]], color=shades(0.45), linewidth=1.3,
                           label="entering: " + meta.get(f"age_{oldest}", f"age {oldest}"))
    spectrum_axis.set_yscale("log")
    spectrum_axis.set_xlim(0.047, 0.469)
    spectrum_axis.set_xlabel("vertical detail (cycles per pixel)")
    spectrum_axis.set_ylabel("mean power of a column")
    spectrum_axis.grid(True, which="both", alpha=0.2)
    spectrum_axis.legend(loc="best", fontsize=8)
    spectrum_axis.set_title("the spectra behind the contrast", fontsize=10)


def title_font(text):
    """the title is a file name: a font that has its characters, if the default one may lack them"""
    from matplotlib import font_manager

    if text.isascii():
        return None
    for family in ("Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Malgun Gothic", "Yu Gothic"):
        try:
            return font_manager.FontProperties(fname=font_manager.findfont(family, fallback_to_default=False), size=11)
        except ValueError:
            continue
    return None


def plot_stairs(axis, curves, meta):
    entering = dict(curves.get("stairs_entering", {}).get(("both", ""), []))
    leaving = dict(curves.get("stairs_leaving", {}).get(("both", ""), []))
    totals = (sum(entering.values()), sum(leaving.values()))
    if not entering or totals[0] + totals[1] == 0:
        axis.text(0.5, 0.5, "no stair steps were found", ha="center", va="center", transform=axis.transAxes, color=INK)
        axis.set_axis_off()
        return
    heights = sorted(entering)
    last = heights[-1]
    width = 0.38
    for offset, counts, total, color, name in ((-width / 2, entering, totals[0], "#08306b", "entering"), (width / 2, leaving, totals[1], REFERENCE_COLOR, "leaving")):
        shares = [100.0 * counts.get(h, 0.0) / total if total else 0.0 for h in heights]
        bars = axis.bar([h + offset for h in heights], shares, width=width - 0.04, color=color, label=f"{name} picture ({int(total)} steps)")
        for bar, share in zip(bars, shares):
            if share >= 5:
                axis.annotate(f"{share:.0f}", (bar.get_x() + bar.get_width() / 2, share), xytext=(0, 2), textcoords="offset points",
                              ha="center", va="bottom", fontsize=8, color=INK)
    axis.set_xticks(heights)
    axis.set_xticklabels([f"{int(h)}+" if h == last else f"{int(h)}" for h in heights])
    axis.set_xlabel("height of a step (output pixels); 1 / height = render scale")
    axis.set_ylabel("share of the steps (%)")
    axis.set_ylim(0, 108)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="upper right", fontsize=8)
    axis.set_title("stair steps of nearly level edges", fontsize=10)


def main():
    parser = argparse.ArgumentParser(description="Plot the data behind the render resolution estimate (trdrop-pp --resolution-data).")
    parser.add_argument("csv", help="file written by trdrop-pp --resolution-data")
    parser.add_argument("--around", type=float, default=None, metavar="FREQ",
                        help="show the pixel grid only around this frequency (cycles per pixel), e.g. 0.42")
    parser.add_argument("--span", type=trdrop_csv.positive_float, default=0.02,
                        help="with --around: this far to both sides (default: 0.02)")
    parser.add_argument("--peaks", type=int, default=4, help="peaks labelled per axis (default: 4)")
    parser.add_argument("--min-strength", type=float, default=1.8,
                        help="label peaks that are at least this many times the background (default: 1.8)")
    parser.add_argument("--direction", choices=("both", "right", "left"), default="both",
                        help="pan: frames with the picture moving this way (default: both)")
    parser.add_argument("--title", default=None, help="plot title")
    parser.add_argument("-o", "--output", default=None,
                        help="save the plot to this file (png, svg, pdf, ...) instead of showing a window")
    parser.add_argument("--dpi", type=int, default=150, help="resolution of the saved image (default: 150)")
    args = parser.parse_args()

    meta, curves = read_data(args.csv)
    trdrop_csv.setup_matplotlib(args)
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 11.5), constrained_layout=True)
    layout = figure.add_gridspec(3, 3, height_ratios=(1.0, 1.0, 1.25), width_ratios=(1.15, 1.0, 0.85))
    columns_axis = figure.add_subplot(layout[0, :])
    rows_axis = figure.add_subplot(layout[1, :], sharex=columns_axis)
    contrast_axis = figure.add_subplot(layout[2, 0])
    spectrum_axis = figure.add_subplot(layout[2, 1])
    stairs_axis = figure.add_subplot(layout[2, 2])

    for axis, name in ((columns_axis, "columns"), (rows_axis, "rows")):
        strength = curves.get(f"grid_{name}_strength", {}).get(("", ""), [])
        share = curves.get(f"grid_{name}_share", {}).get(("", ""), [])
        if strength:
            plot_grid(axis, name, strength, share, args, meta)
        else:
            axis.text(0.5, 0.5, "no pixel grid data", ha="center", va="center", transform=axis.transAxes, color=INK)
    columns_axis.tick_params(labelbottom=False)
    rows_axis.set_xlabel("cycles per pixel  (a render scale of r puts its peak at r or 1 - r)")
    plot_pan(contrast_axis, spectrum_axis, curves, args, meta)
    plot_stairs(stairs_axis, curves, meta)

    size = f"{meta.get('width', '?')}x{meta.get('height', '?')}"
    title = args.title or f"{meta.get('video', args.csv)}  ({size}, {meta.get('frames', '?')} frames)"
    figure.suptitle(title, fontsize=11, fontproperties=title_font(title))
    if args.output:
        figure.savefig(args.output, dpi=args.dpi)
        print(f"saved {args.output}", file=sys.stderr)
    else:
        plt.show()


if __name__ == "__main__":
    main()
