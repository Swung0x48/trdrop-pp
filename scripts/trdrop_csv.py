"""Shared helpers for reading the per frame CSV written by trdrop-pp."""

import argparse
import csv
import io
import math
import re
import sys
from dataclasses import dataclass, field

_COLUMN = re.compile(r"^v(\d+)_(diff|fps|frametime_ms|tears|anim_step|anim_step_spread|anim_error_ms|perceived_fps)$")


@dataclass
class Video:
    """Per frame data of one input video (column group vN_ of the CSV)."""

    index: int  # 1 based, as in the CSV
    label: str
    diff: list = field(default_factory=list)  # NaN for frame 0
    fps: list = field(default_factory=list)
    frametime_ms: list = field(default_factory=list)
    tears: list = field(default_factory=list)
    # only with trdrop-pp --motion, NaN where not available
    anim_step: list = field(default_factory=list)
    anim_step_spread: list = field(default_factory=list)
    anim_error_ms: list = field(default_factory=list)
    perceived_fps: list = field(default_factory=list)


@dataclass
class Analysis:
    frames: list
    videos: list


def _number(text):
    text = text.strip()
    return float(text) if text else math.nan


def read_analysis(path, labels=None):
    """Reads a trdrop-pp CSV from a file path, or from stdin if path is None or '-'."""
    if path in (None, "-"):
        if sys.stdin is None or sys.stdin.isatty():
            raise SystemExit("error: no CSV given - pass a file or pipe trdrop-pp into this script, "
                             "e.g. trdrop-pp video.avi | python " + sys.argv[0])
        stream = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", newline="")
        return _parse(stream, labels, "<stdin>")
    try:
        with open(path, newline="", encoding="utf-8") as stream:
            return _parse(stream, labels, path)
    except OSError as error:
        raise SystemExit(f"error: can't read {path}: {error.strerror}")


def _parse(stream, labels, name):
    reader = csv.reader(stream)
    try:
        header = [column.strip() for column in next(reader)]
    except StopIteration:
        raise SystemExit(f"error: {name} is empty")
    if not header or header[0] != "frame":
        raise SystemExit(f"error: {name} doesn't look like a trdrop-pp CSV (first column must be 'frame')")

    # column index -> (video number, metric)
    columns = {}
    for position, column in enumerate(header):
        match = _COLUMN.match(column)
        if match:
            columns[position] = (int(match.group(1)), match.group(2))
    numbers = sorted({video for video, _ in columns.values()})
    if not numbers:
        raise SystemExit(f"error: {name} contains no vN_ columns")

    labels = list(labels or [])
    videos = {}
    for i, number in enumerate(numbers):
        label = labels[i] if i < len(labels) and labels[i] else f"v{number}"
        videos[number] = Video(number, label)

    frames = []
    for row in reader:
        if not row:
            continue
        frames.append(int(row[0]))
        for position, (number, metric) in columns.items():
            value = _number(row[position]) if position < len(row) else math.nan
            getattr(videos[number], metric).append(value)
    if not frames:
        raise SystemExit(f"error: {name} contains no frames")
    return Analysis(frames, [videos[number] for number in numbers])


def add_common_arguments(parser):
    parser.add_argument("csv", nargs="?", default="-",
                        help="CSV written by trdrop-pp (default: read from stdin)")
    parser.add_argument("--labels", type=lambda text: text.split(","), default=None,
                        help="comma separated legend labels per video, e.g. 'PS5,XSX'")
    parser.add_argument("--title", default=None, help="plot title")
    parser.add_argument("-o", "--output", default=None,
                        help="save the plot to this file (png, svg, pdf, ...) instead of showing a window")
    parser.add_argument("--dpi", type=int, default=150, help="resolution of the saved image (default: 150)")


def finish_plot(figure, args):
    import matplotlib.pyplot as plt

    figure.tight_layout()
    if args.output:
        figure.savefig(args.output, dpi=args.dpi)
        print(f"saved {args.output}", file=sys.stderr)
    else:
        plt.show()


def setup_matplotlib(args):
    """Selects a non interactive backend when only saving, so no display is needed."""
    import matplotlib

    if args.output:
        matplotlib.use("Agg")


def positive_float(text):
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value
