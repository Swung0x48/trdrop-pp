package io.github.swung0x48.trdrop.charts

import android.content.Context
import androidx.core.content.ContextCompat
import io.github.swung0x48.trdrop.FrameData
import io.github.swung0x48.trdrop.R
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * The charts of the plot scripts (scripts/framerate.py, frametime_histogram.py, smoothness.py), built from the
 * per frame CSV for a ChartView.
 */
class Charts(private val context: Context, private val frames: FrameData, private val recordedFps: Double) {
    private val framerateColor = ContextCompat.getColor(context, R.color.chart_series_1)
    private val perceivedColor = ContextCompat.getColor(context, R.color.chart_series_2)
    private val seconds = FloatArray(frames.frames) { (it / recordedFps).toFloat() }
    private val interval = (1000.0 / recordedFps).toFloat()

    private fun percentile(sorted: FloatArray, p: Double): Float {
        if (sorted.isEmpty()) return Float.NaN
        val position = (sorted.size - 1) * p / 100.0
        val lower = position.toInt()
        val upper = minOf(lower + 1, sorted.size - 1)
        return (sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower)).toFloat()
    }

    /** framerate over time, with the perceived smoothness if the motion was measured */
    fun framerate(): ChartData {
        val chart = ChartData(context.getString(R.string.axis_time), context.getString(R.string.axis_fps))
        // the framerate window is still filling up in the first second
        val skip = ceil(recordedFps - 1e-9).toInt()
        val rate = frames.framerateWithLost(recordedFps)
        val shown = FloatArray(frames.frames) { if (it < skip) Float.NaN else rate[it] }
        val lost = frames.lostTotal
        chart.lines += ChartData.Line(
            if (lost > 0) context.getString(R.string.series_framerate_lost, lost) else context.getString(R.string.series_framerate),
            framerateColor, seconds, shown, 1.5f,
        )
        var highest = shown.filter { !it.isNaN() }.maxOrNull() ?: 1f
        if (frames.hasMotion) {
            val perceived = FloatArray(frames.frames) { if (it < skip) Float.NaN else frames.perceivedFps[it] }
            chart.lines += ChartData.Line(context.getString(R.string.series_perceived), perceivedColor, seconds, perceived, 2f)
            highest = max(highest, perceived.filter { !it.isNaN() }.maxOrNull() ?: 0f)
        }
        chart.xMin = if (skip < frames.frames) seconds[skip] else 0f
        chart.xMax = seconds.lastOrNull() ?: 1f
        chart.yMin = 0f
        chart.yMax = highest * 1.1f
        chart.zoomable = true
        chart.xUnit = "s"
        return chart
    }

    /** how long the new frames stayed on screen: the share of the frames per frametime */
    fun frametimes(): ChartData {
        val chart = ChartData(context.getString(R.string.axis_frametime), context.getString(R.string.axis_share))
        // frametimes are multiples of the interval of the recording
        val counts = HashMap<Int, Int>()
        var total = 0
        for (i in 0 until frames.frames) {
            val diff = frames.diff[i]
            if (diff.isNaN() || diff <= 0f || frames.frametimeMs[i].isNaN() || frames.frametimeMs[i] <= 0f) continue
            val intervals = (frames.frametimeMs[i] / interval).roundToInt().coerceAtLeast(1)
            counts[intervals] = (counts[intervals] ?: 0) + 1
            total += 1
        }
        val keys = counts.keys.sorted()
        chart.bars = ChartData.Bars(
            framerateColor,
            FloatArray(keys.size) { it.toFloat() },
            FloatArray(keys.size) { it + 1f },
            FloatArray(keys.size) { 100f * counts[keys[it]]!! / max(1, total) },
            keys.map { String.format("%.1f", it * interval) },
        )
        chart.xMin = 0f
        chart.xMax = max(1, keys.size).toFloat()
        chart.yMin = 0f
        chart.yMax = (chart.bars!!.value.maxOrNull() ?: 1f) * 1.1f
        chart.xUnit = "ms"
        chart.yUnit = "%"
        return chart
    }

    /** range of the animation error axis: the 99th percentile, at least two intervals */
    private fun errorLimit(): Float {
        val absolute = frames.animErrorMs.filter { !it.isNaN() }.map { abs(it) }.sorted().toFloatArray()
        val limit = max(2f * interval, if (absolute.isEmpty()) 0f else percentile(absolute, 99.0) * 1.15f)
        return ceil(limit / 5f) * 5f
    }

    /** the animation error of every new frame over time, with the spread of the votes as a band */
    fun animationError(): ChartData {
        val chart = ChartData(context.getString(R.string.axis_time), context.getString(R.string.axis_error))
        val limit = errorLimit()
        val indices = (0 until frames.frames).filter { !frames.animErrorMs[it].isNaN() }
        val x = FloatArray(indices.size) { seconds[indices[it]] }
        chart.points = ChartData.Points(
            context.getString(R.string.series_error), framerateColor, x,
            FloatArray(indices.size) { frames.animErrorMs[indices[it]].coerceIn(-limit, limit) },
        )
        // how much the moving parts of the picture disagree about a step: errors inside the band say little
        val spread = FloatArray(indices.size) { frames.animStepSpread[indices[it]] * interval }
        if (spread.isNotEmpty() && spread.none { it.isNaN() }) {
            val smooth = FloatArray(spread.size) { k ->
                val window = spread.copyOfRange(max(0, k - 7), minOf(spread.size, k + 8)).sortedArray()
                // no band over stretches without measured motion (longer than half a second)
                if (k > 0 && x[k] - x[k - 1] > 0.5f) Float.NaN else window[window.size / 2].coerceAtMost(limit)
            }
            chart.band = ChartData.Band(context.getString(R.string.series_spread), framerateColor, x, smooth)
        }
        chart.xMin = 0f
        chart.xMax = seconds.lastOrNull() ?: 1f
        chart.yMin = -limit
        chart.yMax = limit
        chart.zoomable = true
        chart.xUnit = "s"
        chart.yUnit = "ms"
        return chart
    }

    /** how often which animation error occurs (values beyond the range are in the outer bars) */
    fun animationErrorHistogram(): ChartData {
        val chart = ChartData(context.getString(R.string.axis_error), context.getString(R.string.axis_frames))
        val limit = errorLimit()
        val bins = 60
        val width = 2f * limit / bins
        val counts = FloatArray(bins)
        for (error in frames.animErrorMs) {
            if (error.isNaN()) continue
            val bin = ((error.coerceIn(-limit, limit - 1e-4f) + limit) / width).toInt().coerceIn(0, bins - 1)
            counts[bin] += 1f
        }
        chart.bars = ChartData.Bars(framerateColor, FloatArray(bins) { -limit + it * width }, FloatArray(bins) { -limit + (it + 1) * width }, counts)
        chart.xMin = -limit
        chart.xMax = limit
        chart.yMin = 0f
        chart.yMax = (counts.maxOrNull() ?: 1f).coerceAtLeast(1f) * 1.1f
        chart.xUnit = "ms"
        chart.yUnit = context.getString(R.string.unit_frames)
        return chart
    }
}
