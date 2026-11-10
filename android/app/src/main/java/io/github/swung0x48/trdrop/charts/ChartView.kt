package io.github.swung0x48.trdrop.charts

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import androidx.core.content.ContextCompat
import io.github.swung0x48.trdrop.R
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/** What a ChartView shows. Values are NaN where there is nothing to draw. */
class ChartData(val xLabel: String, val yLabel: String) {
    class Line(val label: String, val color: Int, val x: FloatArray, val y: FloatArray, val width: Float = 2f)

    /** dots with a thin stem to zero */
    class Points(val label: String, val color: Int, val x: FloatArray, val y: FloatArray)

    /** a band of +-half around zero */
    class Band(val label: String, val color: Int, val x: FloatArray, val half: FloatArray)

    /** bars from left to right (in x values), with a text under each if there are few */
    class Bars(val color: Int, val left: FloatArray, val right: FloatArray, val value: FloatArray, val names: List<String>? = null)

    val lines = ArrayList<Line>()
    var points: Points? = null
    var band: Band? = null
    var bars: Bars? = null
    var xMin = 0f
    var xMax = 1f
    var yMin = 0f
    var yMax = 1f
    /** the x axis is time: it can be zoomed with two fingers */
    var zoomable = false
    var xUnit = ""
    var yUnit = ""
}

/**
 * A chart drawn on a Canvas: lines, points with stems, a band and bars over shared axes, with a legend.
 * Touch: a finger shows the values at its position, two fingers zoom the x axis, a double tap resets it.
 */
class ChartView @JvmOverloads constructor(context: Context, attributes: AttributeSet? = null) : View(context, attributes) {
    var data: ChartData? = null
        set(value) {
            field = value
            viewMin = value?.xMin ?: 0f
            viewMax = value?.xMax ?: 1f
            cursor = Float.NaN
            invalidate()
        }

    private val density = resources.displayMetrics.density
    private val scaledDensity = resources.displayMetrics.density * resources.configuration.fontScale
    private val textPrimary = ContextCompat.getColor(context, R.color.chart_text_primary)
    private val textSecondary = ContextCompat.getColor(context, R.color.chart_text_secondary)
    private val gridColor = ContextCompat.getColor(context, R.color.chart_grid)
    private val axisColor = ContextCompat.getColor(context, R.color.chart_axis)
    private val surface = ContextCompat.getColor(context, R.color.chart_surface)

    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 11f * scaledDensity; color = textSecondary }
    private val legendText = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 12f * scaledDensity; color = textPrimary }
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val plot = RectF()
    private val path = Path()

    private var viewMin = 0f
    private var viewMax = 1f
    /** x value under the finger, NaN: none */
    private var cursor = Float.NaN

    private val scaleDetector = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean {
            val chart = data ?: return false
            if (!chart.zoomable || plot.width() <= 0f) return false
            val focus = viewMin + (detector.focusX - plot.left) / plot.width() * (viewMax - viewMin)
            val span = ((viewMax - viewMin) / detector.scaleFactor).coerceIn((chart.xMax - chart.xMin) / 200f, chart.xMax - chart.xMin)
            val share = ((detector.focusX - plot.left) / plot.width()).coerceIn(0f, 1f)
            viewMin = (focus - share * span).coerceIn(chart.xMin, chart.xMax - span)
            viewMax = viewMin + span
            cursor = Float.NaN
            invalidate()
            return true
        }
    })

    private val gestureDetector = GestureDetector(context, object : GestureDetector.SimpleOnGestureListener() {
        override fun onDoubleTap(event: MotionEvent): Boolean {
            val chart = data ?: return false
            viewMin = chart.xMin
            viewMax = chart.xMax
            cursor = Float.NaN
            invalidate()
            return true
        }
    })

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val width = MeasureSpec.getSize(widthMeasureSpec)
        val wanted = (230f * density).toInt()
        setMeasuredDimension(width, resolveSize(wanted, heightMeasureSpec))
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val chart = data ?: return false
        scaleDetector.onTouchEvent(event)
        gestureDetector.onTouchEvent(event)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                if (event.pointerCount == 1 && !scaleDetector.isInProgress && plot.width() > 0f) {
                    val share = ((event.x - plot.left) / plot.width()).coerceIn(0f, 1f)
                    cursor = viewMin + share * (viewMax - viewMin)
                    // the chart keeps the finger once it moves sideways, the page scrolls otherwise
                    if (event.actionMasked == MotionEvent.ACTION_MOVE && event.historySize > 0
                        && abs(event.x - event.getHistoricalX(0)) > abs(event.y - event.getHistoricalY(0))
                    ) {
                        parent?.requestDisallowInterceptTouchEvent(true)
                    }
                    invalidate()
                } else if (event.pointerCount > 1) {
                    parent?.requestDisallowInterceptTouchEvent(true)
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> parent?.requestDisallowInterceptTouchEvent(false)
        }
        return chart.zoomable || chart.lines.isNotEmpty() || chart.points != null || chart.bars != null
    }

    private fun niceStep(range: Float, wanted: Int): Float {
        if (range <= 0f) return 1f
        val raw = range / wanted
        val magnitude = 10.0.pow(floor(log10(raw.toDouble()))).toFloat()
        val fraction = raw / magnitude
        return magnitude * when {
            fraction <= 1f -> 1f
            fraction <= 2f -> 2f
            fraction <= 5f -> 5f
            else -> 10f
        }
    }

    private fun label(value: Float, step: Float): String = when {
        step >= 1f -> String.format("%.0f", value)
        step >= 0.1f -> String.format("%.1f", value)
        else -> String.format("%.2f", value)
    }

    override fun onDraw(canvas: Canvas) {
        val chart = data ?: return
        val width = width.toFloat()
        val height = height.toFloat()
        val lineHeight = text.textSize * 1.35f

        // legend: a row (or more) of swatches with labels, then the readout of the cursor or the unit of the y axis
        var top = 2f * density
        val entries = ArrayList<Pair<Int, String>>()
        chart.lines.forEach { entries += it.color to it.label }
        chart.points?.let { entries += it.color to it.label }
        chart.band?.let { entries += (it.color and 0x00FFFFFF or 0x59000000) to it.label }
        if (entries.size >= 2 || chart.band != null) {
            var x = 0f
            var baseline = top + legendText.textSize
            for ((color, name) in entries) {
                val entryWidth = 14f * density + legendText.measureText(name) + 14f * density
                if (x > 0f && x + entryWidth > width) {
                    x = 0f
                    baseline += legendText.textSize * 1.5f
                }
                fill.color = color
                canvas.drawRoundRect(x, baseline - 9f * density, x + 9f * density, baseline, 2f * density, 2f * density, fill)
                canvas.drawText(name, x + 14f * density, baseline - 1f * density, legendText)
                x += entryWidth
            }
            top = baseline + 8f * density
        }
        text.color = textSecondary
        text.textAlign = Paint.Align.LEFT
        canvas.drawText(readout(chart) ?: chart.yLabel, 0f, top + text.textSize, text)
        top += lineHeight + 4f * density

        // plot area
        val yStep = niceStep(chart.yMax - chart.yMin, 4)
        var left = 0f
        var tick = ceil(chart.yMin / yStep) * yStep
        while (tick <= chart.yMax + yStep * 0.001f) {
            left = max(left, text.measureText(label(tick, yStep)))
            tick += yStep
        }
        plot.set(left + 8f * density, top, width - 10f * density, height - lineHeight * 2f - 6f * density)
        if (plot.width() <= 0f || plot.height() <= 0f) return
        val xSpan = max(1e-9f, viewMax - viewMin)
        val ySpan = max(1e-9f, chart.yMax - chart.yMin)
        fun px(x: Float) = plot.left + (x - viewMin) / xSpan * plot.width()
        fun py(y: Float) = plot.bottom - (y - chart.yMin) / ySpan * plot.height()

        // grid and y labels
        stroke.strokeWidth = 1f * density
        stroke.color = gridColor
        text.textAlign = Paint.Align.RIGHT
        tick = ceil(chart.yMin / yStep) * yStep
        while (tick <= chart.yMax + yStep * 0.001f) {
            val y = py(tick)
            canvas.drawLine(plot.left, y, plot.right, y, stroke)
            canvas.drawText(label(tick, yStep), plot.left - 6f * density, y + text.textSize * 0.35f, text)
            tick += yStep
        }

        canvas.save()
        canvas.clipRect(plot)
        chart.band?.let { drawBand(canvas, it, ::px, ::py) }
        chart.bars?.let { drawBars(canvas, it, ::px, ::py) }
        chart.points?.let { drawPoints(canvas, it, ::px, ::py) }
        for (line in chart.lines) drawLine(canvas, line, ::px, ::py)
        if (!cursor.isNaN() && chart.bars == null) {
            stroke.color = axisColor
            stroke.strokeWidth = 1f * density
            canvas.drawLine(px(cursor), plot.top, px(cursor), plot.bottom, stroke)
        }
        canvas.restore()

        // axis, x labels and title
        stroke.color = axisColor
        stroke.strokeWidth = 1f * density
        val zero = if (chart.yMin < 0f && chart.yMax > 0f) py(0f) else plot.bottom
        canvas.drawLine(plot.left, zero, plot.right, zero, stroke)
        text.textAlign = Paint.Align.CENTER
        val names = chart.bars?.names
        if (names != null) {
            val bars = chart.bars!!
            // as many names as fit
            val every = max(1, ceil(names.size * (text.measureText("00.0") + 6f * density) / plot.width()).toInt())
            for (i in names.indices step every) {
                canvas.drawText(names[i], px((bars.left[i] + bars.right[i]) / 2f), plot.bottom + lineHeight, text)
            }
        } else {
            val xStep = niceStep(xSpan, max(2, (plot.width() / (64f * density)).toInt()))
            var x = ceil(viewMin / xStep) * xStep
            while (x <= viewMax + xStep * 0.001f) {
                canvas.drawText(label(x, xStep), px(x).coerceIn(plot.left + 8f * density, plot.right - 8f * density), plot.bottom + lineHeight, text)
                x += xStep
            }
        }
        canvas.drawText(chart.xLabel, plot.centerX(), plot.bottom + lineHeight * 2f, text)
    }

    /** the values under the finger */
    private fun readout(chart: ChartData): String? {
        if (cursor.isNaN()) return null
        chart.bars?.let { bars ->
            for (i in bars.left.indices) {
                if (cursor >= bars.left[i] && cursor < bars.right[i]) {
                    val name = bars.names?.get(i) ?: String.format("%.1f – %.1f", bars.left[i], bars.right[i])
                    val value = bars.value[i]
                    return "$name ${chart.xUnit}: ${String.format(if (value == floor(value)) "%.0f" else "%.1f", value)} ${chart.yUnit}"
                }
            }
            return null
        }
        val parts = ArrayList<String>()
        parts += String.format("%.2f %s", cursor, chart.xUnit)
        for (line in chart.lines) {
            val index = nearest(line.x, cursor)
            if (index >= 0 && !line.y[index].isNaN()) parts += String.format("%.1f", line.y[index])
        }
        chart.points?.let { points ->
            val index = nearest(points.x, cursor)
            if (index >= 0) parts += String.format("%+.1f %s", points.y[index], chart.yUnit)
        }
        return parts.joinToString("   ")
    }

    private fun nearest(x: FloatArray, value: Float): Int {
        if (x.isEmpty()) return -1
        var low = 0
        var high = x.size - 1
        while (low < high) {
            val middle = (low + high) / 2
            if (x[middle] < value) low = middle + 1 else high = middle
        }
        return if (low > 0 && abs(x[low - 1] - value) < abs(x[low] - value)) low - 1 else low
    }

    private fun drawLine(canvas: Canvas, line: ChartData.Line, px: (Float) -> Float, py: (Float) -> Float) {
        stroke.color = line.color
        stroke.strokeWidth = line.width * density
        path.rewind()
        var open = false
        // many values per pixel: one vertical stroke from the lowest to the highest of a column
        var column = Int.MIN_VALUE
        var low = 0f
        var high = 0f
        var last = 0f
        for (i in line.x.indices) {
            val x = line.x[i]
            if (x < viewMin - 1f || x > viewMax + 1f) continue
            val y = line.y[i]
            if (y.isNaN()) {
                open = false
                column = Int.MIN_VALUE
                continue
            }
            val sx = px(x)
            val sy = py(y)
            val current = sx.toInt()
            if (current == column) {
                low = min(low, sy)
                high = max(high, sy)
                last = sy
                continue
            }
            if (column != Int.MIN_VALUE && high > low) {
                path.lineTo(column.toFloat(), low)
                path.lineTo(column.toFloat(), high)
                path.lineTo(column.toFloat(), last)
            }
            if (open) path.lineTo(sx, sy) else path.moveTo(sx, sy)
            open = true
            column = current
            low = sy
            high = sy
            last = sy
        }
        canvas.drawPath(path, stroke)
    }

    private fun drawPoints(canvas: Canvas, points: ChartData.Points, px: (Float) -> Float, py: (Float) -> Float) {
        val zero = py(0f)
        stroke.color = points.color and 0x00FFFFFF or 0x66000000
        stroke.strokeWidth = 0.75f * density
        fill.color = points.color
        val radius = 1.5f * density
        for (i in points.x.indices) {
            val x = points.x[i]
            if (x < viewMin || x > viewMax || points.y[i].isNaN()) continue
            val sx = px(x)
            val sy = py(points.y[i])
            canvas.drawLine(sx, zero, sx, sy, stroke)
            canvas.drawCircle(sx, sy, radius, fill)
        }
    }

    private fun drawBand(canvas: Canvas, band: ChartData.Band, px: (Float) -> Float, py: (Float) -> Float) {
        fill.color = band.color and 0x00FFFFFF or 0x33000000
        var start = -1
        fun flush(end: Int) {
            if (start < 0 || end - start < 2) return
            path.rewind()
            path.moveTo(px(band.x[start]), py(band.half[start]))
            for (i in start + 1 until end) path.lineTo(px(band.x[i]), py(band.half[i]))
            for (i in end - 1 downTo start) path.lineTo(px(band.x[i]), py(-band.half[i]))
            path.close()
            canvas.drawPath(path, fill)
        }
        for (i in band.x.indices) {
            if (band.half[i].isNaN()) {
                flush(i)
                start = -1
            } else if (start < 0) {
                start = i
            }
        }
        flush(band.x.size)
    }

    private fun drawBars(canvas: Canvas, bars: ChartData.Bars, px: (Float) -> Float, py: (Float) -> Float) {
        val zero = py(0f)
        for (i in bars.left.indices) {
            if (bars.value[i] <= 0f) continue
            val left = px(bars.left[i])
            val right = px(bars.right[i])
            // a gap of the surface between neighbours
            val gap = if (right - left > 4f * density) 1f * density else 0f
            val under = !cursor.isNaN() && cursor >= bars.left[i] && cursor < bars.right[i]
            fill.color = if (under) bars.color else bars.color and 0x00FFFFFF or 0xD9000000.toInt()
            val radius = min(4f * density, (right - left) / 2f)
            canvas.drawRoundRect(left + gap, py(bars.value[i]), right - gap, zero + radius, radius, radius, fill)
        }
        // the rounding belongs to the data end only: the part below the axis is covered
        fill.color = surface
        canvas.drawRect(plot.left, zero, plot.right, plot.bottom + 6f * density, fill)
    }
}
