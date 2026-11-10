package io.github.swung0x48.trdrop

import android.content.Context
import java.io.File
import java.util.Properties
import kotlin.math.ceil

/** The files of one analysis: a folder in the private storage of the app. */
class Analysis(val directory: File) {
    val id: String get() = directory.name
    val csv: File get() = File(directory, "frames.csv")
    val log: File get() = File(directory, "log.txt")
    val motionVideo: File get() = File(directory, "motion.mp4")
    private val metaFile: File get() = File(directory, "meta.properties")

    val meta: Properties by lazy {
        Properties().also { if (metaFile.exists()) metaFile.inputStream().use(it::load) }
    }

    val name: String get() = meta.getProperty("name", id)
    val startedAt: Long get() = meta.getProperty("started", "0").toLongOrNull() ?: 0L
    val seconds: Double get() = meta.getProperty("seconds", "0").toDoubleOrNull() ?: 0.0
    val asyncCapture: Boolean get() = meta.getProperty("async", "false") == "true"
    val succeeded: Boolean get() = meta.getProperty("exit", "") == "0"

    fun saveMeta(values: Map<String, String>) {
        values.forEach { (key, value) -> meta.setProperty(key, value) }
        metaFile.outputStream().use { meta.store(it, null) }
    }

    /** the messages of the program: progress lines end with a carriage return, they are dropped */
    fun logLines(): List<String> {
        if (!log.exists()) return emptyList()
        return log.readText().split('\n').map { it.substringAfterLast('\r').trimEnd() }.filter { it.isNotEmpty() }
    }

    companion object {
        /** in the app's folder on the shared storage if there is one: its files can be copied off the device */
        fun root(context: Context): File = File(context.getExternalFilesDir(null) ?: context.filesDir, "analyses").apply { mkdirs() }

        fun create(context: Context): Analysis {
            val directory = File(root(context), System.currentTimeMillis().toString())
            directory.mkdirs()
            return Analysis(directory)
        }

        fun open(context: Context, id: String): Analysis = Analysis(File(root(context), id))

        /** finished analyses, newest first */
        fun list(context: Context): List<Analysis> =
            (root(context).listFiles() ?: emptyArray())
                .filter { it.isDirectory && File(it, "meta.properties").exists() }
                .map(::Analysis)
                .sortedByDescending { it.startedAt }
    }
}

/** The per frame CSV of trdrop++ (the columns of the first video), NaN where a value is missing. */
class FrameData(
    val diff: FloatArray,
    val fps: FloatArray,
    val frametimeMs: FloatArray,
    val animStep: FloatArray,
    val animStepSpread: FloatArray,
    val animErrorMs: FloatArray,
    val perceivedFps: FloatArray,
    val lost: FloatArray,
) {
    val frames: Int get() = diff.size
    val hasMotion: Boolean get() = animStep.any { !it.isNaN() }
    val lostTotal: Int get() = lost.sumOf { if (it.isNaN()) 0 else it.toInt() }

    /** vN_fps with the pictures the capture lost: the sum over the last second, as smoothness.py draws it */
    fun framerateWithLost(recordedFps: Double): FloatArray {
        if (lostTotal == 0) return fps
        val window = ceil(recordedFps - 1e-9).toInt().coerceAtLeast(1)
        val total = DoubleArray(frames + 1)
        for (i in 0 until frames) total[i + 1] = total[i] + if (lost[i].isNaN()) 0.0 else lost[i].toDouble()
        return FloatArray(frames) { i -> (fps[i] + total[i + 1] - total[maxOf(0, i + 1 - window)]).toFloat() }
    }

    companion object {
        fun read(file: File): FrameData {
            val lines = file.readLines().filter { it.isNotBlank() }
            require(lines.isNotEmpty()) { "empty CSV" }
            val header = lines[0].split(',').map { it.trim() }
            val rows = lines.size - 1
            val cells = Array(rows) { lines[it + 1].split(',') }
            fun column(name: String): FloatArray {
                val index = header.indexOf("v1_$name")
                val values = FloatArray(rows) { Float.NaN }
                if (index < 0) return values
                for (row in 0 until rows) {
                    val cell = cells[row].getOrNull(index)?.trim().orEmpty()
                    if (cell.isNotEmpty()) values[row] = cell.toFloatOrNull() ?: Float.NaN
                }
                return values
            }
            return FrameData(
                column("diff"), column("fps"), column("frametime_ms"), column("anim_step"), column("anim_step_spread"),
                column("anim_error_ms"), column("perceived_fps"), column("lost"),
            )
        }
    }
}

/** The numbers of the summary lines trdrop++ prints for a video. */
class Summary(lines: List<String>) {
    private val video = lines.firstOrNull { it.startsWith("v1: ") && it.contains(" fps, ") }.orEmpty()
    private val frames = lines.firstOrNull { it.startsWith("v1: effective") }.orEmpty()
    private val motion = lines.firstOrNull { it.startsWith("v1 motion:") }.orEmpty()
    val backend: String = lines.firstOrNull { it.startsWith("backend: ") }?.removePrefix("backend: ").orEmpty()
    val motionVideo: String = lines.firstOrNull { it.startsWith("v1 motion video:") }?.substringAfter('(')?.substringBeforeLast(')').orEmpty()
    val speed: String = lines.firstOrNull { it.startsWith("analyzed ") }?.substringBefore(" |").orEmpty()
    val note: String = lines.firstOrNull { it.startsWith("v1 note:") }?.removePrefix("v1 note: ").orEmpty()
    val error: String = lines.firstOrNull { it.contains("trdrop-pp: error:") }?.substringAfter("error: ").orEmpty()

    private fun number(text: String, pattern: String): Double? = Regex(pattern).find(text)?.groupValues?.get(1)?.toDoubleOrNull()

    /** e.g. 3840x2160, hevc */
    val format: String = Regex("""\((\d+x\d+, [^,]+),""").find(video)?.groupValues?.get(1).orEmpty()
    val recordedFps: Double = number(video, """, ([0-9.]+) fps,""") ?: 60.0
    val effectiveFps = number(frames, """effective ([0-9.]+) fps""")
    val newFrames = number(frames, """new (\d+)""")
    val duplicateFrames = number(frames, """duplicate (\d+)""")
    val tornFrames = number(frames, """torn (\d+)""")
    val fpsAverage = number(frames, """fps avg ([0-9.]+)""")
    val fpsMin = number(frames, """fps avg [0-9.]+ min ([0-9.]+)""")
    val fpsMax = number(frames, """ max ([0-9.]+) \|""")
    val frametimeAverage = number(frames, """frametime avg ([0-9.]+)""")
    val frametimeMax = number(frames, """frametime avg [0-9.]+ ms max ([0-9.]+)""")
    val measuredShare = number(motion, """\(([0-9.]+)%\)""")
    val motionFps = number(motion, """([0-9.]+) new frames/s""")
    val perceivedFps = number(motion, """perceived smoothness ([0-9.]+)""")
    val perceivedLow = number(motion, """10% low ([0-9.]+)""")
    val errorMedian = number(motion, """animation error median ([0-9.]+)""")
    val errorAverage = number(motion, """ms avg ([0-9.]+) ms p95""")
    val errorP95 = number(motion, """p95 ([0-9.]+)""")
    val errorMax = number(motion, """p95 [0-9.]+ ms max ([0-9.]+)""")
    val spread = number(motion, """on a step: ([0-9.]+)""")
    val lostPictures = number(motion, """(\d+) lost pictures""")
    val hasMotion: Boolean = perceivedFps != null
}
