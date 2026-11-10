package io.github.swung0x48.trdrop

import android.content.Context
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.os.ParcelFileDescriptor
import android.provider.OpenableColumns
import java.util.concurrent.atomic.AtomicBoolean

/** What to analyze and how. */
data class AnalysisRequest(
    /** a document (content://...) or a file path */
    val source: String,
    val asyncCapture: Boolean,
    val motionVideo: Boolean,
    val arrowScale: Double = 1.0,
    /** where the analysis runs: auto, vulkan (on the GPU) or ffmpeg (on the CPU) */
    val decode: String = "auto",
    /** more arguments for trdrop-pp, for tests */
    val extraArguments: List<String> = emptyList(),
)

/** Runs one analysis at a time on a background thread; the listener is called on the main thread. */
object AnalysisRunner {
    interface Listener {
        fun onProgress(frames: Long, progress: Double)
        fun onFinished(analysis: Analysis, exitCode: Int)
    }

    private val main = Handler(Looper.getMainLooper())
    private val cancelled = AtomicBoolean(false)

    @Volatile
    var running: Boolean = false
        private set
    var runningName: String = ""
        private set
    var listener: Listener? = null

    /** frames and progress of the running analysis, for a listener that attaches later */
    @Volatile
    var lastFrames: Long = 0
    @Volatile
    var lastProgress: Double = -1.0

    fun cancel() = cancelled.set(true)

    fun displayName(context: Context, source: String): String {
        if (!source.startsWith("content:")) return source.substringAfterLast('/')
        return try {
            context.contentResolver.query(Uri.parse(source), arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) cursor.getString(0) else null
            } ?: source.substringAfterLast('/')
        } catch (e: Exception) {
            source.substringAfterLast('/')
        }
    }

    fun start(context: Context, request: AnalysisRequest): Boolean {
        if (running) return false
        val application = context.applicationContext
        val analysis = Analysis.create(application)
        val name = displayName(application, request.source)
        running = true
        runningName = name
        cancelled.set(false)
        lastFrames = 0
        lastProgress = -1.0

        Thread({
            val started = System.currentTimeMillis()
            var descriptor: ParcelFileDescriptor? = null
            var exitCode = 1
            try {
                // A document may be read but not opened by name (not even through /proc/self/fd): the program
                // gets its open file descriptor
                val path = if (request.source.startsWith("content:")) {
                    descriptor = application.contentResolver.openFileDescriptor(Uri.parse(request.source), "r")
                    "fd:${descriptor!!.fd}"
                } else {
                    request.source
                }
                val arguments = ArrayList<String>()
                arguments += path
                arguments += if (request.asyncCapture) "--async-capture" else "--motion"
                arguments += listOf("-o", analysis.csv.absolutePath)
                if (request.motionVideo) {
                    arguments += listOf("--motion-video", analysis.motionVideo.absolutePath)
                    if (request.arrowScale != 1.0) arguments += listOf("--arrow-scale", request.arrowScale.toString())
                }
                if (request.decode != "auto") arguments += listOf("--decode", request.decode)
                arguments += request.extraArguments
                exitCode = NativeAnalyzer.run(arguments.toTypedArray(), analysis.log.absolutePath) { frames, progress ->
                    lastFrames = frames
                    lastProgress = progress
                    main.post { listener?.onProgress(frames, progress) }
                    !cancelled.get()
                }
            } catch (e: Throwable) {
                analysis.log.appendText("\ntrdrop-pp: error: ${e.message ?: e.javaClass.simpleName}\n")
            } finally {
                try {
                    descriptor?.close()
                } catch (e: Exception) {
                    // nothing to do about it
                }
            }
            if (exitCode != 0) analysis.motionVideo.delete()
            analysis.saveMeta(
                mapOf(
                    "name" to name,
                    "started" to started.toString(),
                    "seconds" to ((System.currentTimeMillis() - started) / 1000.0).toString(),
                    "async" to request.asyncCapture.toString(),
                    "exit" to (if (cancelled.get()) "cancelled" else exitCode.toString()),
                )
            )
            main.post {
                running = false
                listener?.onFinished(analysis, exitCode)
            }
        }, "trdrop-analysis").start()
        return true
    }
}
