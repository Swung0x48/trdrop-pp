package io.github.swung0x48.trdrop

import android.content.ContentValues
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.View
import android.view.ViewGroup
import android.widget.GridLayout
import android.widget.LinearLayout
import android.widget.MediaController
import android.widget.TextView
import android.widget.Toast
import android.widget.VideoView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.card.MaterialCardView
import io.github.swung0x48.trdrop.charts.ChartData
import io.github.swung0x48.trdrop.charts.ChartView
import io.github.swung0x48.trdrop.charts.Charts
import java.io.File

/** The result of an analysis: the numbers of the summary, the charts and the motion video. */
class ResultActivity : AppCompatActivity() {
    private lateinit var analysis: Analysis
    private val density by lazy { resources.displayMetrics.density }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        setContentView(R.layout.activity_result)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.root)) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            WindowInsetsCompat.CONSUMED
        }
        analysis = Analysis.open(this, intent.getStringExtra("id") ?: return finish())
        val toolbar = findViewById<MaterialToolbar>(R.id.toolbar)
        toolbar.title = analysis.name
        toolbar.setNavigationOnClickListener { finish() }

        val summary = Summary(analysis.logLines())
        findViewById<TextView>(R.id.subtitle).text = listOf(summary.format, String.format("%.0f fps", summary.recordedFps), summary.speed)
            .filter { it.isNotEmpty() }.joinToString(" · ")
        showTiles(summary)
        if (summary.note.isNotEmpty() || !summary.hasMotion) {
            findViewById<TextView>(R.id.note).apply {
                text = if (summary.hasMotion) getString(R.string.note_async) else getString(R.string.not_measurable)
                visibility = View.VISIBLE
            }
        }
        findViewById<TextView>(R.id.log).text = analysis.logLines().joinToString("\n")
        findViewById<View>(R.id.share_csv).setOnClickListener { share(analysis.csv, "text/csv") }
        showVideo(summary)

        // the CSV of a long video takes a moment to read
        Thread {
            val frames = try {
                FrameData.read(analysis.csv)
            } catch (e: Exception) {
                null
            }
            runOnUiThread { if (frames != null && !isDestroyed) showCharts(frames, summary) }
        }.start()
    }

    private fun showTiles(summary: Summary) {
        val tiles = findViewById<GridLayout>(R.id.tiles)
        fun tile(label: Int, value: String, detail: String) {
            val card = MaterialCardView(this, null, com.google.android.material.R.attr.materialCardViewFilledStyle).apply {
                setCardBackgroundColor(getColor(R.color.chart_surface))
                radius = 16 * density
                layoutParams = GridLayout.LayoutParams(GridLayout.spec(GridLayout.UNDEFINED, 1f), GridLayout.spec(GridLayout.UNDEFINED, 1f)).apply {
                    width = 0
                    val margin = (4 * density).toInt()
                    setMargins(margin, margin, margin, margin)
                }
            }
            val column = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                val padding = (14 * density).toInt()
                setPadding(padding, (12 * density).toInt(), padding, (12 * density).toInt())
            }
            column.addView(TextView(this).apply {
                setText(label)
                textSize = 13f
                setTextColor(getColor(R.color.chart_text_secondary))
            })
            column.addView(TextView(this).apply {
                text = value
                textSize = 24f
                setTextColor(getColor(R.color.chart_text_primary))
            })
            column.addView(TextView(this).apply {
                text = detail
                textSize = 12f
                setTextColor(getColor(R.color.chart_text_secondary))
            })
            card.addView(column)
            tiles.addView(card)
        }
        // with lost pictures counted in, the framerate is the one of the motion line
        val withLost = (summary.lostPictures ?: 0.0) > 0.0
        val framerate = if (withLost) summary.motionFps else summary.fpsAverage ?: summary.effectiveFps
        if (framerate != null) {
            tile(R.string.stat_framerate, getString(R.string.value_fps, framerate), when {
                withLost -> getString(R.string.stat_framerate_recorded, summary.fpsAverage ?: summary.effectiveFps ?: 0.0)
                summary.fpsMin != null && summary.fpsMax != null -> getString(R.string.stat_framerate_detail, summary.fpsMin, summary.fpsMax)
                else -> ""
            })
        }
        summary.perceivedFps?.let {
            tile(R.string.stat_perceived, getString(R.string.value_fps, it),
                 summary.perceivedLow?.let { low -> getString(R.string.stat_perceived_detail, low) } ?: "")
        }
        summary.frametimeAverage?.let {
            tile(R.string.stat_frametime, getString(R.string.value_ms, it),
                 summary.frametimeMax?.let { max -> getString(R.string.stat_frametime_detail, max) } ?: "")
        }
        summary.errorAverage?.let {
            tile(R.string.stat_error, getString(R.string.value_ms, it),
                 getString(R.string.stat_error_detail, summary.errorMedian ?: 0.0, summary.errorP95 ?: 0.0, summary.errorMax ?: 0.0))
        }
        summary.newFrames?.let {
            tile(R.string.stat_frames, it.toInt().toString(),
                 getString(R.string.stat_frames_detail, (summary.duplicateFrames ?: 0.0).toInt(), (summary.tornFrames ?: 0.0).toInt()))
        }
        summary.lostPictures?.let {
            tile(R.string.stat_lost, it.toInt().toString(), getString(R.string.stat_lost_detail))
        }
    }

    private fun showCharts(frames: FrameData, summary: Summary) {
        val container = findViewById<LinearLayout>(R.id.charts)
        val charts = Charts(this, frames, summary.recordedFps)
        fun card(title: Int, data: ChartData) {
            val card = MaterialCardView(this, null, com.google.android.material.R.attr.materialCardViewFilledStyle).apply {
                setCardBackgroundColor(getColor(R.color.chart_surface))
                radius = 16 * density
                layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                    topMargin = (12 * density).toInt()
                    marginStart = (4 * density).toInt()
                    marginEnd = (4 * density).toInt()
                }
            }
            val column = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                val padding = (16 * density).toInt()
                setPadding(padding, padding, padding, (8 * density).toInt())
            }
            column.addView(TextView(this).apply {
                setText(title)
                textSize = 16f
                setTypeface(typeface, android.graphics.Typeface.BOLD)
                setTextColor(getColor(R.color.chart_text_primary))
            })
            column.addView(ChartView(this).apply {
                this.data = data
                layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                    topMargin = (8 * density).toInt()
                }
            })
            card.addView(column)
            container.addView(card)
        }
        card(if (frames.hasMotion) R.string.chart_framerate else R.string.chart_framerate_plain, charts.framerate())
        card(R.string.chart_frametimes, charts.frametimes())
        if (frames.animErrorMs.any { !it.isNaN() }) {
            card(R.string.chart_error, charts.animationError())
            card(R.string.chart_error_histogram, charts.animationErrorHistogram())
        }
        container.addView(TextView(this).apply {
            setText(R.string.chart_hint)
            textSize = 12f
            setTextColor(getColor(R.color.chart_text_secondary))
            setPadding((8 * density).toInt(), (8 * density).toInt(), 0, 0)
        })
    }

    private fun showVideo(summary: Summary) {
        val file = analysis.motionVideo
        if (!file.exists() || file.length() == 0L) return
        findViewById<View>(R.id.video_card).visibility = View.VISIBLE
        findViewById<TextView>(R.id.video_info).text = summary.motionVideo
        val video = findViewById<VideoView>(R.id.video)
        val controller = MediaController(this)
        controller.setAnchorView(findViewById(R.id.video_frame))
        video.setMediaController(controller)
        // a VideoView without a height never gets a surface, and without a surface the video isn't opened
        video.post {
            video.layoutParams = video.layoutParams.apply { height = video.width * 9 / 16 }
            video.setVideoPath(file.absolutePath)
        }
        video.setOnPreparedListener { player ->
            // the height follows the shape of the video
            if (player.videoWidth > 0) {
                video.layoutParams = video.layoutParams.apply { height = video.width * player.videoHeight / player.videoWidth }
            }
            player.isLooping = true
            video.seekTo(1)
        }
        findViewById<View>(R.id.share_video).setOnClickListener { share(file, "video/mp4") }
        val save = findViewById<View>(R.id.save_video)
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) save.visibility = View.GONE
        save.setOnClickListener { saveToGallery(file) }
    }

    private fun share(file: File, type: String) {
        val uri = FileProvider.getUriForFile(this, "$packageName.files", file)
        val send = Intent(Intent.ACTION_SEND).setType(type).putExtra(Intent.EXTRA_STREAM, uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        startActivity(Intent.createChooser(send, null))
    }

    private fun saveToGallery(file: File) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return
        val name = analysis.name.substringBeforeLast('.') + "_motion.mp4"
        val values = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, name)
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_MOVIES + "/trdrop")
        }
        Thread {
            val saved = try {
                val uri = contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
                if (uri != null) {
                    contentResolver.openOutputStream(uri)?.use { output -> file.inputStream().use { it.copyTo(output) } }
                }
                uri != null
            } catch (e: Exception) {
                false
            }
            runOnUiThread { if (saved) Toast.makeText(this, R.string.saved_video, Toast.LENGTH_SHORT).show() }
        }.start()
    }
}
